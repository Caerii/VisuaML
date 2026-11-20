"""Replay greedy language generation and two exact tool invocations."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import torch

from fork_divergence import first_divergence
from frozen_call_checkpoint import canonical, digest
from internal_tool import prepare_program
from replay_internal_tool import oracle
from synthesize_attention_vm import OUT as EXECUTOR
from two_call_controller import IDS, VOCAB, OUT, encode, reserved


class Session:
    def __init__(self):
        self.spec = json.loads((OUT / "holdout.json").read_text())
        start = perf_counter_ns()
        for name, path in (("controller", OUT / "controller.pt"), ("executor", EXECUTOR / "executor.pt")):
            assert digest(path.read_bytes()) == self.spec[name+"_sha256"]
            setattr(self, name, torch.jit.load(str(path)))
        self.cold_load_ns = perf_counter_ns()-start
        start = perf_counter_ns()
        self.programs = {}
        for key, code in self.spec["programs"].items():
            image = prepare_program(code)
            if any(op == 0 and operand != 0 for op, _, operand in code):
                raise ValueError("HALT requires the canonical zero immediate")
            self.programs[key] = image
        self.prepare_ns = perf_counter_ns()-start

    def generate(self, prefix):
        tokens = []
        for _ in range(6):
            output = self.controller(torch.tensor([encode(prefix)]))
            word = VOCAB[int(output[0, -1].argmax())]
            tokens.append(word)
            prefix += " " + word
            if word == ".":
                return tokens, prefix
        return tokens, prefix

    def run(self, index, mode):
        fixture = self.spec["prompts"][index]
        prefix = fixture["text"]
        events, timings = [], []
        for turn in range(3):
            start = perf_counter_ns()
            tokens, prefix = self.generate(prefix)
            timings.append(dict(phase="language_decode", elapsed_ns=perf_counter_ns()-start))
            events.append(dict(kind="generated", tokens=tokens))
            if tokens == ["stop", "."]:
                return dict(events=events, outcome="stopped"), timings
            if len(tokens) == 3 and tokens[0] == "answer" and tokens[1].isdecimal() and tokens[-1] == ".":
                return dict(events=events, outcome="answered", answer=int(tokens[1])), timings
            if (len(tokens) != 5 or tokens[0] != "call" or tokens[1] not in fixture["slots"]
                or not all(token.isdecimal() for token in tokens[2:4]) or tokens[-1] != "."):
                return dict(events=events, outcome="protocol_error"), timings

            role = fixture["slots"][tokens[1]]
            value, count = map(int, tokens[2:4])
            arguments = [value, count]
            event = dict(kind="tool", slot=tokens[1], role=role, arguments=arguments, trace=[])
            start = perf_counter_ns()
            if not 0 <= value <= 31 or not 1 <= count <= 9 or (role == "offset" and count != 1):
                feedback = "invalid"
            elif mode == "disabled" and turn == 0:
                feedback = "unavailable"
            else:
                number, trace = self.executor(self.programs[role], torch.tensor(arguments))
                live = trace[:, 1].tolist()
                event["trace"] = trace[:live.index(0)+1].tolist() if 0 in live else trace.tolist()
                if live[-1]:
                    feedback = "exhausted"
                else:
                    numeric = int(number)
                    event["computed"] = numeric
                    if mode == "substitute" and turn == 0:
                        numeric += 2
                    feedback = f"result {numeric}" if str(numeric) in IDS else "invalid"
            timings.append(dict(phase="tool_validation_and_trace", elapsed_ns=perf_counter_ns()-start))
            event["feedback"] = feedback
            prefix += " " + feedback + " assistant"
            events.append(event)
        return dict(events=events, outcome="protocol_error"), timings


def event_difference(reference, candidate):
    index = first_divergence(reference["events"], candidate["events"])
    return None if index is None else dict(event=index+1, expected=reference["events"][index:index+1],
                                            actual=candidate["events"][index:index+1])


def test_case(session, index):
    fixture = session.spec["prompts"][index]
    seed, repeat = fixture["seed"], fixture["repeat"]
    assert reserved(seed, repeat)
    runs, raw = {}, {}
    for mode in ("normal", "disabled", "substitute"):
        runs[mode], raw[mode] = session.run(index, mode)
    normal, disabled, substituted = (runs[k] for k in ("normal", "disabled", "substitute"))
    for mode, run in runs.items():
        # Score the protocol, binding, executor, and interpretation separately.
        assert run["outcome"] != "protocol_error", (index, mode, run)
        tools = [event for event in run["events"] if event["kind"] == "tool"]
        assert tools[0]["role"] == "offset" and tools[0]["arguments"] == [seed, 1], (index, mode, tools)
        if mode == "disabled":
            assert len(tools) == 1 and tools[0]["feedback"] == "unavailable"
            assert run["outcome"] == "stopped"
            continue
        assert len(tools) == 2 and tools[1]["role"] == "boost", (index, mode, tools)
        assert tools[0]["computed"] == seed + 4
        first = seed + 4 + (2 if mode == "substitute" else 0)
        assert tools[1]["arguments"] == [first, repeat], (index, mode, tools)
        for event in tools:
            if event["trace"]:
                assert event["trace"] == oracle(session.spec["programs"][event["role"]], event["arguments"])
        if repeat:
            assert tools[1]["computed"] == first + 3*repeat
            assert run["answer"] == tools[1]["computed"]
        else:
            assert tools[1]["feedback"] == "invalid" and not tools[1]["trace"]
            assert run["outcome"] == "stopped"
    delta = event_difference(normal, substituted)
    assert delta["event"] == 2 and disabled["events"][-1]["tokens"] == ["stop", "."]
    assert normal["events"][2]["tokens"] != substituted["events"][2]["tokens"]
    return dict(prompt=fixture["text"], runs=runs, first_substitution_difference=delta), raw


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--write", action="store_true")
    modes.add_argument("--open-failure", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        session = Session()
        binding = digest(canonical(session.spec))
        if args.open_failure:
            saved = json.loads(args.open_failure.read_text())
            assert saved["format_version"] == 1 and saved["holdout_sha256"] == binding
            observed, _ = test_case(session, saved["case"])
            assert observed == saved["replay"]
            print(json.dumps(dict(first_difference=observed["first_substitution_difference"],
                                  second_call_correct=observed["runs"]["normal"]["events"][2],
                                  second_call_substituted=observed["runs"]["substitute"]["events"][2]), indent=2))
            return
        records, timings = [], []
        for index in range(len(session.spec["prompts"])):
            record, timing = test_case(session, index)
            records.append(record)
            timings.append(timing)
        output = dict(format_version=1, holdout_sha256=binding, cases=records)
        if args.write:
            (OUT / "replay.json").write_text(json.dumps(output, indent=2)+"\n")
            (OUT / "timings.json").write_text(json.dumps(dict(cold_load_ns=session.cold_load_ns,
                image_prepare_ns=session.prepare_ns, run_samples=timings), indent=2)+"\n")
            (OUT / "failure.json").write_text(json.dumps(dict(format_version=1, holdout_sha256=binding,
                case=0, replay=records[0]), indent=2)+"\n")
        else:
            assert json.loads((OUT / "replay.json").read_text()) == output
    print("Four two-call tasks, one invalid argument, disabled and substituted feedback verified.")


if __name__ == "__main__":
    main()
