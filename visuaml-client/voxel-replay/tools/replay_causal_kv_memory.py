"""Replay the cached memory primitive and a hybrid assembly execution."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import torch
from torch import nn

from audit_causal_assembly import reference
from causal_kv_memory import OUT
from causal_tensor_assembly import CausalAssembly, OUT as STRUCTURED, decode, observe, start, word
from frozen_call_checkpoint import digest
from vocabulary_tensor_assembly import OUT as VOCABULARY, FRAME, encode_image, generate


def write_tokens(address, bits):
    return torch.cat((address.long().reshape(1), bits.long() + 4, torch.tensor([6])))


class HybridAssembly(nn.Module):
    def __init__(self, memory):
        super().__init__()
        self.memory = memory
        self.transition = CausalAssembly()

    def forward(self, image: torch.Tensor, fuel: int):
        # The four initialized rows are part of the original program image ABI.
        initial = torch.cat([write_tokens(image[i, 155:159].argmax(), image[i, 27:59])
                             for i in range(image.size(0) - 5, image.size(0) - 1)])
        _, first, second = self.memory(initial)
        history = image
        for step in range(fuel):
            instruction = self.transition.select(history[-1, 75:91], history[:, :16], history[:, 16:75])
            op = instruction[:5]
            loaded = image.new_zeros(32)
            if bool(op[1] == 1):
                address = instruction[7:11].argmax().reshape(1)
                read, first, second = self.memory(torch.cat((address, torch.tensor([7]))), first, second)
                loaded = read[-1]
            # Supplying loaded bits skips the structured latest-write operation.
            record = self.transition(history, loaded)
            if bool(op[2] == 1):
                _, first, second = self.memory(write_tokens(record[155:159].argmax(), record[27:59]),
                                               first, second)
            history = torch.cat((history, record[None]))
            if bool(record[-1] == 0):
                break
        return history[image.size(0):]


def tape_from(spec, replacement=False):
    tokens, probes = [], []
    for row in spec:
        address = row["address"]
        if "value" in row:
            value = row.get("replace", row["value"]) if replacement else row["value"]
            tokens.extend([address] + [4 + ((value >> i) & 1) for i in range(32)])
            if row.get("commit", True):
                tokens.append(6)
        else:
            tokens.extend([address, 7])
            probes.append(len(tokens)-1)
    return torch.tensor(tokens), probes


def cached(model, tokens):
    first, second = None, None
    rows = []
    for token in tokens:
        row, first, second = model(token.reshape(1), first, second)
        rows.append(row)
    return torch.cat(rows), first, second


def tape_replay(model, spec, replace):
    tokens, probes = tape_from(spec, replace)
    full, _, _ = model(tokens)
    incremental, first, second = cached(model, tokens)
    assert torch.equal(full[probes], incremental[probes])
    values = [word(row) for row in full[probes]]
    return dict(tokens=tokens.tolist(), read_positions=probes, full_and_cached=values,
                cache_bytes=sum(t.numel()*t.element_size() for t in first + second))


def time_call(fn):
    start_time = perf_counter_ns()
    result = fn()
    return perf_counter_ns() - start_time, result


def measure(model, tape, hybrid, image, fuel, structured, vocabulary):
    samples = {}
    initial = encode_image(image)
    states = decode(structured, image, fuel)
    histories = [torch.cat((image, states[:i])) for i in range(len(states))]

    def repeated_transition():
        for history in histories:
            for _ in range(FRAME):
                structured(history)

    calls = {"hybrid": lambda: hybrid(image, fuel), "structured": lambda: decode(structured, image, fuel),
             "vocabulary": lambda: generate(vocabulary, initial, fuel*FRAME),
             "repeated_transition_only": repeated_transition}
    for call in calls.values():
        call()
    for index in range(8):
        for name in list(calls)[::(-1 if index % 2 else 1)]:
            elapsed, _ = time_call(calls[name])
            samples.setdefault(name, []).append(elapsed)
    growth = {}
    for count in (32, 128, 256, 511):
        prefix = tape.repeat((count//len(tape))+1)[:count]
        _, first, second = model(prefix)
        probe = torch.tensor([8])
        model(probe, first, second)
        growth[count] = dict(prefill_ns=[time_call(lambda: model(prefix))[0] for _ in range(8)],
                             incremental_ns=[time_call(lambda: model(probe, first, second))[0] for _ in range(8)],
                             cache_bytes=sum(t.numel()*t.element_size() for t in first+second),
                             prefill_storage_bytes=sum(t.untyped_storage().nbytes() for t in first+second))
    return dict(source_replay_ns=samples, cache_growth=growth)


def replay(write):
    decoder_path = OUT / "decoder.pt"
    model = torch.jit.load(str(decoder_path))
    spec = json.loads((OUT / "holdout.json").read_text())
    original = tape_replay(model, spec["events"], False)
    intervention = tape_replay(model, spec["events"], True)
    assert original["full_and_cached"] == spec["expected"]
    assert intervention["full_and_cached"] == spec["intervened"]
    # The same already-built cache also supports a multi-token suffix.
    tokens = torch.tensor(original["tokens"])
    pivot = len(tokens)-5
    _, first, second = model(tokens[:pivot])
    updated, _, _ = model(tokens[pivot:], first, second)
    assert [word(updated[index-pivot]) for index in original["read_positions"] if index >= pivot] == spec["expected"][-2:]

    # Keep the sole matching commit near the beginning of a saturated context.
    oldest = torch.cat((tokens[:34], torch.full((476,), 8), torch.tensor([1, 7])))
    far, _, _ = model(oldest)
    _, first, second = model(oldest[:-1])
    far_cached, _, _ = model(oldest[-1:], first, second)
    assert word(far[-1]) == word(far_cached[-1]) == spec["expected"][0]

    artifact = OUT / "hybrid.pt"
    if write:
        compiled = torch.jit.freeze(torch.jit.script(HybridAssembly(model).eval()))
        compiled.save(str(artifact))
    hybrid = torch.jit.load(str(artifact))
    structured = torch.jit.load(str(STRUCTURED / "decoder.pt"))
    vocabulary = torch.jit.load(str(VOCABULARY / "decoder.pt"))
    fixtures = json.loads((VOCABULARY / "holdout.json").read_text())
    archived = json.loads((VOCABULARY / "replay.json").read_text())
    integration = []
    for case in fixtures:
        program = case["program"]
        memory = [case["seed"], 0, 0, 0]
        image = start(program, memory, case["counter"])
        fuel = case["fuel"]
        result = hybrid(image, fuel)
        trace = observe(result, memory)
        scalar = reference(program, memory, case["counter"], fuel)
        assert trace == scalar
        assert torch.equal(result, decode(structured, image, fuel))
        assert trace == next(row["trace"] for row in archived["cases"] if row["name"] == case["name"])
        integration.append(dict(name=case["name"], source_steps=len(trace), trace=trace))
    output = dict(decoder_sha256=digest(decoder_path.read_bytes()), hybrid_sha256=digest(artifact.read_bytes()),
                  original=original, intervention=intervention, instruction_witnesses=integration,
                  full_context_read=dict(tokens=len(oldest), value=word(far[-1])))
    if write:
        # Time the same taken-branch workload across the three paths.
        first = fixtures[0]
        image = start(first["program"], [first["seed"], 0, 0, 0], first["counter"])
        timings = measure(model, tokens, hybrid, image, first["fuel"], structured, vocabulary)
        (OUT / "timings.json").write_text(json.dumps(timings, indent=2)+"\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--open", type=Path)
    options = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        result = replay(options.write)
    if options.write:
        (OUT / "replay.json").write_text(json.dumps(result, indent=2)+"\n")
    else:
        assert result == json.loads((options.open or OUT / "replay.json").read_text())
    print("Full-prefix, cached, intervened and hybrid memory witnesses verified.")
