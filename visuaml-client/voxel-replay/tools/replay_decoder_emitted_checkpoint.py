"""Compare decoder-emitted resume tokens with the host-built checkpoint path."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import torch

from audit_causal_assembly import reference
from decoder_emitted_checkpoint import CHECKPOINT_TOKENS, OUT
from frozen_call_checkpoint import digest
from replay_decoder_execution_loop import run
from replay_rolling_context import OUT as ROLLING, checkpoint, restore, first_diff
from replay_word_decoder import audit, pack
from word_decoder import COMMIT, FRAME, HALT, OUT as WORD


def splice(code, snapshot, program_size):
    """Check positions and framing, then forward the emitted state unchanged."""
    if snapshot.ndim != 1 or len(snapshot) != CHECKPOINT_TOKENS or snapshot.dtype != torch.long:
        raise ValueError("Incomplete emitted checkpoint")
    rows = snapshot.reshape(5, FRAME)
    if (rows[:, 0].tolist() != [266]*5 or rows[:, -1].tolist() != [COMMIT]*5
        or rows[:, 6].tolist() != [0,1,2,3,265]
        or not 0 <= int(rows[-1, 1]) < program_size
        or not bool(((rows[:4, 7:11] >= 0) & (rows[:4, 7:11] <= 255)).all())
        or not bool(((rows[-1, 2:6] >= 0) & (rows[-1, 2:6] <= 255)).all())
        or bool(rows[:4, 1:6].any()) or bool(rows[-1, 7:11].any())):
        raise ValueError("Invalid checkpoint framing")
    return torch.cat((code, snapshot))


def snapshot_after(model, initial, steps):
    tokens, _, _, cache = run(model, initial, True, budget=steps*FRAME, halt_token=HALT)
    return tokens, cache, model.checkpoint(cache, tokens[-1])


def replay_path(model, program, initial, graph_hash, emitted, reserve,
                resume=None, remaining=1024):
    original = pack(program, initial)
    code = original[:-CHECKPOINT_TOKENS]
    prefix, state, retired = resume or (original, [0,1,0,0,*initial], 0)
    trace, segments, samples = [], [], []
    while len(trace) < remaining:
        step_budget = min((512-reserve-len(prefix))//FRAME, remaining-len(trace))
        assert step_budget > 0
        emitted_tokens, prefill, decode, cache = run(model, prefix, True, budget=step_budget*FRAME,
                                                    halt_token=HALT)
        will_resume = emitted_tokens[-1] != HALT and len(trace)+len(emitted_tokens)//FRAME < remaining
        readout_ns, rebuild_ns = 0, 0
        snapshot, next_prefix = None, None
        if emitted and will_resume:
            began = perf_counter_ns()
            snapshot = model.checkpoint(cache, emitted_tokens[-1])
            readout_ns = perf_counter_ns()-began
            began = perf_counter_ns()
            next_prefix = splice(code, snapshot, len(program))
            rebuild_ns = perf_counter_ns()-began
        # Independent audit is deliberately downstream of the emitted path.
        began = perf_counter_ns()
        states, _ = audit(emitted_tokens, state[4:])
        audited_ns = perf_counter_ns()-began
        state = states[-1]
        retired += len(states)
        trace.extend(states)
        if will_resume:
            began = perf_counter_ns()
            expected = restore(program, graph_hash, checkpoint(program, graph_hash, state, retired))
            if not emitted:
                next_prefix = expected
            baseline_build_ns = perf_counter_ns()-began
            if emitted:
                assert torch.equal(next_prefix, expected), retired
            else:
                rebuild_ns = baseline_build_ns
        segments.append(dict(retired=retired, source_steps=len(states), tokens=emitted_tokens,
                             execution_tokens=len(prefix)+len(emitted_tokens),
                             checkpoint_tokens=None if snapshot is None else snapshot.tolist(),
                             peak_reserved_tokens=len(prefix)+len(emitted_tokens)+
                                 (CHECKPOINT_TOKENS if snapshot is not None else 0)))
        samples.append(dict(prefill_ns=prefill, decode_ns=decode, readout_ns=readout_ns,
                            prefix_build_ns=rebuild_ns, audit_ns=audited_ns))
        if not will_resume:
            break
        prefix = next_prefix
    return dict(trace=trace, segments=segments), samples


def controls(model, new, graph_hash):
    program, initial = new["program"], new["initial"]
    prefix = pack(program, initial)
    code = prefix[:-CHECKPOINT_TOKENS]
    gen, cache, saved = snapshot_after(model, prefix, 2)
    initial_state = audit(gen, initial)[0][-1]
    healthy = splice(code, saved, len(program))
    # The full readout is atomic at acceptance, even if a handoff delivers a short suffix.
    truncated = []
    for count in range(CHECKPOINT_TOKENS):
        try:
            splice(code, saved[:count], len(program))
        except ValueError:
            truncated.append(count)
        else:
            raise AssertionError("Truncated snapshot accepted")
    assert torch.equal(healthy, restore(program, graph_hash,
                                       checkpoint(program, graph_hash, initial_state, 2)))
    partials = []
    for num in range(1, FRAME):
        tokens, _, _, partial = run(model, prefix, True, budget=FRAME+num, halt_token=HALT)
        try:
            model.checkpoint(partial, tokens[-1])
        except torch.jit.Error:
            partials.append(num)
        else:
            raise AssertionError("Checkpoint taken in an unfinished STORE")
    # Remove the initial address-2 COMMIT after the early generated write supersedes it.
    tape = torch.cat((prefix, torch.tensor(gen)))
    begin = len(code)+2*FRAME
    _, pruned = model(torch.cat((tape[:begin], tape[begin+FRAME:])), [])
    assert torch.equal(saved, model.checkpoint(pruned, COMMIT))

    literal = reference(program, initial, 0, 512)
    results = {}
    for name, fixture in (("intervention", new["intervention"]), ("missing_counter", new["missing_counter"])):
        assert fixture["checkpoint_after"] == 2
        altered = saved.clone()
        address = fixture["address"]
        if name == "intervention":
            value = fixture["replacement"]
        else:
            missing = torch.cat((saved[:address*FRAME], saved[(address+1)*FRAME:]))
            try:
                splice(code, missing, len(program))
            except ValueError:
                pass
            else:
                raise AssertionError("Checkpoint with a missing word accepted")
            value = fixture["default"]
        altered[address*FRAME+7:address*FRAME+11] = torch.tensor(list(value.to_bytes(4, "little")))
        changed = initial_state.copy()
        changed[4+address] = value
        remaining = 512 if name == "intervention" else 4
        target = splice(code, altered, len(program))
        result, _ = replay_path(model, program, initial, graph_hash, True, CHECKPOINT_TOKENS,
                                resume=(target, changed, 2), remaining=remaining)
        actual = result["trace"]
        assert actual == reference(program, changed[4:], 0, remaining,
                                   next_pc=changed[0], register=changed[2])
        effect = next(i for i,(a,b) in enumerate(zip(literal[2:], actual)) if a[:4] != b[:4])
        retired = effect+3
        if name == "intervention":
            assert retired == fixture["expected_load_retired"] and actual[effect][2] == 0
            assert actual[effect+1][0] == fixture["expected_branch_pc"]
        else:
            assert retired == fixture["expected_first_wrong_retired"] and actual[effect][2] == 0
        results[name] = dict(emitted_before=saved.tolist(), modified_snapshot=altered.tolist(),
                             first_observed_effect=dict(retired=retired, expected=literal[retired-1],
                                                        actual=actual[effect]),
                             continuation=result)
    # A rejected handoff left the original block available and unchanged.
    assert torch.equal(splice(code, model.checkpoint(cache, COMMIT), len(program)), healthy)
    return dict(truncated_handoffs_rejected=truncated, partial_store_cuts_rejected=partials,
                superseded_frame_removed=True, negatives=results)


def replay():
    word_path = WORD / "decoder.pt"
    model_path = OUT / "decoder.pt"
    source_hash = digest(word_path.read_bytes())
    model = torch.jit.load(str(model_path))
    old = torch.jit.load(str(word_path))
    existing = json.loads((ROLLING / "holdout.json").read_text())
    earlier = json.loads((ROLLING / "replay.json").read_text())
    new = json.loads((OUT / "holdout.json").read_text())
    cases = {
        "rolling-519": (existing["program_live_four"], [128,*existing["remaining_initial_memory"]]),
        "new-holdout": (new["program"], new["initial"]),
    }
    rows, samples = {}, {}
    for index,(name,(program, initial)) in enumerate(cases.items()):
        traces, times = {}, {}
        paths = [("emitted", model, True, CHECKPOINT_TOKENS), ("host_matched", old, False, CHECKPOINT_TOKENS),
                 ("host_unreserved", old, False, 0)]
        for key, engine, emit, reserve in paths[::1 if index == 0 else -1]:
            run(engine, pack(program, initial), True, budget=FRAME, halt_token=HALT)
            traces[key], times[key] = replay_path(engine, program, initial, source_hash, emit, reserve)
            assert first_diff(reference(program, initial, 0, 1024), traces[key]["trace"]) is None
        assert [s["tokens"] for s in traces["emitted"]["segments"]] == [
            s["tokens"] for s in traces["host_matched"]["segments"]]
        if name == "rolling-519":
            assert earlier["runs"]["live-4-countdown-128"]["trace"] == traces["emitted"]["trace"]
        rows[name], samples[name] = dict(program=program, initial=initial, paths=traces), times
    return dict(profile="visuaml-emitted-checkpoint-v1", decoder_sha256=digest(model_path.read_bytes()),
                word_sha256=source_hash, holdout=new, cases=rows,
                controls=controls(model, new, source_hash)), samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--open", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        output, samples = replay()
    if args.write:
        for name, data in (("replay", output), ("timings", samples)):
            (OUT / (name+".json")).write_text(json.dumps(data, indent=2)+"\n")
    else:
        assert output == json.loads((args.open or OUT / "replay.json").read_text())
    print("Decoder-emitted and host-built checkpoints, corrupted state and atomic handoff verified.")
