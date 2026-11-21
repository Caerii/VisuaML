"""Restart the frozen word decoder at committed source boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import torch

from audit_causal_assembly import reference
from replay_decoder_execution_loop import run
from replay_word_decoder import audit, pack
from word_decoder import COMMIT, FRAME, HALT, OUT as WORD

OUT = WORD.parent / "rolling-context"
CONTEXT = 512


def encoded(obj):
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()


def program_hash(program):
    return hashlib.sha256(encoded(program)).hexdigest()


def checkpoint(program, graph_hash, state, retired):
    pc, live, reg, r1, *memory = state
    if r1 != 0 or len(memory) != 4:
        raise ValueError("Outside the word decoder checkpoint ABI")
    return dict(format=1, program_sha256=program_hash(program), decoder_sha256=graph_hash,
                next_pc=pc, live=live, r0=reg, words=memory, retired=retired)


def restore(program, graph_hash, saved):
    if (saved["format"], saved["program_sha256"], saved["decoder_sha256"]) != (
            1, program_hash(program), graph_hash):
        raise ValueError("Checkpoint identity mismatch")
    if saved["live"] != 1:
        raise ValueError("The computation has halted")
    if (not 0 <= saved["next_pc"] < len(program) or not 0 <= saved["r0"] < 2**32
            or saved["retired"] < 0):
        raise ValueError("Invalid continuation")
    prefix = pack(program, saved["words"])
    prefix[-FRAME+1] = saved["next_pc"]
    prefix[-FRAME+2:-FRAME+6] = torch.tensor(list(saved["r0"].to_bytes(4, "little")))
    return prefix


def export(program, graph_hash, saved, tokens):
    states, _ = audit(tokens, saved["words"])
    if not states:
        return saved
    return checkpoint(program, graph_hash, states[-1], saved["retired"]+len(states))


def first_diff(left, right):
    return next((dict(step=i+1, expected=left[i:i+1], actual=right[i:i+1])
                 for i in range(max(len(left), len(right))) if left[i:i+1] != right[i:i+1]), None)


def rolling(model, program, graph_hash, initial, source_limit=1024, saved=None):
    saved = saved or checkpoint(program, graph_hash, [0,1,0,0,*initial], 0)
    chunks, trace = [], []
    stats = []
    while saved["live"] and len(trace) < source_limit:
        began = perf_counter_ns()
        prefix = restore(program, graph_hash, saved)
        restore_ns = perf_counter_ns()-began
        capacity = (CONTEXT-prefix.numel())//FRAME
        steps = min(capacity, source_limit-len(trace))
        if steps <= 0:
            raise ValueError("No room to run an instruction")
        tokens, prefill_ns, decode_ns, cache = run(
            model, prefix, True, budget=steps*FRAME, halt_token=HALT)
        began = perf_counter_ns()
        states, _ = audit(tokens, saved["words"])
        next_saved = checkpoint(program, graph_hash, states[-1], saved["retired"]+len(states))
        snapshot = encoded(next_saved)
        # Materialize a new checkpoint object at each boundary as an import would.
        next_saved = json.loads(snapshot)
        export_ns = perf_counter_ns()-began
        trace.extend(states)
        chunks.append(dict(start=saved, end=next_saved, tokens=tokens, prefix_tokens=len(prefix),
                           peak_active_tokens=len(prefix)+len(tokens), checkpoint_bytes=len(snapshot),
                           kv_bytes=sum(t.numel()*t.element_size() for pair in cache for t in pair)))
        stats.append(dict(prefill_ns=prefill_ns, decode_ns=decode_ns,
                          restore_ns=restore_ns, export_ns=export_ns))
        saved = next_saved
    return dict(chunks=chunks, trace=trace, halted=saved["live"] == 0,
                generated_tokens=sum(len(chunk["tokens"]) for chunk in chunks)), stats


def short_controls(model, program, graph_hash):
    initial = [3,7,0,11]
    base = pack(program, initial)
    full, _, _, _ = run(model, base, False, budget=240, halt_token=HALT)
    cut = 7  # The counter's first overwrite has completed.
    prefix = full[:cut*FRAME]
    start = checkpoint(program, graph_hash, [0,1,0,0,*initial], 0)
    saved = export(program, graph_hash, start, prefix)
    compacted, _, _, _ = run(model, restore(program, graph_hash, saved), True,
                             budget=144, halt_token=HALT)
    assert compacted == full[cut*FRAME:]
    # Remove the initial counter COMMIT (not the subsequent store). On this
    # suffix it is superseded; the splice changes positions by one full frame.
    superseded_start = len(program)*FRAME
    original_history = torch.cat((base, torch.tensor(prefix)))
    pruned = torch.cat((original_history[:superseded_start], original_history[superseded_start+FRAME:]))
    assert run(model, pruned, True, budget=144, halt_token=HALT)[0] == compacted
    cuts = []
    for partial_length in range(1, FRAME):
        budget = FRAME+partial_length
        partial, _, _, cache = run(model, base, True, budget=budget, halt_token=HALT)
        partial_saved = export(program, graph_hash, start, partial)
        assert partial_saved["retired"] == 1 and partial_saved["words"][2] == 0
        assert not cache[2][0][0, len(base)+FRAME:, :4].any()
        resumed = restore(program, graph_hash, partial_saved)
        assert run(model, resumed, True, budget=228, halt_token=HALT)[0] == full[FRAME:]
        cuts.append(partial_length)
    assert prefix[-1] == COMMIT
    return dict(copy_checkpoint=saved, identical_suffix=compacted, superseded_frame_removed=FRAME,
                partial_store_lengths=cuts, uninterrupted_tokens=full)


def test_negative(model, program, graph_hash, scenario, baseline):
    initial = [64,*scenario["remaining_initial_memory"]]
    original = baseline["trace"]
    combined, timings = {}, {}
    for name, spec in (("retained_value_intervention", scenario["late_intervention"]),
                       ("incomplete_checkpoint", scenario["incomplete_control"])):
        after = spec["after_retired"]
        saved = checkpoint(program, graph_hash, original[after-1], after)
        if name == "incomplete_checkpoint":
            # Production import rejects a missing word. Also model a defective
            # exporter that silently fills the omitted live word with zero.
            missing = dict(saved, words=[word for address, word in enumerate(saved["words"])
                                         if address != spec["omitted_address"]])
            try:
                restore(program, graph_hash, missing)
            except ValueError:
                pass
            else:
                raise AssertionError("Incomplete memory was accepted")
            saved["words"][spec["omitted_address"]] = spec["default_word"]
            trace, times = rolling(model, program, graph_hash, initial, source_limit=4, saved=saved)
            actual = trace["trace"]
            assert actual[0][2] == 0 and original[after][2] == 64
            assert after+1 == spec["first_wrong_retired"]
            causal = dict(retired=after+1, source_pc=saved["next_pc"], kind="load",
                          expected_word=original[after][2], actual_word=actual[0][2])
        else:
            saved["words"][spec["address"]] = spec["replacement"]
            trace, times = rolling(model, program, graph_hash, initial, saved=saved)
            actual = trace["trace"]
            first_read = next(i for i in range(1, len(actual)) if actual[i-1][0] == spec["load_pc"])
            assert actual[first_read][2] == spec["replacement"] and actual[first_read][0] == spec["branch_pc"]
            assert actual[first_read+1][0] == spec["expected_next_pc"]
            assert trace["halted"] and actual[-1][7] == 0 and original[-1][7] == 11
            # Until the late load, compare everything except the intentionally edited word.
            assert [s[:6]+s[7:] for s in actual[:first_read]] == [
                s[:6]+s[7:] for s in original[after:after+first_read]]
            causal = dict(retired=after+first_read+1, source_pc=spec["load_pc"], kind="load",
                          actual_word=actual[first_read][2], expected_word=original[after+first_read][2],
                          next_branch_pc=actual[first_read+1][0])
        assert actual == reference(program, saved["words"], 0, len(actual),
                                   next_pc=saved["next_pc"], register=saved["r0"])
        combined[name] = dict(continuation=trace, resumed_from=after,
                              first_state_difference=first_diff(original[after:], actual),
                              first_observed_effect=causal)
        timings[name] = times
    return combined, timings


def replay():
    path = WORD / "decoder.pt"
    graph_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    model = torch.jit.load(str(path))
    scenario = json.loads((OUT / "holdout.json").read_text())
    runs, timings = {}, {}
    for label, address_count in (("two", 2), ("four", 4)):
        program = scenario["program_live_"+label]
        for count in scenario["countdowns"]:
            initial = [count,*scenario["remaining_initial_memory"]]
            result, timing = rolling(model, program, graph_hash, initial)
            expected = reference(program, initial, 0, 1024)
            assert result["halted"] and first_diff(expected, result["trace"]) is None
            assert all(chunk["end"]["words"][2] == 7 for chunk in result["chunks"])
            key = f"live-{address_count}-countdown-{count}"
            runs[key] = dict(live_during_loop=address_count, **result)
            timings[key] = timing
    program = scenario["program_live_four"]
    negatives, negative_times = test_negative(model, program, graph_hash, scenario, runs["live-4-countdown-64"])
    timings.update(negative_times)
    return dict(profile="visuaml-rolling-context-v1", graph_sha256=graph_hash, holdout=scenario, runs=runs,
                short_controls=short_controls(model, program, graph_hash), negatives=negatives), timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--open", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        actual, timings = replay()
    if args.write:
        for name, record in (("replay", actual), ("timings", timings)):
            (OUT / (name+".json")).write_text(json.dumps(record, indent=2)+"\n")
    else:
        assert actual == json.loads((args.open or OUT / "replay.json").read_text())
    print("Rolling histories, live-word negatives, partial cuts and superseded-write replay verified.")
