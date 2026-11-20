"""Exhaustive passage search and exact replay of the selected session."""
import argparse
from itertools import product
import json
from pathlib import Path
import statistics
import tracemalloc
import struct
from time import perf_counter_ns
from unittest.mock import patch

from fork_divergence import first_divergence
from frozen_call_checkpoint import canonical, digest, encode
from replay_frozen_recurrent import PACKED, wasmtime_run
from replay_voxel_call_checkpoint import CELL, OUT as PASSAGE, Witness, page
from voxel_fork_state import Run, begin_from_page, export_state, fork, restore, seal

OUT = PASSAGE.parent / "voxel-fork-search"
HORIZON = 5


def make_checkpoint(witness):
    initial = bytearray(65536)
    struct.pack_into("<6i", initial, CELL, 7, 1, 0, 0, -1, 0)
    return export_state(Run(witness, witness.start(initial), (0,) * (HORIZON + 1), 1, 0), 14)


def edits(actions):
    return {index: value for index, value in enumerate(actions, 1) if value}


def finish_branch(run):
    run.finish_tick()
    while run.next_event < len(run.inputs):
        run.tick()
    return run.outputs


def verify_branch(witness, checkpoint, changes, detailed_tick):
    run = fork(witness, checkpoint, changes)
    output, suffix = run.finish_tick(trace=True)
    reference = json.loads((PASSAGE / "replay.json").read_text())
    assert suffix == reference["suffix"]
    assert output["world"] == [0, 0, 7, 1]
    detail = None
    while run.next_event < len(run.inputs):
        tick = run.next_event
        prior = bytearray(page(run.vm))
        struct.pack_into("<i", prior, CELL+4, 0)
        struct.pack_into("<i", prior, CELL+12, run.inputs[tick])
        result, trace = run.tick(trace=tick == detailed_tick)
        assert wasmtime_run(witness.raw, CELL, prior) == (result["returned"], page(run.vm))
        if trace:
            detail = trace
    return run.outputs, detail


def enumerate_branches(witness, checkpoint, shared):
    branches, fork_ns, run_ns = [], [], []
    ticks = held = peak_pages = peak_checkpoints = 0

    def collect(actions, outputs):
        crossing = next((r["tick"] for r in outputs if r["world"][1] == 1), None)
        for n, result in enumerate(outputs):
            position = int(any(actions[:n]))
            assert result["world"] == [0, position, 7 if n == 0 else 0, n+1]
            assert result["returned"] == position
        branches.append(dict(actions=actions, edit_count=sum(actions), crossing_tick=crossing,
                             outputs=outputs))

    def visit(prefix, run):
        nonlocal ticks, peak_pages, peak_checkpoints
        if len(prefix) == HORIZON:
            collect(prefix, run.outputs)
            return
        # The halted invocation has no continuation registers for the next
        # tick. The tick ABI starts it from this immutable page and input.
        saved = page(run.vm)
        peak_checkpoints = max(peak_checkpoints, len(prefix) + 2)
        for action in (0, 1):
            child_prefix = (*prefix, action)
            inputs = (0, *child_prefix, *((0,) * (HORIZON-len(child_prefix))))
            start = perf_counter_ns()
            child = Run(witness, begin_from_page(witness, saved, action), inputs, run.next_event + 1,
                        run.retired, list(run.outputs))
            ready = perf_counter_ns()
            child.finish_tick()
            fork_ns.append(ready-start)
            run_ns.append(perf_counter_ns()-ready)
            ticks += 1
            peak_pages = max(peak_pages, 2 * len(prefix) + 4)
            visit(child_prefix, child)
            del child

    if shared:
        start = perf_counter_ns()
        root = restore(witness, checkpoint)
        ready = perf_counter_ns()
        root.finish_tick()
        fork_ns.append(ready-start)
        run_ns.append(perf_counter_ns()-ready)
        held = 1
        visit((), root)
    else:
        for actions in product((0, 1), repeat=HORIZON):
            start = perf_counter_ns()
            run = fork(witness, checkpoint, edits(actions))
            ready = perf_counter_ns()
            collect(actions, finish_branch(run))
            fork_ns.append(ready-start)
            run_ns.append(perf_counter_ns()-ready)
            ticks += HORIZON
            held += 1
            del run
        peak_pages = 4
        peak_checkpoints = 1
    return branches, dict(future_ticks=ticks, suspended_suffixes=held,
                          peak_retained_checkpoints=peak_checkpoints,
                          peak_page_buffer_bound_bytes=peak_pages*65536,
                          fork_ns=fork_ns, execution_ns=run_ns)


def search(witness, checkpoint):
    branches, traversal = enumerate_branches(witness, checkpoint, True)
    fork_ns, run_ns = traversal["fork_ns"], traversal["execution_ns"]
    baseline = branches[0]
    winner = min((b for b in branches if b["crossing_tick"] is not None),
                 key=lambda b: (b["edit_count"], b["crossing_tick"], b["actions"]))
    first_input = next(iter(edits(winner["actions"])))
    first_effect = first_divergence([r["world"] for r in baseline["outputs"]],
                                    [r["world"] for r in winner["outputs"]])
    _, wait = verify_branch(witness, checkpoint, {}, first_effect)
    checked, walk = verify_branch(witness, checkpoint, edits(winner["actions"]), first_effect)
    assert checked == winner["outputs"]
    assert not any(r["access"][:2] == [2, CELL+8] for r in wait)
    write = next(r for r in walk if r["access"] == [2, CELL+8, 0, 1])
    selected = branches[-1]
    assert verify_branch(witness, checkpoint, edits(selected["actions"]), -1)[0] == selected["outputs"]

    exported, restored, copied = [], [], []
    seed = witness.export()[-1]
    for _ in range(32):
        start = perf_counter_ns()
        assert export_state(Run(witness, seed, (0,) * (HORIZON+1), 1, 14), 14) == checkpoint
        exported.append(perf_counter_ns()-start)
        start = perf_counter_ns()
        recovered = restore(witness, checkpoint)
        restored.append(perf_counter_ns()-start)
        start = perf_counter_ns()
        copy = recovered.vm.vm.page.clone()
        copied.append(perf_counter_ns()-start)
        assert copy.data_ptr() != recovered.vm.vm.page.data_ptr()
    sample = dict(export_ns=exported, restore_ns=restored, copy_page_ns=copied,
                  fork_ns=fork_ns, branch_execution_ns=run_ns)
    metrics = dict(traversal=traversal, checkpoint_bytes=len(canonical(checkpoint))+1,
                   private_linear_memory_bytes_per_branch=copy.numel()*copy.element_size(),
                   completed_future_branch_ticks=traversal["future_ticks"],
                   # The denominator includes restoring each branch and finishing its held tick.
                   future_branch_ticks_per_second=traversal["future_ticks"]*1e9/(sum(fork_ns)+sum(run_ns)),
                   median_ns={key: statistics.median(values) for key, values in sample.items()},
                   samples_ns=sample)
    session = dict(version=1, checkpoint_sha256=checkpoint["sha256"],
                   future_input_override={str(k): v for k, v in edits(winner["actions"]).items()},
                   outputs=checked, first_input_tick=first_input, first_world_effect_tick=first_effect,
                   source_write=write, source_trace=walk, baseline_trace=wait)
    return dict(branches=branches, winner=session, metrics=metrics)


def check_checkpoint(witness, checkpoint):
    left, right = (restore(witness, checkpoint) for _ in range(2))
    right.step()
    assert left.vm.vm.page.data_ptr() != right.vm.vm.page.data_ptr()
    assert encode(left.vm, witness.identity) == checkpoint["vm"]
    left.finish_tick()
    left.begin_tick()
    later = export_state(left, left.retired + 3)
    assert later["next_event"] == 2 and len(later["pending_outputs"]) == 1
    restored = restore(witness, later)
    assert restored.finish_tick(trace=True) == left.finish_tick(trace=True)
    for invalid in (dict(checkpoint, next_event=2),
                    seal({k: v if k != "input_log_sha256" else "invalid"
                          for k, v in checkpoint.items() if k != "sha256"})):
        try:
            restore(witness, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt checkpoint accepted")
    try:
        fork(witness, checkpoint, {0: 1})
    except ValueError:
        pass
    else:
        raise AssertionError("Consumed input was overwritten")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--export", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--session", type=Path)
    args = parser.parse_args()
    witness = Witness()
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        checkpoint = json.loads(args.checkpoint.read_text()) if args.checkpoint else make_checkpoint(witness)
        if args.export:
            args.export.write_bytes(canonical(checkpoint)+b"\n")
            return
        if args.session:
            session = json.loads(args.session.read_text())
            assert session["checkpoint_sha256"] == checkpoint["sha256"] and session["version"] == 1
            changes = {int(k): v for k, v in session["future_input_override"].items()}
            detail_tick = session["first_world_effect_tick"]
            verified, trace = verify_branch(witness, checkpoint, changes, detail_tick)
            baseline, wait = verify_branch(witness, checkpoint, {}, detail_tick)
            assert min(changes) == session["first_input_tick"]
            assert detail_tick == next(a["tick"] for a, b in zip(verified, baseline)
                                       if a["world"] != b["world"])
            assert verified == session["outputs"] and trace == session["source_trace"]
            assert wait == session["baseline_trace"]
            assert session["source_write"] == next(r for r in trace if r["access"] == [2, CELL+8, 0, 1])
            print(json.dumps(session))
            return
        check_checkpoint(witness, checkpoint)
        if args.benchmark:
            samples = {"flat": [], "tree": []}
            for order in ((False, True), (True, False), (False, True)):
                for shared in order:
                    start = perf_counter_ns()
                    outputs, metrics = enumerate_branches(witness, checkpoint, shared)
                    elapsed = perf_counter_ns() - start
                    samples["tree" if shared else "flat"].append(dict(elapsed_ns=elapsed, **metrics))
                    if shared:
                        assert canonical(outputs) == canonical(reference)
                    else:
                        reference = outputs
            for shared, label in ((False, "flat"), (True, "tree")):
                tracemalloc.start()
                enumerate_branches(witness, checkpoint, shared)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                samples[label + "_peak_python_bytes"] = peak
            print(json.dumps(samples, indent=2))
            return

        result = search(witness, checkpoint)
        if args.write:
            (OUT / "checkpoint.json").write_bytes(canonical(checkpoint)+b"\n")
            for name, data in (("search.json", result), ("winning-session.json", result["winner"])):
                (OUT / name).write_text(json.dumps(data, indent=2)+"\n")
        else:
            prior = json.loads((OUT / "search.json").read_text())
            assert canonical(prior["branches"]) == canonical(result["branches"])
            assert prior["winner"] == result["winner"]
        print(json.dumps(dict(winner=result["winner"]["future_input_override"],
                              first_effect=result["winner"]["first_world_effect_tick"],
                              metrics={k: v for k, v in result["metrics"].items() if k != "samples_ns"})))


if __name__ == "__main__":
    main()
