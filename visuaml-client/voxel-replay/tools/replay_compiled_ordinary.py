"""Exact replay and long paired timing of the compiled ordinary CPU graph."""
import argparse
from collections import Counter
import json
import math
import statistics
from time import perf_counter_ns, sleep
from unittest.mock import patch

import torch
import wasmtime

import terrain_verify as terrain
from build_compiled_ordinary import OUT, register
from replay_composed_loads import CALLS
from replay_frozen_calls import NESTED
from replay_frozen_recurrent import ARCHIVE, PACKED, graph_case, sha, wasmtime_run
from replay_guest_rewrite import EXECUTOR
from replay_ordinary_effects import OUT as ORDINARY, audit, descriptors, drive, no_regions, runtime_profile
from replay_rule_portfolio import OUT as PORTFOLIO, PAIR, PRIOR, prepare, with_rows
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image
from visuaml.compiler_lab.frozen_recurrent_v58 import snapshot_page


def native_rows(prepared):
    rows = prepared.rows
    eligible = (rows[:, 0] < 20) & (rows[:, 11] < 0) & (rows[:, 16] == 0)
    eligible[-1] = False
    return with_rows(prepared, torch.cat((rows, eligible.to(torch.int64).unsqueeze(1)), 1))


def result_stats(samples, terrain_case=True):
    times = sorted(s["ns"] for s in samples)
    answer = {"median_ns": statistics.median(times), "samples": len(times),
              "p95_ns": times[math.ceil(.95 * len(times)) - 1], "raw": samples}
    if terrain_case:
        answer.update(deadline_misses=sum(x > 50_000_000 for x in times),
                      agent_ticks_per_s=8 * len(times) * 1e9 / sum(times))
    return answer


def paired(graphs, images, raw, argument, page, events, steps, world=True):
    samples = {name: [] for name in graphs}
    for i, delta in enumerate(events):
        if delta is not None:
            terrain.write(page, 320, [4 * m for m in delta])
        expected, oracle = wasmtime_run(raw, argument, page)
        metrics = {}
        for name in (("lazy", "native") if i % 2 == 0 else ("native", "lazy")):
            changed, result = drive(graphs[name], images[name], argument, page)
            assert changed == oracle and result["result"] == expected and result["steps"] == steps
            metrics[name] = result
            samples[name].append({**result, "page_sha256": sha(changed)})
        assert all(metrics["lazy"][field] == metrics["native"][field]
                   for field in ("steps", "calls", "fusions"))
        if delta is not None:
            page = bytearray(oracle)
    return {name: result_stats(rows, world) for name, rows in samples.items()}


def captures(graph, image, raw, ticks, checkpoint_every=12):
    scene = terrain.scale_session(8, False, ticks)
    page = terrain.initial_page(scene["initial"])
    page.extend(bytes((image.memory_pages - 1) * terrain.PAGE_BYTES))
    start = perf_counter_ns()
    template = initialize(image, terrain.BASE, page, 100_000)
    setup_ns = perf_counter_ns() - start
    live = template.vm.page
    snapshots = {}
    samples = []
    display_bytes = []
    drive(graph, image, terrain.BASE, page)
    period_ns = 50_000_000
    epoch = perf_counter_ns()
    for i, step in enumerate(scene["inputs"]):
        scheduled = epoch + i * period_ns
        idle = scheduled - perf_counter_ns()
        if idle > 0:
            sleep(idle / 1e9)
        began = perf_counter_ns()
        before = snapshot_page(template.vm)
        capture_started = perf_counter_ns()
        terrain.write(live.numpy(), 320, [4 * d for d in step["moves"]])
        state = template
        calls = 0
        while int(state.vm.status) == 0:
            state = graph(image.rows, state, 64).state
            calls += 1
        simulated = perf_counter_ns()
        assert int(state.vm.status) == 1 and int(state.vm.fuel) == 100_000 - 1650 and calls == 26
        publication_started = perf_counter_ns()
        after = snapshot_page(state.vm)
        finished = perf_counter_ns()
        if (i + 1) % checkpoint_every == 0 or i + 1 == ticks:
            snapshots[i] = after
        start, stop = terrain.BASE, terrain.BASE + terrain.ABI_BYTES
        display_bytes.append((before[start:stop], after[start:stop]))
        samples.append({"ns": finished - scheduled, "simulation_and_input_ns": simulated - capture_started,
                        "schedule_delay_ns": began - scheduled,
                        "publication_ns": capture_started - began + finished - publication_started,
                        "steps": 1650, "calls": calls})
    # No oracle, trace, or hashing runs during the continuous simulation.
    store = wasmtime.Store()
    exports = wasmtime.Instance(store, wasmtime.Module(store.engine, raw), []).exports(store)
    exports["memory"].write(store, page)
    offset = terrain.BASE + 320
    hashes = {}
    for i, step in enumerate(scene["inputs"]):
        start, stop = terrain.BASE, terrain.BASE + terrain.ABI_BYTES
        assert exports["memory"].read(store, start, stop) == display_bytes[i][0]
        terrain.write(page, 320, [4 * d for d in step["moves"]])
        exports["memory"].write(store, page[offset:offset + 32], offset)
        assert exports["run"](store, terrain.BASE) == 0
        assert exports["memory"].read(store, start, stop) == display_bytes[i][1]
        if i in snapshots:
            assert snapshots[i] == exports["memory"].read(store)
            hashes[i + 1] = sha(snapshots[i])
    return {"scheduled_tick_stats": result_stats(samples), "period_ns": period_ns,
            "full_copies_per_tick": 2, "retained_checkpoint_every_ticks": checkpoint_every,
            "setup_ns": setup_ns, "snapshots_verified": hashes, "display_regions_verified": ticks}


def check_profiles(graphs, images, arg, page, sites):
    result = {}
    for name, machine in graphs.items():
        data = runtime_profile(machine, images[name], arg, page, sites)
        # Add a true budget-64 profile to count graph-to-C++ crossings.
        state = initialize(images[name], arg, page, 100_000)
        for _ in range(7):
            machine(images[name].rows, initialize(images[name], arg, page, 100_000), 64)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as recorded:
            machine(images[name].rows, state, 64)
        data["budget_64_prefix_counts"] = {e.key: e.count for e in recorded.key_averages()}
        result[name] = data
    return result


def verify_histories(graphs, images, raw):
    output = {}
    for label in ("terrain", "terrain-omitted"):
        expected = json.loads((OUT.parents[1] / "fixtures" / f"{label}.proof.json").read_text())
        page = terrain.initial_page(expected["session"]["initial"])
        rows = []
        for inputs, save in zip(expected["session"]["inputs"], expected["ticks"], strict=True):
            terrain.write(page, 4, [int(inputs["edit"])])
            terrain.write(page, 320, [4 * x for x in inputs["moves"]])
            data = paired(graphs, images, raw, terrain.BASE, page, [None], save["steps"], world=False)
            assert all(v["raw"][0]["page_sha256"] == save["page_sha256"] for v in data.values())
            _, after = wasmtime_run(raw, terrain.BASE, page)
            page = bytearray(after)
            rows.append(data)
        output[label] = rows
    return output


def special_witnesses(one, machine):
    nested = bytes(wasmtime.wat2wasm(NESTED))
    rollback = (ARCHIVE / "v58-array-memory/rollback.wasm").read_bytes()
    original = bytearray(65536)
    original[1:5] = original[7:11] = bytes.fromhex("01000080")
    fixture = OUT / "after-freeze.wat"
    later = fixture.with_suffix(".wasm").read_bytes()
    assert later == bytes(wasmtime.wat2wasm(fixture.read_text()))
    results = {}
    for name, raw, arg, page, fuel, status in (
            ("nested", nested, 0, bytes(65536), 100, 1),
            ("helper_trap", nested, 65533, bytes(65536), 100, 3),
            ("before_store", rollback, 0, original, 8, 2),
            ("after_store", rollback, 0, original, 9, 2),
            ("rollback", rollback, 0, original, 100, 1),
            ("post_freeze_branch", later, 8, bytes(65536), 100, 1),
            ("post_freeze_else", later, 0, bytes(65536), 100, 1),
            ("post_freeze_signed_min", later, -2147483648, bytes(65536), 100, 1)):
        plain = load_call_image(raw)
        native = native_rows(no_regions(plain))
        data = audit(one, machine, plain, native, arg, page, fuel, set())
        assert data["status"] == status
        if status != 2:
            expected, actual = wasmtime_run(raw, arg, page, trap_expected=status == 3)
            assert data["return_i32"] == expected and data["page_sha256"] == sha(actual)
        results[name] = {**data, "wasm_sha256": sha(raw), "fuel": fuel}
    changed = original.copy()
    changed[7:11] = bytes.fromhex("78563412")
    assert results["before_store"]["page_sha256"] == sha(original)
    assert results["after_store"]["page_sha256"] == sha(changed)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--witnesses", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    registration = register()
    graphs = {name: torch.jit.load(str(path)) for name, path in (
        ("lazy", ORDINARY / "lazy.pt"), ("native", OUT / "machine.pt"))}
    one = torch.jit.load(str(CALLS / "machine.pt"))
    result = {"registration": registration,
              "graph_sha256": sha((OUT / "machine.pt").read_bytes())}
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        if args.witnesses:
            result["witnesses"] = special_witnesses(one, graphs["native"])
        else:
            base = torch.jit.load(str(EXECUTOR / "machine.pt"))
            planners = {1: (PAIR / "planner.wasm").read_bytes(), 2: (PRIOR / "load-planner.wasm").read_bytes()}
            chooser = (PORTFOLIO / "chooser.wasm").read_bytes()
            training = json.loads((PORTFOLIO / "training.json").read_text())
            small = terrain.GUEST.read_bytes()
            large = (CALLS / "terrain-256.wasm").read_bytes()
            bfs = (ARCHIVE / "v58-stateful-scale-7125937/bfs.wasm").read_bytes()
            graph_argument, graph_page, *_ = graph_case(0)
            for name, raw, pages in (("terrain_64k", small, 1), ("terrain_16m", large, 256), ("bfs", bfs, 1)):
                if args.continuous and name == "bfs":
                    continue
                is_bfs = name == "bfs"
                started = perf_counter_ns()
                plain = load_call_image(raw, memory_pages=pages)
                counts = {int(k): n for k, n in training["bfs" if is_bfs else "terrain"]["pc_counts"].items()}
                layouts, _, selected, _ = prepare(plain, base, planners, chooser, counts)
                prior_prep_ns = perf_counter_ns() - started
                started = perf_counter_ns()
                lazy = descriptors(layouts["portfolio"])
                native = native_rows(lazy)
                prep_ns = perf_counter_ns() - started
                images = {"lazy": lazy, "native": native}
                if args.continuous:
                    result[name] = captures(graphs["native"], native, raw, 240)
                    continue
                audit_page = None
                runs = {}
                for phase in ((0,) if is_bfs else (0, 1)):
                    if is_bfs:
                        page, argument, moves, steps = graph_page, graph_argument, [None]*6, 41591
                    else:
                        scene = terrain.scale_session(8, False, 64)
                        page = terrain.initial_page(scene["initial"])
                        page.extend(bytes((pages - 1) * terrain.PAGE_BYTES))
                        argument, steps = terrain.BASE, 1650
                        moves = [[(-1 if phase else 1) * d for d in tick["moves"]]
                                 for tick in scene["inputs"]]
                        terrain.write(page, 320, [4 * d for d in moves[0]])
                    for mode, machine in graphs.items():
                        drive(machine, images[mode], argument, page)
                    if phase == 0:
                        audit_page = page.copy()
                    runs[str(phase)] = paired(graphs, images, raw, argument, page, moves, steps, not is_bfs)
                row = {"runs": runs, "wasm_sha256": sha(raw), "ordinary_descriptor_ns": prep_ns,
                       "prior_preparation_ns": prior_prep_ns, "prepared_image_sha256": native.image_sha256,
                       "descriptor_extra_bytes": len(native.rows) * 8}
                if name == "terrain_64k":
                    row["all_source_boundaries"] = audit(one, graphs["native"], plain, native, argument,
                                                        audit_page, 5000, set(selected))
                    row["both_histories"] = verify_histories(graphs, images, raw)
                if name != "terrain_16m":
                    row["profile"] = check_profiles(graphs, images, argument, audit_page, set(selected))
                    eligible = Counter()
                    for pc, kind in enumerate(plain.rows[:, 0].tolist()):
                        if int(native.rows[pc, 17]):
                            eligible[kind] += 1
                    row["admitted_non_memory_sites_by_opcode"] = dict(eligible)
                result[name] = row
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
