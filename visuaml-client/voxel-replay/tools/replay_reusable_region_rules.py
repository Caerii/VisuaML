"""Proof-bound admission and paired replay of unchanged terrain and BFS."""
from collections import Counter
from dataclasses import replace
import json
import math
import statistics
from time import perf_counter_ns
from unittest.mock import patch

import torch
import wasmtime

import terrain_verify as terrain
from prove_reusable_region_rules import OUT, VERSION, binding, eager_counterexample
from replay_frozen_multistep import CALLS, state_equal
from replay_frozen_recurrent import ARCHIVE, PACKED, graph_case, sha, signed, wasmtime_run
from replay_guest_region_plan import OUT as PAIR, propose, prove as prior_proof
from replay_guest_rewrite import EXECUTOR
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image
from visuaml.compiler_lab.frozen_recurrent_v58 import MAX_LOCALS, OPCODES, snapshot_page

RULES = {"local.get": 1, "i32.load": 2}


def checked_certificate():
    path = OUT / "rules.json"
    raw = path.read_bytes()
    certificate = json.loads(raw)
    if (sha(raw) != (OUT / "rules.sha256").read_text().strip()
            or certificate["version"] != VERSION or certificate["sha256_bindings"] != binding()
            or len(certificate["obligations"]) != 5
            or any(row["result"] != "unsat" for row in certificate["obligations"])):
        raise ValueError("Unrecognized reusable rule certificate or graph semantics")
    return sha(raw)


def with_rows(image, rows):
    return replace(image, rows=rows, image_sha256=sha(rows.numpy().astype("<i8", copy=False).tobytes()))


def admit(image, baseline, planners):
    plan = torch.full((image.rows.size(0), 3), -1, dtype=torch.int64)
    emitted, matches = {}, {}
    started = perf_counter_ns()
    guest_ns, accepted = 0, Counter()
    for start, ops in zip(image.starts, image.functions, strict=True):
        for index, (first, second) in enumerate(zip(ops, ops[1:])):
            if first != "local.get" or second not in RULES:
                continue
            pc = start + index
            before, after = image.rows[pc:pc + 2].tolist()
            template = RULES[second]
            a, b = before[1], after[1] if template == 1 else after[2]
            if (before[0] != OPCODES.index(first) or after[0] != OPCODES.index(second)
                    or before[3:5] != [pc + 1] * 2 or after[3:5] != [pc + 2] * 2
                    or any(before[5:8]) or any(after[5:8]) or after[10]
                    or pc + 2 >= start + len(ops) or not 0 <= a <= MAX_LOCALS
                    or not 0 <= b <= (MAX_LOCALS if template == 1 else 0xFFFFFFFF)):
                raise ValueError("Source region fails certificate assumptions")
            key = template, a, b
            if key not in emitted:
                clock = perf_counter_ns()
                proposal = propose(baseline, planners[template], ((first, a), (second, signed(b))),
                                   second_kind=second)
                expected = (template, a, signed(b), a if template == 1 else 2, 1, 2)
                if proposal["plan_words"] != expected:
                    raise ValueError("Guest plan fails certified template binding")
                emitted[key] = proposal
                guest_ns += perf_counter_ns() - clock
            plan[pc, 0] = a
            plan[pc, 1] = b if template == 1 else -2
            if template == 2:
                plan[pc, 2] = b
            matches[pc] = key
            accepted[second] += 1
    only_pair = plan[:, :2].clone()
    only_pair[plan[:, 1] == -2] = -1
    images = (with_rows(image, torch.cat((image.rows, only_pair), 1)),
              with_rows(image, torch.cat((image.rows, plan), 1)))
    elapsed = perf_counter_ns() - started
    return images, {"sites": matches, "static_coverage": dict(accepted),
                    "other_source_rows": sum(map(len, image.functions)) - len(matches),
                    "site_admission_ns": elapsed - guest_ns, "proposal_and_oracle_ns": guest_ns,
                    "guest_proposals": list(emitted.values()), "image_sha256": [x.image_sha256 for x in images]}


def compare_access(ref, other):
    assert int(ref[0]) == int(other[0])
    if int(ref[0]):
        assert torch.equal(ref, other)


def audit(one, fast, image, planned, sites, argument, page, fuel):
    a = initialize(image, argument, page, fuel)
    b = initialize(planned, argument, page, fuel)
    counts = Counter()
    while int(a.vm.status) == 0:
        pc = int(a.vm.pc)
        ordinary = one(image.rows, a)
        cut = fast(planned.rows, b, 1)
        state_equal(ordinary.state, cut.state)
        compare_access(ordinary.access, cut.access)
        counts["checked_transitions"] += 1
        if pc in sites and int(ordinary.state.vm.status) == 0:
            second = one(image.rows, ordinary.state)
            candidate = fast(planned.rows, b, 2)
            state_equal(second.state, candidate.state)
            compare_access(second.access, candidate.access)
            counts["checked_transitions"] += 1
            counts["fused_pairs"] += candidate.fused_pairs
            counts["fused_loads"] += candidate.fused_loads
            a, b = second.state, candidate.state
        else:
            a, b = ordinary.state, cut.state
    return {**dict(counts), "source_steps": fuel - int(a.vm.fuel), "status": int(a.vm.status),
            "return_i32": signed(a.vm.stack[0]) if int(a.vm.status) == 1 else None,
            "page_sha256": sha(snapshot_page(a.vm))}


def execute(graph, image, argument, page, mode):
    state = initialize(image, argument, page, 100_000)
    calls = pairs = loads = 0
    started = perf_counter_ns()
    while int(state.vm.status) == 0:
        out = graph(image.rows, state, 64)
        state = out.state
        calls += 1
        if mode != "baseline":
            pairs += out.fused_pairs
        if mode == "both":
            loads += out.fused_loads
    elapsed = perf_counter_ns() - started
    assert int(state.vm.status) == 1
    return snapshot_page(state.vm), {"execution_ns": elapsed, "host_calls": calls, "fused_pairs": pairs,
                                     "fused_loads": loads, "source_steps": 100_000 - int(state.vm.fuel),
                                     "return_i32": signed(state.vm.stack[0])}


def bench(graphs, images, raw, argument, page, inputs, steps):
    samples = {mode: [] for mode in graphs}
    names = tuple(graphs)
    for i, moves in enumerate(inputs):
        if moves is not None:
            terrain.write(page, 320, [4 * move for move in moves])
        expected, oracle = wasmtime_run(raw, argument, page)
        rotation = i % len(graphs)
        for mode in names[rotation:] + names[:rotation]:
            changed, row = execute(graphs[mode], images[mode], argument, page, mode)
            assert changed == oracle and row["return_i32"] == expected and row["source_steps"] == steps
            samples[mode].append({**row, "page_sha256": sha(changed)})
        if moves is not None:
            page = bytearray(oracle)
    output = {}
    for mode, rows in samples.items():
        measured = sorted(row["execution_ns"] for row in (rows[1:] if len(rows) > 4 else rows))
        output[mode] = {"samples": len(measured), "median_ns": statistics.median(measured), "rows": rows}
        if len(measured) > 4:
            output[mode]["p95_ns"] = measured[math.ceil(.95 * len(measured)) - 1]
    return output


def histories(graphs, images, raw):
    runs = {}
    for name in ("terrain", "terrain-omitted"):
        proof = json.loads((OUT.parents[1] / "fixtures" / f"{name}.proof.json").read_text())
        page = terrain.initial_page(proof["session"]["initial"])
        runs[name] = []
        for inputs, recorded in zip(proof["session"]["inputs"], proof["ticks"], strict=True):
            terrain.write(page, 4, [int(inputs["edit"])])
            terrain.write(page, 320, [4 * delta for delta in inputs["moves"]])
            final, result = execute(graphs["both"], images["both"], terrain.BASE, page, "both")
            actual, baseline = execute(graphs["baseline"], images["baseline"], terrain.BASE, page, "baseline")
            assert actual == final and result["source_steps"] == baseline["source_steps"] == recorded["steps"]
            assert wasmtime_run(raw, terrain.BASE, page) == (result["return_i32"], final)
            assert result["return_i32"] == baseline["return_i32"]
            assert sha(final) == recorded["page_sha256"]
            runs[name].append({**result, "page_sha256": sha(final)})
            page = bytearray(final)
    return runs


def postfreeze(one, fast, raw, image, planned, sites, baseline, planner):
    counterexample = eager_counterexample()
    output, cases = {}, []
    for argument in (8, counterexample["argument"], -1):
        page = bytearray(65536)
        page[9:13] = bytes.fromhex("78563412")
        for fuel in (0, 1, 2, 100):
            audit_result = audit(one, fast, image, planned, sites, argument, page, fuel)
            if fuel == 100:
                expected, oracle = wasmtime_run(raw, argument, page, trap_expected=argument != 8)
                assert audit_result["return_i32"] == expected and audit_result["page_sha256"] == sha(oracle)
            cases.append({"argument": argument, "fuel": fuel, **audit_result})
    zero = bytes(65536)
    argument = counterexample["argument"]
    old = initialize(image, argument, zero, 1)
    ref_prefix = one(image.rows, old)
    exhausted = one(image.rows, ref_prefix.state)
    new = initialize(planned, argument, zero, 1)
    cutoff = fast(planned.rows, new, 64)
    state_equal(exhausted.state, cutoff.state)
    assert int(cutoff.state.vm.status) == 2 and int(cutoff.access[0]) == 0
    enough = initialize(image, argument, zero, 2)
    eager = one(image.rows, one(image.rows, enough).state)
    assert int(eager.state.vm.status) == 3 and int(eager.access[0]) == 3
    proposal = propose(baseline, planner, (("local.get", 0), ("i32.load", 1)),
                       challenge=True, second_kind="i32.load")
    assert proposal["plan_words"] == (2, 0, 1, 1, 1, 2)
    load_boundary = fast(planned.rows, initialize(planned, argument, zero, 1), 1)
    assert int(load_boundary.state.vm.status) == 0 and int(load_boundary.state.vm.fuel) == 0
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        stopped = fast(planned.rows, load_boundary.state, 64)
    exhaustion_ops = {event.key: event.count for event in profile.key_averages()}
    assert "aten::gather" not in exhaustion_ops and "aten::index_select" not in exhaustion_ops
    state_equal(stopped.state, cutoff.state)
    output["rejected"] = {"proposal": proposal, "proof": counterexample,
                          "exhaustion_operator_counts": exhaustion_ops,
                          "reference_status": 2, "reference_memory_event": 0,
                          "eager_status": 3, "eager_memory_event": eager.access.tolist(),
                          "first_boundary_pc": int(ref_prefix.state.vm.pc),
                          "first_boundary_stack": ref_prefix.state.vm.stack[:1].tolist()}
    output["cases"] = cases
    return output


def operator_profile(graphs, images, page, sites):
    state = initialize(images["baseline"], terrain.BASE, page, 100)
    ordinary = graphs["baseline"]
    while sites.get(int(state.vm.pc), (0,))[0] != 2:
        state = ordinary(images["baseline"].rows, state, 1).state
        assert int(state.vm.status) == 0
    counts = {}
    for mode in ("baseline", "both"):
        graph, image = graphs[mode], images[mode]
        for _ in range(5):
            graph(image.rows, state, 2)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            pulse = graph(image.rows, state, 2)
        if mode == "both":
            assert pulse.fused_loads == 1
        counts[mode] = {event.key: event.count for event in profile.key_averages()}
    return {"terrain_pc": int(state.vm.pc), "operators": counts}


def main():
    torch.set_num_threads(1)
    started = perf_counter_ns()
    certificate = checked_certificate()
    check_ns = perf_counter_ns() - started
    graphs = {name: torch.jit.load(str(path)) for name, path in (
        ("baseline", EXECUTOR / "machine.pt"), ("pair", PAIR / "machine.pt"), ("both", OUT / "machine.pt"))}
    one = torch.jit.load(str(CALLS / "machine.pt"))
    planners = {1: (PAIR / "planner.wasm").read_bytes(), 2: (OUT / "load-planner.wasm").read_bytes()}
    prior = json.loads((PAIR / "replay.json").read_text())
    rejected = prior["rejected"]
    slots = tuple(rejected["input_words"][i] for i in (1, 3))
    assert [row["result"] for row in prior_proof(slots, rejected["plan_words"])] == ["sat", "unsat"]
    raw = terrain.GUEST.read_bytes()
    bfs = (ARCHIVE / "v58-stateful-scale-7125937/bfs.wasm").read_bytes()
    later = (OUT / "after-freeze.wasm").read_bytes()
    assert later == bytes(wasmtime.wat2wasm((OUT / "after-freeze.wat").read_text()))
    output = {}
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        for name, binary in (("terrain", raw), ("bfs", bfs), ("after_freeze", later)):
            started = perf_counter_ns()
            image = load_call_image(binary)
            validate_ns = perf_counter_ns() - started
            (pair, both), preparation = admit(image, graphs["baseline"], planners)
            images = {"baseline": image, "pair": pair, "both": both}
            sites = preparation["sites"]
            row = {"binary_sha256": sha(binary), "validate_ns": validate_ns, "preparation": preparation}
            if name == "terrain":
                scene = terrain.scale_session(8, False, 25)
                page = terrain.initial_page(scene["initial"])
                terrain.write(page, 320, [4 * d for d in scene["inputs"][0]["moves"]])
                row["audit"] = audit(one, graphs["both"], image, both, sites, terrain.BASE, page, 5000)
                row["histories"] = histories(graphs, images, raw)
                row["operator_profile"] = operator_profile(graphs, images, page, sites)
                row["timings"] = bench(graphs, images, raw, terrain.BASE, page,
                                       [step["moves"] for step in scene["inputs"]], 1650)
            elif name == "bfs":
                argument, page, *_ = graph_case(0)
                row["timings"] = bench(graphs, images, bfs, argument, page, (None, None), 41591)
            else:
                row["replay"] = postfreeze(one, graphs["both"], later, image, both, sites,
                                           graphs["baseline"], planners[2])
            output[name] = row
    print(json.dumps({"certificate_sha256": certificate, "certificate_check_ns": check_ns,
                      "rule_version": VERSION, "planners": {i: sha(raw) for i, raw in planners.items()},
                      "previous_negative_pair_regression": "SAT at first boundary, UNSAT at second",
                      "workloads": output}, indent=2))


if __name__ == "__main__":
    main()
