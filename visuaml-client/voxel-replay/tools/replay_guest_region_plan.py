"""Guest proposal, independent prefix proof, lowering, and unchanged terrain replay."""
from dataclasses import replace
import json
import math
import statistics
import struct
from time import perf_counter_ns
from unittest.mock import patch

import torch
import z3

import terrain_verify as terrain
from replay_frozen_multistep import CALLS, state_equal
from replay_frozen_recurrent import PACKED, sha, wasmtime_run
from replay_guest_rewrite import EXECUTOR, invoke
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image
from visuaml.compiler_lab.frozen_recurrent_v58 import MAX_LOCALS, MAX_STACK, OPCODES, snapshot_page

OUT = EXECUTOR.parent / "guest-region-plan"
LOCAL_GET = OPCODES.index("local.get")
PAGE_SIZE = 65536


def propose(machine, planner, source, challenge=False):
    description = [word for opcode, slot in source for word in (int(opcode != "local.get"), slot)]
    words = [*description, int(challenge), *([-1] * 6)]
    page = bytearray(PAGE_SIZE)
    struct.pack_into("<11i", page, 512, *words)
    output, replay = invoke(machine, planner, 512, page)
    assert replay["status"] == 1
    emitted = struct.unpack_from("<6i", output, 532)
    assert replay["return_i32"] == emitted[0]
    assert wasmtime_run(planner, 512, page) == (emitted[0], output)
    assert output[:532] == page[:532] and output[556:] == page[556:]
    return {"input_words": words[:5], "plan_words": emitted, "replay": replay}


def prove(slots, plan):
    template, first, second, preview, one, two = plan
    assert template == 1 and all(0 <= i <= MAX_LOCALS for i in (*slots, first, second, preview))
    locals_ = [z3.BitVec(f"local_{i}", 32) for i in range(MAX_LOCALS + 1)]
    stack = [z3.BitVec(f"stack_{i}", 32) for i in range(MAX_STACK)]
    memory = z3.Array("memory", z3.IntSort(), z3.BitVecSort(8))
    frames = z3.Array("frames", z3.IntSort(), z3.BitVecSort(64))
    height, pc, fuel, depth = z3.Ints("height pc fuel depth")
    zero = z3.BitVecVal(0, 32)
    # The reference implements both push and the canonical stack-tail mask.
    def push(active, at, value):
        return [z3.If(i < at, cell, z3.If(i == at, value, zero))
                for i, cell in enumerate(active)]

    shared = (memory, frames, depth, *locals_, 0, 0)
    after_first = push(stack, height, locals_[slots[0]])
    reference = ((after_first, height + 1, pc + 1, fuel - 1),
                 (push(after_first, height + 1, locals_[slots[1]]), height + 2, pc + 2, fuel - 2))
    # The plan's indexed scatter carries all untouched cells through.
    preview_stack = [z3.If(i == height, locals_[preview], cell) for i, cell in enumerate(stack)]
    final_stack = [z3.If(i == height, locals_[first], z3.If(i == height + 1, locals_[second], cell))
                   for i, cell in enumerate(stack)]
    candidate = ((preview_stack, height + 1, pc + 1, fuel - one),
                 (final_stack, height + 2, pc + 2, fuel - two))
    output = []
    for boundary, (ref, plan_state) in enumerate(zip(reference, candidate, strict=True), 1):
        solver = z3.Solver()
        solver.add(height >= 0, height <= MAX_STACK - 2, pc >= 0, fuel >= 2)
        solver.add(*(z3.Implies(i >= height, cell == zero) for i, cell in enumerate(stack)))
        lhs = (*ref[0], *ref[1:], *shared)
        rhs = (*plan_state[0], *plan_state[1:], *shared)
        solver.add(z3.Or(*(a != b for a, b in zip(lhs, rhs, strict=True))))
        result = solver.check()
        record = {"source_boundary": boundary, "result": str(result), "query": solver.to_smt2()}
        if result == z3.sat:
            model = solver.model()
            record["counterexample"] = {
                "locals_u32": [model.eval(x, model_completion=True).as_long() for x in locals_],
                "height": model.eval(height).as_long(), "fuel": model.eval(fuel).as_long(),
                "reference_pushed_u32": model.eval(locals_[slots[0]], model_completion=True).as_long(),
                "plan_pushed_u32": model.eval(locals_[preview], model_completion=True).as_long(),
            }
        else:
            assert result == z3.unsat
        output.append(record)
    return output


def prepare(machine, planner, image):
    metadata = torch.full((image.rows.size(0), 2), -1, dtype=torch.int64)
    proposals, sites = {}, {}
    planner_ns, proof_ns = 0, 0
    for start, functions in zip(image.starts, image.functions, strict=True):
        for offset, pair in enumerate(zip(functions, functions[1:])):
            if pair != ("local.get", "local.get"):
                continue
            pc = start + offset
            row, successor = image.rows[pc:pc + 2]
            # Admit only two decoded, normal fall-through, nonterminal rows.
            assert int(row[0]) == int(successor[0]) == LOCAL_GET
            assert row[3:5].tolist() == [pc + 1] * 2
            assert successor[3:5].tolist() == [pc + 2] * 2
            assert not row[5:8].any() and not successor[5:8].any()
            assert int(successor[10]) == 0 and pc + 2 < start + len(functions)
            slots = int(row[1]), int(successor[1])
            if slots not in proposals:
                source = list(zip(pair, slots, strict=True))
                started = perf_counter_ns()
                emission = propose(machine, planner, source)
                planner_ns += perf_counter_ns() - started
                started = perf_counter_ns()
                checks = prove(slots, emission["plan_words"])
                proof_ns += perf_counter_ns() - started
                if any(check["result"] != "unsat" for check in checks):
                    raise ValueError("Region planner emitted a candidate that failed the prefix proof")
                proposals[slots] = {**emission, "proof": checks}
            plan = proposals[slots]["plan_words"]
            # Only the fixed proven template ABI is supported by this lowering.
            assert plan == (1, *slots, slots[0], 1, 2)
            metadata[pc] = torch.tensor(plan[1:3])
            sites[pc] = slots
    rows = torch.cat((image.rows, metadata), dim=1)
    digest = sha(rows.numpy().astype("<i8", copy=False).tobytes())
    return replace(image, rows=rows, image_sha256=digest), sites, proposals, planner_ns, proof_ns


def audit_boundaries(one, graph, ordinary, planned, sites, page):
    a = initialize(ordinary, terrain.BASE, page, 5000)
    b = initialize(planned, terrain.BASE, page, 5000)
    pairs, boundaries = 0, 0
    while int(a.vm.status) == 0:
        pc = int(a.vm.pc)
        if pc in sites:
            middle = one(ordinary.rows, a)
            fallback = graph(planned.rows, b, 1)
            assert fallback.fused_pairs == 0 and int(middle.access[0]) == int(fallback.access[0]) == 0
            state_equal(middle.state, fallback.state)
            # The same source prefix also has to survive an exhausted fuel
            # budget inside a multi-step invocation.
            if pairs == 0:
                def cut(state):
                    return state._replace(vm=state.vm._replace(fuel=torch.tensor(1)))
                expected = one(ordinary.rows, one(ordinary.rows, cut(a)).state).state
                stopped = graph(planned.rows, cut(b), 64)
                assert stopped.fused_pairs == 0 and int(stopped.state.vm.status) == 2
                state_equal(expected, stopped.state)
            a = one(ordinary.rows, middle.state).state
            candidate = graph(planned.rows, b, 2)
            assert candidate.fused_pairs == 1
            pairs += 1
            boundaries += 2
        else:
            reference = one(ordinary.rows, a)
            a = reference.state
            candidate = graph(planned.rows, b, 1)
            assert candidate.fused_pairs == 0
            if int(reference.access[0]):
                assert torch.equal(reference.access, candidate.access)
            else:
                assert int(candidate.access[0]) == 0
            boundaries += 1
        b = candidate.state
        state_equal(a, b)
    return {"source_boundaries": boundaries, "pair_boundaries": pairs,
            "memory_sha256": sha(snapshot_page(b.vm))}


def negative_replay(one, image, site, slots, counterexample, page):
    state = initialize(image, terrain.BASE, page, counterexample["fuel"])
    vm = state.vm._replace(pc=torch.tensor(site), height=torch.tensor(counterexample["height"]),
                           locals=torch.tensor(counterexample["locals_u32"], dtype=torch.int64))
    state = state._replace(vm=vm)
    mid = one(image.rows, state).state
    end = one(image.rows, mid).state
    height = int(vm.height)
    wrong_prefix = int(vm.locals[slots[1]])
    assert int(mid.vm.stack[height]) == counterexample["reference_pushed_u32"] != wrong_prefix
    final_pair = end.vm.stack[height:height + 2].tolist()
    assert final_pair == vm.locals[list(slots)].tolist()
    return {"site": site, "prefix_reference": int(mid.vm.stack[height]),
            "prefix_candidate": wrong_prefix, "final_pair_both": final_pair,
            "fuel_after_prefix": int(mid.vm.fuel), "fuel_after_pair": int(end.vm.fuel)}


def local_profile(baseline, specialized, ordinary, planned, sites, page):
    state = initialize(ordinary, terrain.BASE, page, 5000)
    while int(state.vm.pc) not in sites:
        state = baseline(ordinary.rows, state, 1).state
        assert int(state.vm.status) == 0
    site = int(state.vm.pc)
    expected = baseline(ordinary.rows, state, 2)
    accelerated = specialized(planned.rows, state, 2)
    assert accelerated.fused_pairs == 1
    state_equal(expected.state, accelerated.state)
    variants = (("baseline", baseline, ordinary), ("specialized", specialized, planned))
    profile, raw = {}, {name: [] for name, _, _ in variants}
    for name, graph, image in variants:
        for _ in range(8):
            graph(image.rows, state, 2)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as records:
            graph(image.rows, state, 2)
        profile[name] = {event.key: event.count for event in records.key_averages()}
    for sample in range(31):
        for name, graph, image in variants[::(1 if sample % 2 == 0 else -1)]:
            started = perf_counter_ns()
            for _ in range(16):
                graph(image.rows, state, 2)
            raw[name].append((perf_counter_ns() - started) / 16)
    return {"site": site, "slots": sites[site], "samples": 31, "calls_per_sample": 16,
            "median_ns": {name: statistics.median(values) for name, values in raw.items()},
            "raw_ns_per_call": raw, "runtime_operator_counts": profile}


def execute(machine, image, page, planned=False):
    state = initialize(image, terrain.BASE, page, 5000)
    count = pairs = 0
    started = perf_counter_ns()
    while int(state.vm.status) == 0:
        tick = machine(image.rows, state, 64)
        state = tick.state
        if planned:
            pairs += tick.fused_pairs
        count += 1
    elapsed = perf_counter_ns() - started
    assert int(state.vm.status) == 1
    return snapshot_page(state.vm), {"execution_ns": elapsed, "source_steps": 5000 - int(state.vm.fuel),
                                     "host_calls": count, "fused_pairs": pairs, "result": int(state.vm.stack[0])}


def benchmark(baseline, specialized, raw, ordinary, planned):
    scene = terrain.scale_session(8, False, 25)
    page = terrain.initial_page(scene["initial"])
    timings = {"baseline": [], "specialized": []}
    cases = (("baseline", baseline, ordinary), ("specialized", specialized, planned))
    for tick, inputs in enumerate(scene["inputs"]):
        terrain.write(page, 320, [4 * move for move in inputs["moves"]])
        expected, oracle = wasmtime_run(raw, terrain.BASE, page)
        for name, graph, image in cases[::(1 if tick % 2 == 0 else -1)]:
            changed, row = execute(graph, image, page, planned=name == "specialized")
            assert changed == oracle and row["result"] == expected and row["source_steps"] == 1650
            timings[name].append(row)
        page = bytearray(oracle)
    return {name: {"warm_samples": len(rows) - 1, "median_ns": statistics.median(
                        row["execution_ns"] for row in rows[1:]),
                   "p95_ns": sorted(row["execution_ns"] for row in rows[1:])[
                       math.ceil(.95 * (len(rows) - 1)) - 1],
                   "rows": rows}
            for name, rows in timings.items()}


def histories(baseline, specialized, raw, ordinary, planned):
    output = {}
    for name in ("terrain", "terrain-omitted"):
        proof = json.loads((OUT.parents[1] / "fixtures" / f"{name}.proof.json").read_text())
        page = terrain.initial_page(proof["session"]["initial"])
        output[name] = []
        for inputs, expected in zip(proof["session"]["inputs"], proof["ticks"], strict=True):
            terrain.write(page, 4, [int(inputs["edit"])])
            terrain.write(page, 320, [4 * move for move in inputs["moves"]])
            a, base = execute(baseline, ordinary, page)
            b, fast = execute(specialized, planned, page, planned=True)
            assert a == b
            assert base["source_steps"] == fast["source_steps"] == expected["steps"]
            assert base["result"] == fast["result"]
            assert wasmtime_run(raw, terrain.BASE, page) == (fast["result"], b)
            assert sha(a) == expected["page_sha256"]
            output[name].append({"source_steps": fast["source_steps"], "result": fast["result"],
                                 "fused_pairs": fast["fused_pairs"], "page_sha256": sha(a)})
            page = bytearray(a)
    return output


def main():
    torch.set_num_threads(1)
    load_started = perf_counter_ns()
    baseline = torch.jit.load(str(EXECUTOR / "machine.pt"))
    one = torch.jit.load(str(CALLS / "machine.pt"))
    path = OUT / "machine.pt"
    specialized = torch.jit.load(str(path))
    load_ns = perf_counter_ns() - load_started
    ir = str(specialized.inlined_graph)
    assert "prim::PythonOp" not in ir and "aten::scatter(" in ir and "aten::scatter_(" in ir
    planner = (OUT / "planner.wasm").read_bytes()
    raw = terrain.GUEST.read_bytes()
    started = perf_counter_ns()
    ordinary = load_call_image(raw)
    validate_ns = perf_counter_ns() - started
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        started = perf_counter_ns()
        planned, sites, accepted, planner_ns, proof_ns = prepare(baseline, planner, ordinary)
        prepare_ns = perf_counter_ns() - started
        slots = next(slots for slots in accepted if slots[0] != slots[1])
        source = [("local.get", slot) for slot in slots]
        negative = propose(baseline, planner, source, challenge=True)
        checks = prove(slots, negative["plan_words"])
        assert [check["result"] for check in checks] == ["sat", "unsat"]
        declined = propose(baseline, planner, (source[0], ("i32.load", 0)))
        assert declined["plan_words"][0] == 0
        scene = terrain.scale_session(8, False, 1)
        page = terrain.initial_page(scene["initial"])
        terrain.write(page, 320, [4 * move for move in scene["inputs"][0]["moves"]])
        rejected_site = next(pc for pc, members in sites.items() if members == slots)
        rejected_replay = negative_replay(one, ordinary, rejected_site, slots,
                                          checks[0]["counterexample"], page)
        local = local_profile(baseline, specialized, ordinary, planned, sites, page)
        audit = audit_boundaries(one, specialized, ordinary, planned, sites, page)
        solved = histories(baseline, specialized, raw, ordinary, planned)
        timings = benchmark(baseline, specialized, raw, ordinary, planned)
    print(json.dumps({"wasm_sha256": sha(raw), "image_sha256": ordinary.image_sha256,
                      "plan_image_sha256": planned.image_sha256, "source_pcs": sites,
                      "planner_sha256": sha(planner), "graph_sha256": sha(path.read_bytes()),
                      "operators": sorted(torch.jit.export_opnames(specialized)),
                      "accepted": list(accepted.values()), "rejected": {**negative, "proof": checks, "counterexample_replay": rejected_replay},
                      "declined": declined, "audit": audit, "histories": solved,
                      "graph_load_total_ns": load_ns, "validate_ns": validate_ns, "prepare_ns": prepare_ns,
                      "planner_and_oracle_ns": planner_ns, "proof_ns": proof_ns,
                      "local_profile": local, "timings": timings}, indent=2))


if __name__ == "__main__":
    main()
