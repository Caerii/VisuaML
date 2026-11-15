"""v10 integration gates: canonical, schedule, hybrid, optical, failure modes."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from visuaml.compiler_spike import OpRecord, ProgramIR, TensorSpec, capture
from visuaml.compiler_lab.assembly_schedule import (
    Area, AssemblySchedule, Fiber, Project, lower_assembly_schedule,
)
from visuaml.compiler_lab.effect_ir import EffectIRError, recurrent_assembly_effect_program
from visuaml.compiler_lab.hybrid_execution import HybridExecutionError, execute_hybrid
from visuaml.compiler_lab.optical_dialect import (
    OpticalContractError, OpticalProfile, lower_linear_to_optical_surrogate,
    run_optical_surrogate,
)
from visuaml.compiler_lab.schedule_sim import ScheduleError, estimate_schedule
from visuaml.compiler_lab.semantic_ir import (
    SemanticIRError, canonicalize_effect_ir, canonicalize_tensor_ir,
    validate_semantic_program,
)
from visuaml.compiler_lab.v10_lab import run_v10_lab


class Tiny(nn.Module):
    def forward(self, x, y):
        a = torch.matmul(x, y)
        return torch.relu(a) + 1.0


def example_capture(seed=2):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(2, 4)).astype(np.float32))
    y = torch.tensor(rng.normal(size=(4, 3)).astype(np.float32))
    return capture(Tiny(), x, y), x, y


def example_schedule(seed=5):
    spec = AssemblySchedule(
        (Area("A", 6, 2), Area("B", 5, 2), Area("C", 8, 3)),
        (Fiber("A", "C", .1), Fiber("B", "C", .2), Fiber("C", "A", .15)),
        (Project("C", ("A", "B")), Project("A", ("C",))), 3,
    )
    rng = np.random.default_rng(seed)
    state = {f"{a.name}.winners": np.arange(a.winners, dtype=np.int64) for a in spec.areas}
    sizes = {a.name: a.population for a in spec.areas}
    for f in spec.fibers:
        state[f"{f.source}.{f.target}.weights"] = rng.integers(0, 4, (sizes[f.source], sizes[f.target])).astype(np.float32)
    return spec, state


@pytest.mark.parametrize("seed", [1, 5, 11, 42])
def test_same_canonical_envelope_for_two_frontends(seed):
    c, _, _ = example_capture(seed)
    t = canonicalize_tensor_ir(c.ir)
    e = canonicalize_effect_ir(recurrent_assembly_effect_program(8, 2, steps=3, beta=.1, refraction_strength=.03))
    assert t.schema == e.schema == "visuaml.semantic-envelope.v1"
    assert t.origin == "torch-fx" and e.origin == "effect-program"
    assert len(t.digest()) == len(e.digest()) == 64
    assert any(op.op == "matmul" for op in t.regions[0].operations)
    assert any(op.effect == "write" for op in e.regions[0].operations)
    validate_semantic_program(t)
    validate_semantic_program(e)
    assert t.digest() == canonicalize_tensor_ir(c.ir).digest()


def test_canonical_rejects_bad_types_dangling_ssa_and_effect_token():
    c, _, _ = example_capture()
    orig = canonicalize_tensor_ir(c.ir)
    op = orig.regions[0].operations[0]
    invalid = replace(orig, regions=(replace(orig.regions[0], operations=(replace(op, operands=("undefined",)),) + orig.regions[0].operations[1:]),))
    with pytest.raises(SemanticIRError, match="Use before"):
        validate_semantic_program(invalid)
    effect = canonicalize_effect_ir(recurrent_assembly_effect_program(8, 2, steps=3, beta=.1, refraction_strength=.03))
    old = effect.regions[0]
    ops = list(old.operations)
    ops[0] = replace(ops[0], token_in="stale")
    with pytest.raises(SemanticIRError, match="effect"):
        validate_semantic_program(replace(effect, regions=(replace(old, operations=tuple(ops)),)))


@pytest.mark.parametrize("seed", [0, 4, 17, 25, 44])
def test_general_schedule_matches_independent_scalar_oracle(seed):
    spec, state = example_schedule(seed)
    lowered = lower_assembly_schedule(spec)
    current = {k: v.copy() for k, v in state.items()}
    for _ in range(spec.rounds):
        for step in spec.projections:
            drive = np.zeros(next(a.population for a in spec.areas if a.name == step.target), dtype=np.float32)
            for source in step.sources:
                for index in current[f"{source}.winners"]:
                    drive += current[f"{source}.{step.target}.weights"][int(index)]
            k = next(a.winners for a in spec.areas if a.name == step.target)
            ranked = sorted(range(len(drive)), key=lambda i: (-float(drive[i]), i))
            winners = np.asarray(sorted(ranked[:k]), dtype=np.int64)
            for source in step.sources:
                fiber = next(f for f in spec.fibers if f.source == source and f.target == step.target)
                w = current[f"{source}.{step.target}.weights"]
                for i in current[f"{source}.winners"]:
                    for j in winners:
                        w[int(i),int(j)] = np.float32(min(fiber.ceiling, float(np.float32(w[int(i), int(j)] * np.float32(1+fiber.beta)))))
            current[f"{step.target}.winners"] = winners
    result = lowered.execute(state)
    assert len(result.receipts) == spec.rounds
    for key in state:
        np.testing.assert_allclose(result.final_state[key], current[key], atol=1e-6, rtol=0)
        assert not np.shares_memory(state[key], result.final_state[key])


def test_schedule_supports_source_order_and_rejects_dangling_or_duplicate_sources():
    spec, state = example_schedule()
    alt = replace(spec, projections=(Project("C", ("B", "A")), *spec.projections[1:]))
    assert lower_assembly_schedule(spec).program.digest() != lower_assembly_schedule(alt).program.digest()
    with pytest.raises(EffectIRError, match="Projection"):
        lower_assembly_schedule(replace(spec, projections=(Project("C", ("A", "A")),)))
    with pytest.raises(EffectIRError, match="Projection"):
        lower_assembly_schedule(replace(spec, projections=(Project("B", ("C",)),)))
    state["A.winners"] = np.array([0,0], np.int64)
    with pytest.raises(EffectIRError, match="winner"):
        lower_assembly_schedule(spec).execute(state)


def test_generalized_schedule_two_area_reciprocal_and_multifiber_bridge_parity():
    from visuaml.compiler_lab.multifiber_bridge import MultiFiberConfig, execute_multifiber
    cfg = MultiFiberConfig(6, 5, 8, 2, 2, 3, .1, .2)
    spec, state = example_schedule()
    compact = replace(spec, projections=(spec.projections[0],), rounds=2)
    got = lower_assembly_schedule(compact).execute(state)
    mb_state = {
        "a.winners": state["A.winners"], "b.winners": state["B.winners"],
        "c.winners": state["C.winners"], "ac.weights": state["A.C.weights"],
        "bc.weights": state["B.C.weights"],
    }
    reference = execute_multifiber(cfg, mb_state, steps=2)
    np.testing.assert_array_equal(got.final_state["C.winners"], reference.final_state["c.winners"])
    np.testing.assert_array_equal(got.final_state["A.C.weights"], reference.final_state["ac.weights"])
    np.testing.assert_array_equal(got.final_state["B.C.weights"], reference.final_state["bc.weights"])


@pytest.mark.parametrize("seed", [2, 3, 5, 10])
def test_hybrid_true_software_dispatch_and_witness(seed):
    model, x, y = example_capture(seed)
    out, check = execute_hybrid(model, x, y, placements={"matmul": "device", "relu": "device"})
    assert check["passed"]
    assert [event.adapter for event in out.events] == ["device", "device", "host"]
    assert any(t.direction == "host->device" for t in out.transfers)
    assert any(t.direction == "device->host" for t in out.transfers)
    assert out.as_dict()["total_transfer_bytes"] > 0
    assert out.wall_ns > 0 and out.peak_live_bytes > 0


def test_hybrid_refuses_wrong_input_and_nonexistent_backend():
    program, x, y = example_capture()
    with pytest.raises(HybridExecutionError, match="shape or dtype"):
        execute_hybrid(program, x.double(), y)
    with pytest.raises(HybridExecutionError, match="nonexistent"):
        execute_hybrid(program, x, y, placements={"phantom":"device"})
    with pytest.raises(HybridExecutionError, match="budget"):
        execute_hybrid(program, x, y, max_retained_bytes=10)


def schedule_graph():
    spec = (TensorSpec(("2","3"),"torch.float32",24),)
    nodes = (
        OpRecord("x", "input", "", (), spec, 0,()),
        OpRecord("a", "matmul", "", ("x",), spec, 10,()),
        OpRecord("b", "relu", "", ("x",), spec, 10,()),
        OpRecord("c", "add", "", ("a", "b"), spec, 10,()),
        OpRecord("out", "output", "", ("c",), (), 0,()),
    )
    return ProgramIR(nodes, ("x",), ("c",))


def test_schedule_explicit_lanes_allow_independent_overlap():
    ir = schedule_graph()
    loc = {"x": "host", "a": "device", "b": "host", "c": "device", "out": "host"}
    result = estimate_schedule(ir, loc, {"a": 10, "b": 12, "c": 9}, transfer_us=2, us_per_byte=.1)
    assert result.transfers == 3
    assert result.makespan_us < result.total_work_us
    assert result.device_residency_upper_bound == 96
    assert len(result.input_sha256) == 64


def test_schedule_rejects_missing_costs_memory_budget_unknown_tensor():
    ir = schedule_graph()
    loc = {"x": "host", "a": "device", "b": "host", "c": "device", "out": "host"}
    with pytest.raises(ScheduleError, match="cover"):
        estimate_schedule(ir, loc, {"a":1}, transfer_us=1, us_per_byte=.1)
    with pytest.raises(ScheduleError, match="memory"):
        estimate_schedule(ir, loc, {"a": 1, "b": 1, "c": 1}, transfer_us=1, us_per_byte=.1,device_memory_budget=20)
    damaged = list(ir.nodes)
    damaged[1] = replace(damaged[1],tensors=(TensorSpec(("?",),"torch.float32",None),))
    with pytest.raises(ScheduleError, match="Unknown"):
        estimate_schedule(replace(ir,nodes=tuple(damaged)), loc, {"a": 1, "b": 1, "c": 1}, transfer_us=1, us_per_byte=.1)


def test_optical_ideal_linear_equivalence_and_nonideal_drift():
    torch.manual_seed(10)
    model = nn.Linear(4,3)
    plan = lower_linear_to_optical_surrogate(model)
    x = np.random.default_rng(4).normal(size=(6,4)).astype(np.float64)
    ideal = run_optical_surrogate(plan,x,OpticalProfile())
    np.testing.assert_allclose(ideal.result,model(torch.tensor(x,dtype=torch.float32)).detach().numpy(),atol=1e-6)
    noisy = run_optical_surrogate(plan,x,OpticalProfile(attenuation_db=1, phase_bits=3, gaussian_field_noise=.02),seed=7)
    assert noisy.max_abs_error > ideal.max_abs_error
    assert noisy.report()["source_kind"] == "synthetic"
    assert np.array_equal(run_optical_surrogate(plan,x,noisy.profile,seed=7).result,noisy.result)


def test_optical_fail_closed_when_gain_exceeds_budget_and_bad_profile():
    layer = nn.Linear(3,2)
    with torch.no_grad():layer.weight.mul_(1000)
    with pytest.raises(OpticalContractError, match="gain"):
        lower_linear_to_optical_surrogate(layer,max_amplitude_gain=1)
    with pytest.raises(OpticalContractError, match="Phase"):
        OpticalProfile(phase_bits=0)
    with pytest.raises(OpticalContractError):
        OpticalProfile(source_kind="measured")


def test_v10_lab_is_reproducible_and_json_safe():
    import json
    result=run_v10_lab(seed=23)
    assert result["hybrid"]["witness"]["passed"]
    assert len(result["stateful"]["iterations"]) == 3
    assert result["scheduled"]["transfers"] > 0
    assert result["semantic"]["effect"]["sha256"] == run_v10_lab(seed=23)["semantic"]["effect"]["sha256"]
    assert result["optical"]["source_kind"] == "synthetic"
    json.dumps(result,allow_nan=False)
    with pytest.raises(ValueError):run_v10_lab(seed=True)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("neural_assemblies") is None,
    reason="Pinned Caerii/assemblies library is required for actual upstream schedule parity",
)
def test_generic_schedule_vs_pinned_explicit_reference():
    """Three areas, two ordered projections per round, distinct fiber rates."""
    from neural_assemblies.core.numpy_engine._explicit import NumpyExplicitEngine

    schedule, initial = example_schedule(33)
    engine = NumpyExplicitEngine(p=.25, seed=7, w_max=20.0)
    for area in schedule.areas:
        engine.add_area(area.name, area.population, area.winners, .05)
        engine.set_winners(area.name, initial[f"{area.name}.winners"])
    for fiber in schedule.fibers:
        engine.set_beta(fiber.target, fiber.source, fiber.beta)
        engine._area_conns[fiber.source][fiber.target].weights[:] = initial[
            f"{fiber.source}.{fiber.target}.weights"]
    for _ in range(schedule.rounds):
        for projection in schedule.projections:
            engine.project_into(projection.target, [], list(projection.sources))
    compiler = lower_assembly_schedule(schedule).execute(initial).final_state
    for area in schedule.areas:
        np.testing.assert_array_equal(np.sort(engine.get_winners(area.name)), compiler[f"{area.name}.winners"])
    for fiber in schedule.fibers:
        name = f"{fiber.source}.{fiber.target}.weights"
        np.testing.assert_allclose(engine._area_conns[fiber.source][fiber.target].weights, compiler[name], atol=1e-6, rtol=0)


def test_v10_integrates_into_existing_allowlisted_service():
    from scripts.compiler_lab_service import analyze

    report = analyze("mlp", seed=7)
    assert report["schema"] if "schema" in report else report["api_contract"]["schema"]
    assert report["v10"]["schema"] == "visuaml.compiler-lab.v10-integration.v1"
    assert report["v10"]["hybrid"]["witness"]["passed"]
    assert report["api_contract"]["source"] == "fixed-local-demo-fixtures"
    with pytest.raises(ValueError, match="Scenario"):
        analyze("../../untrusted")
