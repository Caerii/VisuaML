"""Measure sustained terrain replay and compare its hot path with archived BFS."""
import argparse
import cProfile
import json
import pstats
from pathlib import Path
from time import perf_counter_ns

from scripts.benchmark_neural_vm_v58 import environment
from scripts.stateful_scale_workloads_v58 import graph_case
from visuaml.compiler_lab.neural_microop_executor_v57 import execute_microop_routed_control
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted, decode_control

from terrain_verify import scale_session, summarize, verify

ARCHIVE = Path(__file__).resolve().parents[3] / "benchmarks/results/v58-stateful-scale-7125937"


def hot_paths(call):
    profile = cProfile.Profile()
    profile.runcall(call)
    stats = pstats.Stats(profile).stats
    functions = {
        "signed_comparison": ("bit_neural_vm_v49.py", "_signed_lt"),
        "array_driver": ("neural_microop_executor_v57.py", "execute_microop_routed_control"),
        "fixed_alu": ("neural_alu_v56.py", "execute_alu"),
        "frame_transition": ("neural_frame_circuit_v53.py", "step_frames"),
        "operand_transition": ("neural_operand_storage_v55.py", "step_operands"),
        "route": ("neural_microop_executor_v57.py", "_array_route"),
    }
    total = sum(row[2] for row in stats.values())
    return {"profile_ms": total * 1000, "functions": {
        key: {"calls": calls, "self_ms": self_time * 1000,
              "cumulative_ms": cumulative * 1000, "cumulative_share": cumulative / total}
        for key, (file, name) in functions.items()
        for (path, _, function), (_, calls, self_time, cumulative, _) in stats.items()
        if Path(path).name == file and function == name
    }}


def bfs():
    started = perf_counter_ns()
    program = decode_control((ARCHIVE / "bfs.wasm").read_bytes(), scaled=True)
    argument, page, *_ = graph_case(0)
    preparation_ns = perf_counter_ns() - started

    def run():
        try:
            execute_microop_routed_control(program, argument, array_memory=page, fuel=4000,
                                           capture_trace=False)
        except ControlFuelExhausted:
            return
        raise AssertionError("Expected fuel cutoff in the archived BFS")

    return {"source_steps": 4000, "decode_and_input_ns": preparation_ns, **hot_paths(run)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    if args.ticks < 4:
        parser.error("Use at least four ticks")
    args.output.mkdir()
    report = {"environment": environment(), "cases": {}}
    if not args.profile_only:
        for agents in (2, 4, 8):
            for crowded in (False, True):
                name = f"{agents}-{'crowded' if crowded else 'sparse'}"
                proof = verify(scale_session(agents, crowded, args.ticks))
                summary = summarize(proof)
                expected_conflicts = agents * sum(i % 4 < 2 for i in range(args.ticks)) if crowded else 0
                assert summary["agent_ticks_in_conflict"] == expected_conflicts
                report["cases"][name] = summary
                (args.output / f"{name}.json").write_text(json.dumps(proof, indent=2) + "\n")
                print(name, summary, flush=True)
    report["profiles"] = {
        "crowded_world_8": hot_paths(lambda: verify(scale_session(8, True, 4))),
        "bfs_prefix": bfs(),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
