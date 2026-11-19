"""Profile and count fixed ALU candidates independently of timing pairs."""
import argparse
import cProfile
import json
import pstats
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_neural_vm_v58 import environment
from visuaml.compiler_lab import neural_alu_v56 as alu
from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted

from operand_pairs import workload
from operand_pairs import run as run_terrain

CIRCUITS = {"_add", "_sub", "_mul", "_signed_lt", "_eqz", "_and", "_xor", "_unsigned_lt",
            "_admit_binary", "execute_alu", "_run_prepared_alu", "stack", "bincount"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", action="store_true")
    args = parser.parse_args()
    report = {}
    for name in ("sparse", "crowded", "bfs", "sudoku"):
        case = workload(name)

        def run():
            if isinstance(case, dict):
                return run_terrain(case, [], None, prepare_operand_pop=True, prepare_alu=args.prepared)
            program, page, argument, _, _, steps, _ = case
            fuel = min(4000, steps)
            try:
                return vm.execute_microop_routed_control(
                    program, argument, array_memory=page.copy(), capture_trace=False,
                    fuel=fuel, prepare_alu=args.prepared,
                )
            except ControlFuelExhausted:
                assert fuel < steps
                return None

        selected = Counter()
        original = alu.execute_alu
        original_prepared = alu._run_prepared_alu
        direct = Counter()

        def count(selector, *operands, **kwargs):
            selected[alu.ALU_OPS[int(selector.argmax())]] += 1
            return original(selector, *operands, **kwargs)

        def count_prepared(circuit, *operands):
            name = {"_add": "i32.add", "_less_than_word": "i32.lt_s"}[circuit.__name__]
            selected[name] += 1
            direct[name] += 1
            return original_prepared(circuit, *operands)

        with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
                   side_effect=AssertionError("Unexpected packed entry")):
            run()
            profiler = cProfile.Profile()
            with patch.object(vm, "execute_alu", count), patch.object(
                alu, "_run_prepared_alu", count_prepared,
            ):
                profiler.runcall(run)
        rows = pstats.Stats(profiler)
        report[name] = {
            "selected": dict(selected), "prepared_selected": dict(direct), "profile_seconds": rows.total_tt,
            "functions": [{"file": Path(path).name, "function": function, "calls": calls,
                           "self_seconds": own, "cumulative_seconds": cumulative}
                          for (path, _, function), (_, calls, own, cumulative, _) in rows.stats.items()
                          if function in CIRCUITS],
        }
    print(json.dumps({"environment": environment(), "prepared": args.prepared, "cases": report}, indent=2))


if __name__ == "__main__":
    main()
