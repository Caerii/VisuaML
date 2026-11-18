"""Intervene on each live data-gate gain and compare the float64 and packed native outcomes."""
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

from scripts.benchmark_neural_vm_v58 import digest, git_identity, workloads
from visuaml.compiler_lab.neural_alu_v56 import construct_alu_weights
from visuaml.compiler_lab.neural_local_bank_v57 import construct_local_bank_weights
from visuaml.compiler_lab.neural_microstep_executor_v58 import PreparedMicrostepProgram
from visuaml.compiler_lab.neural_microstep_v58 import construct_microstep_weights
from visuaml.compiler_lab.neural_operand_storage_v55 import fixed_operand_weights
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlOperation, ControlProgram

factories = (
    ("microstep_fixed", construct_microstep_weights, ("pop_gain", "produce_gain", "write_gain")),
    ("operand_fixed", fixed_operand_weights, ("action_gain",)),
    ("alu_fixed", construct_alu_weights, ("output_gains",)),
    ("local_fixed", construct_local_bank_weights, ("read_gain", "write_gain")),
)


def outcome(program, value, native, options):
    try:
        result = PreparedMicrostepProgram(program, use_native_loop=True, use_packed_scalar=native, **options).run(value)
    except ValueError as err:
        return {"error": type(err).__name__, "message": str(err)}
    return {"result": result.result_i32, "trace_sha256": digest(result.trace)}


source = git_identity(Path.cwd())
programs = {work.name: work.program for work in workloads(scaling=True)}
ops = [ControlOperation("local.get", 0)]
for slot in range(1, 5):
    ops.extend((ControlOperation("local.tee", slot), ControlOperation("drop"),
                ControlOperation("local.get", slot)))
ops.append(ControlOperation("end"))
programs["all-locals"] = ControlProgram(4, tuple(ops))
rows = []
counts = Counter()
for name in ("generated_02", "heldout_133", "all-locals"):
    program = programs[name]
    for value in (-2147483648, 17):
        baseline = outcome(program, value, False, {})
        assert "error" not in baseline
        assert outcome(program, value, True, {}) == baseline
        for key, factory, fields in factories:
            for field in fields:
                fixed = factory()
                nominal = getattr(fixed, field)
                for index in range(nominal.size):
                    for replacement in (0.0, 0.5):
                        mutated = nominal.copy()
                        mutated[index] = replacement
                        options = {key: replace(fixed, **{field: mutated})}
                        reference = outcome(program, value, False, options)
                        native = outcome(program, value, True, options)
                        assert native == reference, (name, value, field, index, replacement, native, reference)
                        status = "trapped" if "error" in native else "changed" if native != baseline else "same"
                        counts[status] += 1
                        rows.append({
                            "program": name, "input": value, "field": field, "index": index,
                            "replacement": replacement, "outcome": native, "status": status,
                        })
with Path(sys.argv[1]).open("x") as stream:
    json.dump({"source": source, "counts": counts, "cases": rows}, stream, indent=2, sort_keys=True)
    stream.write("\n")
