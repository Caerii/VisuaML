"""v57 generated source WASM compositional regression.

This test reuses v56's seeded programs, changing ONLY the executor that
owns the source-bound typed microoperation decoder. It must be run locally.
"""
from __future__ import annotations

import pytest
from visuaml.compiler_lab.neural_alu_executor_v56 import (
    execute_alu_routed_control,
)
from visuaml.compiler_lab.neural_microop_executor_v57 import (
    execute_microop_routed_control,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    encode_control,
    execute_control,
)

from test_generated_wasm_alu_v56 import SAMPLES, generated


@pytest.mark.parametrize("seed", range(16))
def test_seeded_wasm_replay_with_microop_fetch_and_local_bank(seed):
    wasmtime = pytest.importorskip("wasmtime")
    program = generated(seed)
    source = encode_control(program)
    engine = wasmtime.Engine()
    wasmtime.Module.validate(engine, source)
    store = wasmtime.Store(engine)
    run = wasmtime.Instance(
        store, wasmtime.Module(engine, source), []
    ).exports(store)["run"]
    for argument in SAMPLES:
        candidate = execute_microop_routed_control(
            program, argument, fuel=500
        )
        historical = execute_alu_routed_control(
            program, argument, fuel=500
        )
        independent = execute_control(program, argument, fuel=500)
        assert candidate.result_i32 == historical.result_i32
        assert candidate.result_i32 == independent.value
        assert candidate.result_i32 == run(store, argument)
        for new, old, oracle in zip(
            candidate.trace, historical.trace, independent.trace, strict=True
        ):
            for key in (
                "pc", "next_pc", "stack_i32", "locals_i32",
                "control_labels", "halted",
            ):
                assert new[key] == old[key] == oracle[key]
