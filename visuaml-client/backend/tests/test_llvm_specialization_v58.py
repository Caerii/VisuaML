from importlib.util import find_spec

import numpy as np
import pytest

from visuaml.compiler_lab.bit_neural_vm_v49 import BitVMAdmissionError
from visuaml.compiler_lab.llvm_specialization_v58 import LlvmSpecialization
from visuaml.compiler_lab.neural_microstep_executor_v58 import MicrostepAdmissionError
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted, encode_control
from visuaml.compiler_lab.wasm_research_corpus_v57 import gcd12, generated

pytestmark = pytest.mark.skipif(
    find_spec("llvmlite") is None,
    reason="optional LLVM backend unavailable",
)


@pytest.mark.parametrize("level", (0, 1, 2))
def test_llvm_batch_matches_original_wasm_and_keeps_engines_independent(level):
    wasmtime = pytest.importorskip("wasmtime")
    inputs = np.array([-(1 << 31), -1, 0, 1, (1 << 31) - 1] * 5, dtype=np.int32)
    programs = [encode_control(generated(seed)) for seed in range(4)]
    candidates = [LlvmSpecialization(source, optimization_level=level) for source in programs]
    engine = wasmtime.Engine()
    for source, candidate in zip(programs, candidates, strict=True):
        store = wasmtime.Store(engine)
        direct = wasmtime.Instance(store, wasmtime.Module(engine, source), []).exports(store)["run"]
        expected = [direct(store, int(value)) for value in inputs]
        output = inputs.copy()
        candidate.run_batch_inplace(output)
        assert output.tolist() == expected
        assert [candidate.run(int(value)) for value in inputs] == expected
        candidate.run_batch_inplace(output[:0])
        with pytest.raises(ControlFuelExhausted):
            candidate.run_batch_inplace(output, fuel=1)
        with pytest.raises(BitVMAdmissionError):
            candidate.run_batch_inplace(output[::2])
        assert output.tolist() == expected

    with pytest.raises(MicrostepAdmissionError):
        LlvmSpecialization(encode_control(gcd12()))



def test_llvm_rejects_unsupported_optimization_levels():
    source = encode_control(generated(0))
    for invalid in (False, "0", -1, 3):
        with pytest.raises(ValueError, match="optimization level"):
            LlvmSpecialization(source, optimization_level=invalid)
