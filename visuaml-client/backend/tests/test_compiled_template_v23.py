"""One verified compiled graph can serve independently owned KV sessions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from test_hf_llama_checkpoint_v19 import fixture
from visuaml.compiler_lab.compiled_template_v23 import (
    CompiledLlamaTemplate,
    CompiledTemplateError,
)
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.llama_semantic_v20 import (
    lower_llama_semantic_program,
    make_llama_semantic_program,
)


def setup(tmp_path, *, layers=2, batch=1):
    fixture(tmp_path, layers=layers)
    weights = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(weights, batch=batch),
        weights, batch=batch,
    )
    return weights, plan


@pytest.mark.parametrize("layers", (1, 2, 4))
def test_compile_once_many_sessions_with_independent_kv(tmp_path, layers):
    weights, plan = setup(tmp_path, layers=layers)
    template = CompiledLlamaTemplate(plan)
    first = template.new_session()
    second = template.new_session()
    third = template.new_session()
    signature = (1, -1, -1)
    assert first._compiled[signature] is second._compiled[signature]
    assert second._compiled[signature] is third._compiled[signature]
    assert template.report()["compiled_graphs"] == 1
    assert second.report()["traces"] == []
    assert third.report()["traces"] == []
    first.step(np.array([[1, 2, 3]], np.int64))
    assert first.cursor == 3 and second.cursor == 0
    second.step(np.array([[5, 4]], np.int64))
    assert second.cursor == 2 and third.cursor == 0
    first_got = first.step(np.array([[7]], np.int64))
    second_got = second.step(np.array([[6, 7]], np.int64))
    np.testing.assert_allclose(
        first_got, weights.forward(np.array([[1, 2, 3, 7]], np.int64))[:, -1:],
        atol=5e-5, rtol=5e-5,
    )
    np.testing.assert_allclose(
        second_got, weights.forward(np.array([[5, 4, 6, 7]], np.int64))[:, -2:],
        atol=5e-5, rtol=5e-5,
    )
    assert first.report()["specializations"] == 1
    assert second.report()["specializations"] == 1
    assert template.report()["compiled_graphs"] == 1
    third.step(np.array([[2]], np.int64))
    first.reset()
    assert first.cursor == 0 and third.cursor == 1
    assert template.report()["semantic_sha256"] == plan.semantic_sha256


def test_compiled_template_isolation_from_untrusted_caller_mutation(tmp_path):
    checkpoint, plan = setup(tmp_path)
    template = CompiledLlamaTemplate(plan)
    left = template.new_session()
    original = left.step(np.array([[1, 2]], np.int64))
    # Caller retains their original arrays, but plans are immutable snapshots.
    checkpoint.arrays["model.norm.weight"] = np.zeros(
        (checkpoint.profile.hidden,), np.float32
    )
    other = template.new_session()
    np.testing.assert_allclose(
        original, other.step(np.array([[1, 2]], np.int64)), atol=0, rtol=0
    )
    assert left.cursor == 2 and other.cursor == 2
    assert template.report()["compiled_graphs"] == 1


def test_bad_compilation_plan_rejected_before_graph_creation(tmp_path):
    _, plan = setup(tmp_path)
    with pytest.raises(CompiledTemplateError):
        CompiledLlamaTemplate("untrusted_module")
    with pytest.raises(ValueError):
        CompiledLlamaTemplate(replace(plan, batch=2))


def test_template_concurrent_session_creation_uses_one_verified_trace(tmp_path):
    checkpoint, plan = setup(tmp_path, layers=1)
    template = CompiledLlamaTemplate(plan)

    def worker(token: int):
        session = template.new_session()
        tokens = np.array([[token, 2]], dtype=np.int64)
        got = session.step(tokens)
        np.testing.assert_allclose(
            got, checkpoint.forward(tokens), atol=5e-5, rtol=5e-5
        )
        return session.report()["specializations"], session.cursor

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(worker, (1, 3, 5)))
    assert all(count == 1 and cursor == 2 for count, cursor in results)
    assert template.report()["compiled_graphs"] == 1
