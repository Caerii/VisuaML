"""Architecture atlas is an evidence-backed plan, not a model-support claim."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict

import pytest

from visuaml.compiler_lab.architecture_catalog import (
    ARCHITECTURES,
    DOMAINS,
    FEATURES,
    Architecture,
    capability_gap,
    catalog_report,
    get_architecture,
    rank_missing_capabilities,
)
from visuaml.compiler_lab.architecture_probes import fixture_names, probe_fixture
from visuaml.compiler_lab.numpy_runtime import OPS


def test_catalog_contains_major_and_exotic_families_and_no_duplicate_ids():
    ids = [x.id for x in ARCHITECTURES]
    assert len(ids) >= 50 and len(set(ids)) == len(ids)
    assert {entry.domain for entry in ARCHITECTURES} == DOMAINS
    for name in ("decoder_transformer", "moe", "mamba", "rwkv", "xlstm", "dit",
                 "graph_conv", "snn_lif", "kan", "fno", "gaussian_splatting",
                 "assembly_calculus", "diffusion_language", "audio_codec", "vlm"):
        assert get_architecture(name).id == name
    assert all(set(x.features) <= FEATURES for x in ARCHITECTURES)


def test_catalog_digest_deterministic_independent_of_input_order():
    first = catalog_report(available_ops=OPS)
    second = catalog_report(available_ops=list(reversed(sorted(OPS))))
    assert first == second
    content = {key: value for key, value in first.items() if key != "sha256"}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert first["sha256"] == digest
    assert len(first["families"]) == len(ARCHITECTURES)
    assert "not" in first["claim"]


def test_catalog_infers_only_primitive_candidates_not_model_support():
    mlp = get_architecture("mlp")
    covered = capability_gap(mlp, available_ops=mlp.operators)
    assert covered.status == "candidate-for-validation"
    assert covered.coverage_fraction == 1
    missing = capability_gap(get_architecture("moe"), available_ops=OPS)
    assert missing.status == "blocked-by-contract"
    assert "expert_dispatch" in missing.missing_features
    assert missing.coverage_fraction > 0
    assert missing.coverage_fraction < 1


def test_coverage_does_not_upgrade_when_only_features_are_supplied():
    sparse = get_architecture("moe")
    result = capability_gap(sparse, available_ops=(), available_features=sparse.features)
    assert len(result.missing_features) == 0
    assert result.coverage_fraction == 0
    assert result.status == "blocked-by-contract"


def test_ranked_gaps_are_reproducible_and_do_not_claim_performance():
    gaps = rank_missing_capabilities(available_ops=OPS, limit=20)
    assert gaps == rank_missing_capabilities(available_ops=set(OPS), limit=20)
    assert gaps and gaps[0]["weighted_incidence"] >= gaps[-1]["weighted_incidence"]
    assert all("not observed" in g["metric"] for g in gaps)
    assert any(g["capability"].startswith("feature:") for g in gaps)


def test_bad_catalog_entries_and_feature_names_fail_closed():
    entry = get_architecture("mlp")
    with pytest.raises(ValueError, match="Unknown/duplicate feature"):
        Architecture(entry.id, entry.name, entry.domain, 1, entry.operators,
                     ("invented_semantic_magic",), entry.exemplars, entry.note)
    with pytest.raises(ValueError, match="Unknown declared"):
        catalog_report(available_ops=OPS, available_features={"arbitrary"})
    with pytest.raises(ValueError, match="Invalid planning limit"):
        rank_missing_capabilities(available_ops=(), limit=101)


@pytest.mark.parametrize("name", ["mlp", "residual-conv", "attention-core", "graph-scatter",
                                   "recurrent-scan", "lif-gate", "unet-upsampling",
                                   "spline-sine", "moe-topk"])
def test_aot_microprobe_executes_and_matches_reference(name):
    result = probe_fixture(name)
    assert result.status == "tiny-aot-witness-passed", asdict(result)
    assert result.capture_passed and result.aot_passed and result.reference_executed
    assert result.max_abs_error is not None and result.max_abs_error < 1e-4
    assert result.unsupported_aot_operations == ()
    assert "Only this synthetic" in result.note


def test_unsupported_gru_cell_stays_a_visible_operator_gap():
    result = probe_fixture("gru-cell")
    assert result.status == "aot-rejected"
    assert result.primitive_operations == ("module:GRUCell",)
    assert not result.aot_passed
    assert result.error_kind == "UnsupportedOperation"


def test_allowlist_rejects_untrusted_fixture_selection_and_bad_seed():
    with pytest.raises(ValueError, match="Unknown fixture"):
        probe_fixture("../../tmp/injected.py")
    with pytest.raises(ValueError, match="seed"):
        probe_fixture("mlp", seed=True)
    assert len(fixture_names()) >= 10


def test_probing_does_not_modify_global_random_stream():
    import torch
    torch.manual_seed(987)
    before = torch.random.get_rng_state().clone()
    probe_fixture("mlp")
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)


def test_topk_tie_is_not_silently_claimed_portable():
    import torch
    from torch import nn
    from visuaml.compiler_spike import capture
    from visuaml.compiler_lab.aot_compile import compile_numpy_aot
    from visuaml.compiler_lab.numpy_runtime import BundleError

    class Router(nn.Module):
        def forward(self, x):
            return torch.topk(x, 2, dim=-1).indices

    captured = capture(Router().eval(), torch.tensor([[2.0, 2.0, 1.0]], dtype=torch.float32))
    artifact = compile_numpy_aot(captured)
    with pytest.raises(BundleError, match="ties"):
        artifact.run(__import__('numpy').array([[2, 2, 1]], dtype='float32'))


def test_model_family_registry_does_not_contain_mutable_cross_run_state():
    before = [asdict(a) for a in ARCHITECTURES]
    cloned = copy.deepcopy(before)
    cloned[0]["name"] = "Changed"
    assert [asdict(a) for a in ARCHITECTURES] == before


def test_checked_in_snapshot_matches_runtime_operator_contract():
    from pathlib import Path

    public = Path(__file__).resolve().parents[2] / 'public' / 'architecture_atlas.v1.json'
    assert public.is_file()
    snapshot = json.loads(public.read_text(encoding='utf-8'))
    assert snapshot == json.loads(json.dumps(catalog_report(available_ops=OPS)))
