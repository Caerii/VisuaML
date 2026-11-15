"""Machine-readable certification must never imply whole-family support."""
import json
import hashlib
from dataclasses import replace
import pytest
from visuaml.compiler_lab.family_certification_v12 import FAMILIES, certification_report, run_witness

@pytest.mark.parametrize("family", [item[0] for item in FAMILIES])
def test_each_family_differential_witness(family):
    rec=run_witness(family,19)
    assert rec.passed and rec.checked_arrays>=1
    assert rec.max_abs_error < 5e-5
    assert len(rec.input_sha256)==len(rec.reference_sha256)==64


def test_deterministic_certificates_and_claim_ladder():
    a=certification_report(seeds=(7,19))
    b=certification_report(seeds=(7,19))
    assert a==b
    assert len(a["certifications"])==6
    assert "BLOCK parity" in a["claim"]
    assert a["source_kind"]=="synthetic-cpu-fixtures"
    assert all(x["seed_count"]==2 and not x["family_checkpoint_certified"]
               and not x["full_model_compiler_executable"] and not x["physical_hardware_validated"]
               for x in a["certifications"])
    digest=a.pop("sha256")
    assert hashlib.sha256(json.dumps(a,sort_keys=True,separators=(",", ":"),allow_nan=False).encode()).hexdigest()==digest

@pytest.mark.parametrize("seed", [-1,True,100001])
def test_rejects_invalid_seed(seed):
    with pytest.raises(ValueError):run_witness("decoder_transformer",seed)


def test_rejects_unknown_family_and_duplicate_seeds():
    with pytest.raises(ValueError):run_witness("/tmp/evil",19)
    with pytest.raises(ValueError):certification_report(seeds=(19,19))


def test_checked_in_snapshot_digest_and_claims():
    from pathlib import Path
    path=Path(__file__).resolve().parents[2]/"public"/"family_block_certificates.v12.json"
    payload=json.loads(path.read_text(encoding="utf-8"))
    checksum=payload.pop("sha256")
    encoded=json.dumps(payload,sort_keys=True,separators=(",", ":"),allow_nan=False).encode()
    assert hashlib.sha256(encoded).hexdigest()==checksum
    assert {c["architecture_id"] for c in payload["certifications"]}=={name for name,_ in FAMILIES}
    assert all(c["level"]=="cross-framework-reference-parity"
               and not c["family_checkpoint_certified"]
               and not c["full_model_compiler_executable"]
               for c in payload["certifications"])
