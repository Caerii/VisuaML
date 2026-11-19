from base64 import b64decode
from copy import deepcopy
from json import loads
from pathlib import Path

import pytest

import terrain_verify as terrain

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.parametrize(("fault", "tick"), [("input", 1), ("state", 2)])
def test_first_divergent_tick(monkeypatch, fault, tick):
    proof = loads((FIXTURES / "terrain.proof.json").read_text())
    session = deepcopy(proof["session"])
    if fault == "input":
        session["inputs"][0]["edit"] = False
    else:
        original = terrain.execute_microop_routed_control
        calls = 0

        def corrupt_second_tick(*args, array_memory, **kwargs):
            nonlocal calls
            calls += 1
            result = original(*args, array_memory=array_memory, **kwargs)
            if calls == 2:
                array_memory[-1] ^= 1
            return result

        monkeypatch.setattr(terrain, "execute_microop_routed_control", corrupt_second_tick)
    with pytest.raises(terrain.Divergence) as caught:
        terrain.verify(session, proof)
    report = caught.value.report
    assert report["tick"] == tick
    assert report["input"] == session["inputs"][tick - 1]
    assert len(b64decode(report["page_before_base64"])) == terrain.PAGE_BYTES
    if fault == "state":
        assert "engine state" in report["reasons"]
        assert b64decode(report["numpy_page_base64"])[-1] == 1
        assert b64decode(report["wasmtime_page_base64"])[-1] == 0
