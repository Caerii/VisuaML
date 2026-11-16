"""The global research roadmap is an audited snapshot, never guessed progress."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "visuaml-client" / "public" / "roadmap.json"


@pytest.fixture(scope="module")
def manifest():
    doc = json.loads(PATH.read_text(encoding="utf-8"))
    assert doc["schema"] == "visuaml.public-research-roadmap.v1"
    return doc


def test_published_acceptance_weights_and_progress_are_reproducible(manifest):
    streams = manifest["workstreams"]
    assert len(streams) == 8
    assert sum(stream["weight"] for stream in streams) == 100
    progress = {}
    for stream in streams:
        assert sum(gate["weight"] for gate in stream["criteria"]) == 100
        assert len({gate["id"] for gate in stream["criteria"]}) == len(stream["criteria"])
        assert all(
            gate["status"] in {"verified", "in-progress", "planned"}
            for gate in stream["criteria"]
        )
        for gate in stream["criteria"]:
            if gate["status"] == "verified":
                assert gate["evidence"], (stream["id"], gate["id"])
            else:
                assert gate["weight"] > 0
        progress[stream["id"]] = sum(
            gate["weight"] for gate in stream["criteria"]
            if gate["status"] == "verified"
        )
    assert progress["backends"] < 50  # Do not mistake TorchScript for independent AOT.
    assert progress["hardware"] < 50  # Simulated placement is not physical execution.
    score = sum(
        stream["weight"] * progress[stream["id"]] for stream in streams
    ) / 100
    assert score == 66.5


def test_evidence_urls_are_local_to_the_public_repository(manifest):
    prefix = "https://github.com/Caerii/VisuaML/"
    checked = 0
    for stream in manifest["workstreams"]:
        for gate in stream["criteria"]:
            for evidence in gate["evidence"]:
                url = evidence["url"]
                assert url.startswith(prefix), url
                parsed = urlparse(url)
                assert parsed.netloc == "github.com"
                path = unquote(parsed.path)
                if "/blob/main/" in path:
                    rel = path.split("/blob/main/", 1)[1]
                    target = (ROOT / rel).resolve()
                    assert target.is_relative_to(ROOT)
                    assert target.is_file(), rel
                    checked += 1
                elif "/tree/main/" in path:
                    rel = path.split("/tree/main/", 1)[1]
                    target = (ROOT / rel).resolve()
                    assert target.is_relative_to(ROOT)
                    assert target.is_dir(), rel
                    checked += 1
                else:
                    assert re.search(r"/(actions|commit|pull|tree)/", path)
    assert checked >= 15


def test_snapshot_contains_no_fake_deadlines_or_claimed_live_telemetry(manifest):
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", manifest["asOf"])
    assert re.fullmatch(r"[a-f0-9]{40}", manifest["referenceCommit"])
    methodology = manifest["methodology"]
    assert "not a live monitor" in methodology["refresh"].lower()
    assert "not research certainty" in methodology["caveat"].lower()
    assert all(stream["next"] and stream["impact"] for stream in manifest["workstreams"])
