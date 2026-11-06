#!/usr/bin/env python3
"""Read-only, allowlisted VisuaML Compiler Lab API for the interactive frontend.

Deliberately does NOT load submitted Python files, arbitrary model paths, packages,
or artifact archives. The returned partition sweep is a hypothetical cost-model
exercise, not an executable multi-device backend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from examples.compiler_lab_demo import run

SCENARIOS = ("mlp", "attention", "assembly", "convolution")
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def analyze(scenario: str, seed: int = 19) -> dict:
    if scenario not in SCENARIOS:
        raise ValueError("Scenario must be one of: " + ", ".join(SCENARIOS))
    if type(seed) is not int or not 0 <= seed <= 100_000:
        raise ValueError("seed must be an integer in [0, 100000]")
    # run() has checks for actual execution of both targets and per-pass witnesses.
    with TemporaryDirectory(prefix="visuaml-lab-") as directory:
        report = run(scenario, Path(directory), seed)
    # Only report serializable measurements; generated bundles don't leave the temp dir.
    report["api_contract"] = {
        "schema": "visuaml.compiler-lab.api.v1",
        "source": "fixed-local-demo-fixtures",
        "result_kind": "measured-cpu-witness-and-hypothetical-device-costs",
        "backend_execution": "numpy-aot + torch export CPU",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    try:
        report = analyze(args.scenario, args.seed)
        payload = json.dumps(report, separators=(",", ":"), allow_nan=False)
        if len(payload.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise ValueError("Compiler analysis response exceeds limit")
        print(payload)
        return 0
    except Exception as exc:
        print(
            f"Compiler Lab analysis failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
