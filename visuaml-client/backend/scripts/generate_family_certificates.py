"""Regenerate versioned synthetic family-block evidence on a trusted CPU runner.

The file records reference parity, never whole-family or physical support.
Usage: PYTHONPATH=backend python backend/scripts/generate_family_certificates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from visuaml.compiler_lab.family_certification_v12 import certification_report


def main() -> None:
    target = BACKEND.parent / "public" / "family_block_certificates.v12.json"
    report = certification_report()
    target.parent.mkdir(exist_ok=True, parents=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {target}; {len(report['certifications'])} reference blocks; SHA={report['sha256']}")


if __name__ == "__main__":
    main()
