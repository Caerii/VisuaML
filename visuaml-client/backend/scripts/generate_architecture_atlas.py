"""Generate the conservative, offline model-family coverage snapshot.

Usage: python backend/scripts/generate_architecture_atlas.py
No external models, weights, network requests, or arbitrary source execution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from visuaml.compiler_lab.architecture_catalog import catalog_report
from visuaml.compiler_lab.numpy_runtime import OPS


def generate() -> dict:
    return catalog_report(available_ops=OPS)


def main() -> None:
    root = BACKEND.parent
    target = root / "public" / "architecture_atlas.v1.json"
    report = generate()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Updated {target} ({len(report['families'])} families) SHA={report['sha256']}"
    )


if __name__ == "__main__":
    main()
