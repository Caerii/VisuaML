"""Regenerate v14 bounded multi-layer AOT evidence from locally installed sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visuaml.compiler_lab.stacked_certification_v14 import certify_stacked_decoder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = certify_stacked_decoder()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: {len(report['cases'])} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
