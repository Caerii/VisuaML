"""Regenerate fixed synthetic decoder compiler evidence; no weights downloaded."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from visuaml.compiler_lab.decoder_certification_v13 import certify_decoder_compiler


def main() -> None:
    target = BACKEND.parent / "public" / "decoder_compiler_certificate.v13.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    result = certify_decoder_compiler()
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Decoder AOT certificate: {result['sha256']} ({len(result['cases'])} seeds)")


if __name__ == "__main__":
    main()
