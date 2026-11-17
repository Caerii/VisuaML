"""Generate or verify the deterministic public v34 research evidence catalog.

This reads only reviewed, repository-pinned observations. It does not ingest
untrusted proposals or execute model Python/native source. No source data is
changed by --check. Commits, timestamps, and CI status are not guessed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visuaml_evidence.library_v34 import EvidenceLibrary


def render(repo: Path) -> bytes:
    catalog = EvidenceLibrary.from_repo(repo).catalog()
    return (json.dumps(
        catalog, indent=2, sort_keys=True, allow_nan=False
    ) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("visuaml-client/public/evidence/library-v34.json"),
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Fail if checked-in catalog differs from reviewed evidence",
    )
    args = parser.parse_args()
    root = args.repo.resolve()
    output = args.output
    if not output.is_absolute():
        output = root / output
    expected = render(root)
    if args.check:
        if not output.is_file() or output.read_bytes() != expected:
            raise SystemExit(
                "Evidence catalog does not exactly match reviewed source artifacts"
            )
        print(f"Evidence catalog verified: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(expected)
        print(f"Evidence catalog generated: {output}")
    snapshot = json.loads(expected)
    print(json.dumps({
        "schema": snapshot["schema"],
        "catalog_sha256": snapshot["catalog_sha256"],
        "counts": snapshot["counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
