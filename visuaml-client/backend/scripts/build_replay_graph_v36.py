"""Render and byte-verify the approved v36 independent replay knowledge graph.

The source of truth is the archived, SHA-bound fresh pretrained CI result;
this script never downloads a model or executes inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visuaml_evidence.knowledge_graph_v36 import knowledge_graph


def render(repo: Path) -> bytes:
    value = knowledge_graph(repo)
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("visuaml-client/public/evidence/knowledge-v36.json"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.repo.resolve()
    target = args.output if args.output.is_absolute() else root / args.output
    expected = render(root)
    if args.check:
        if not target.is_file() or target.read_bytes() != expected:
            raise SystemExit("v36 graph does not equal reviewed source-bound evidence")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(expected)
    graph = knowledge_graph(root)
    print(json.dumps({
        "schema": graph["schema"],
        "knowledge_graph_sha256": graph["knowledge_graph_sha256"],
        "counts": graph["counts"],
        "verified_exact_bytes": args.check,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
