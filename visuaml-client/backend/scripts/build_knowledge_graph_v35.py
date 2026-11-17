"""Build, verify or preflight the source-bound v35 computational graph.

No network access, model Python execution, native compilation or inference
occurs. --preflight is a REQUIREMENT REPORT, not a numerical replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visuaml_evidence.knowledge_graph_v35 import (
    knowledge_graph,
    preflight_replay,
)


def render(root: Path) -> bytes:
    return (json.dumps(
        knowledge_graph(root), indent=2, sort_keys=True, allow_nan=False
    ) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("visuaml-client/public/evidence/knowledge-v35.json"),
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.check and args.preflight:
        parser.error("--check and --preflight are separate operations")
    if args.check and args.checkpoint:
        parser.error("checkpoint must be supplied with --preflight")
    graph = knowledge_graph(repo)
    if args.preflight:
        result = preflight_replay(
            graph["plans"][0], repo, checkpoint=args.checkpoint,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        # Diagnostic blockers are not process failures: the status is in JSON.
        return
    destination = (
        args.output if args.output.is_absolute() else repo / args.output
    )
    expected = render(repo)
    if args.check:
        if not destination.is_file() or destination.read_bytes() != expected:
            raise SystemExit("v35 knowledge graph differs from reviewed source bytes")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(expected)
    print(json.dumps({
        "schema": graph["schema"],
        "digest": graph["knowledge_graph_sha256"],
        "counts": graph["counts"],
        "path": str(destination),
        "checked": args.check,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
