"""Inspect and validate VisuaML's entire offline benchmark registry.

Shows which suites have executable harnesses versus future integration
contracts, without fabricating progress percentages or measured values.
No GitHub Actions or network access required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SCHEMA = "visuaml.system-benchmark-registry.v58"
ALLOWED = {"implemented_unmeasured", "planned", "blocked"}
REQUIRED = {
    "id", "title", "status", "runner", "units", "dimensions",
    "baselines", "source_binding", "admission",
}


def read_registry(root: Path) -> dict:
    path = root / "benchmarks/VISUAML_SYSTEM_BENCHMARK_REGISTRY_V58.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema") != SCHEMA:
        raise ValueError("Unrecognized system benchmark registry")
    if registry.get("measured_results") is not None:
        raise ValueError("Registry is a declaration, not a results archive")
    suites = registry.get("suites")
    if not isinstance(suites, list) or not suites:
        raise ValueError("Missing system benchmark suites")
    seen = set()
    for suite in suites:
        if not REQUIRED.issubset(suite):
            raise ValueError("Incomplete benchmark declaration")
        if suite["id"] in seen:
            raise ValueError(f"Duplicate suite ID: {suite['id']}")
        seen.add(suite["id"])
        if suite["status"] not in ALLOWED:
            raise ValueError(f"Invalid suite state: {suite['id']}")
        if (not suite["units"] or not suite["baselines"]
                or not suite["dimensions"] or not suite["admission"]):
            raise ValueError(f"Insufficient measurement contract: {suite['id']}")
        runner = suite["runner"]
        if runner is None:
            if suite["status"] == "implemented_unmeasured":
                raise ValueError(f"Runnable suite has no runner: {suite['id']}")
        else:
            if (not isinstance(runner, str)
                    or not runner.startswith("visuaml-client/backend/scripts/")
                    or not (root / runner).is_file()):
                raise ValueError(f"Invalid local benchmark runner: {suite['id']}")
    if not registry.get("policy", {}).get("github_actions_disabled"):
        raise ValueError("Unexpected GitHub Actions benchmark policy")
    return registry


def summary(registry: dict) -> dict:
    suites = registry["suites"]
    return {
        "schema": SCHEMA,
        "suite_count": len(suites),
        "implemented_unmeasured": sum(
            s["status"] == "implemented_unmeasured" for s in suites
        ),
        "planned": sum(s["status"] == "planned" for s in suites),
        "blocked": sum(s["status"] == "blocked" for s in suites),
        "measured_result_count": 0,
        "suites": [
            {
                "id": s["id"], "status": s["status"],
                "runner": s["runner"], "dimensions": s["dimensions"],
            }
            for s in suites
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = summary(read_registry(root))
    if args.json:
        print(json.dumps(output, sort_keys=True, indent=2))
        return
    print(
        f"VisuaML benchmark registry: {output['suite_count']} suites "
        f"({output['implemented_unmeasured']} implemented/unmeasured, "
        f"{output['planned']} planned, {output['blocked']} blocked). "
        "No measured performance numbers are available."
    )
    for entry in output["suites"]:
        print(f"  {entry['status']:24s} {entry['id']}")


if __name__ == "__main__":
    main()
