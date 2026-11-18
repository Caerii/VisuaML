"""Offline v57 verification and source-bound evidence handoff.

Deliberately does not use the GitHub API or GitHub Actions.
Recommended: python -m pip install wasmtime pytest numpy
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path,
        help="New evidence JSON filename (will never overwrite)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Include complete backend test regression, beyond v49-v57",
    )
    args = parser.parse_args()
    for package in ("pytest", "numpy", "wasmtime"):
        if importlib.util.find_spec(package) is None:
            raise SystemExit(f"Missing local dependency: {package}")
    backend = Path(__file__).resolve().parents[1]
    client = backend.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((
        str(backend),
        env.get("PYTHONPATH", ""),
    ))
    focused = (
        "test_neural_alu_v56.py",
        "test_neural_alu_executor_v56.py",
        "test_generated_wasm_alu_v56.py",
        "test_typed_microop_ir_v57.py",
        "test_neural_microop_executor_v57.py",
        "test_generated_wasm_microop_v57.py",
    )
    paths = (
        ["backend/tests"] if args.full
        else [f"backend/tests/{name}" for name in focused]
    )
    subprocess.run(
        (sys.executable, "-m", "pytest", "-q", *paths),
        cwd=client, env=env, check=True,
    )
    subprocess.run(
        (
            sys.executable, "backend/scripts/probe_typed_microop_v57.py",
            "--output", str(args.output.resolve()),
        ),
        cwd=client, env=env, check=True,
    )


if __name__ == "__main__":
    main()
