"""v58 local-only, no-Actions verification and benchmark orchestration.

Run a cheap smoke preflight first. Use --full for complete backend
regression; --benchmark for real timed measurements; --scaling for opt-in
capacity sweeps. No upload, GitHub Actions, background work or network API.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def invoke(argv: tuple[str, ...], *, client: Path, env: dict) -> None:
    print("Executing:", " ".join(argv), flush=True)
    subprocess.run(argv, cwd=client, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full", action="store_true",
                        help="Run entire backend Python regression")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run measured public/heldout benchmark")
    parser.add_argument("--scaling", action="store_true",
                        help="Include 26 scaling source programs")
    parser.add_argument("--faults", action="store_true",
                        help="Run 10 activated fault-injection tests")
    parser.add_argument("--roundtrip", action="store_true",
                        help="Execute blinded v50-to-v58 SMT-verified round trips")
    parser.add_argument("--representation-cost", action="store_true",
                        help="Measure original-vs-recovered WASM after proof replay")
    parser.add_argument("--roundtrip-holdout", action="store_true",
                        help="Include eight additional round-trip source programs")
    parser.add_argument("--memory", action="store_true",
                        help="Measure traced Python allocations separately")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    if args.scaling and not args.benchmark:
        parser.error("--scaling requires --benchmark")
    if args.roundtrip_holdout and not args.roundtrip:
        parser.error("--roundtrip-holdout requires --roundtrip")
    if args.representation_cost and not args.roundtrip:
        parser.error("--representation-cost requires --roundtrip")
    packages = ["numpy", "pytest", "wasmtime", "ruff"]
    if args.roundtrip:
        packages.append("z3")
    for package in packages:
        if importlib.util.find_spec(package) is None:
            parser.error(f"Missing local dependency: {package}")

    backend = Path(__file__).resolve().parents[1]
    client = backend.parent
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("Output directory must be absent or empty")
    # Output location must not be inside the repo: evidence generator's
    # clean-tree/source-bound gate must remain meaningful.
    root = client.parent.resolve()
    if output_dir == root or root in output_dir.parents:
        parser.error("Choose an output directory OUTSIDE the Git checkout")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((
        str(backend), env.get("PYTHONPATH", ""),
    ))
    # Evidence producers require a clean Git tree. Prevent pytest, Python
    # and Ruff from creating cache artifacts inside the source checkout.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["RUFF_NO_CACHE"] = "1"
    env["PYTEST_ADDOPTS"] = (
        env.get("PYTEST_ADDOPTS", "") + " -p no:cacheprovider"
    ).strip()

    lint = (
        "backend/visuaml/compiler_lab/neural_microstep_v58.py",
        "backend/visuaml/compiler_lab/neural_microstep_executor_v58.py",
        "backend/visuaml/compiler_lab/wasm_scale_corpus_v58.py",
        "backend/scripts/benchmark_neural_vm_v58.py",
        "backend/scripts/compare_neural_benchmarks_v58.py",
        "backend/scripts/summarize_neural_benchmarks_v58.py",
        "backend/scripts/benchmark_faults_v58.py",
        "backend/scripts/list_system_benchmarks_v58.py",
        "backend/scripts/verify_offline_v58.py",
        "backend/scripts/benchmark_neural_symbolic_roundtrip_v58.py",
        "backend/scripts/verify_neural_roundtrip_evidence_v58.py",
        "backend/scripts/summarize_neural_roundtrips_v58.py",
        "backend/scripts/benchmark_representation_cost_v58.py",
        "backend/tests/test_roundtrip_benchmarks_v58.py",
        "backend/visuaml/compiler_lab/neural_symbolic_roundtrip_v58.py",
        "backend/tests/test_neural_symbolic_roundtrip_v58.py",
        "backend/tests/test_neural_microstep_v58.py",
        "backend/tests/test_neural_benchmarks_v58.py",
        "backend/tests/test_neural_scaling_faults_v58.py",
    )
    invoke((sys.executable, "-m", "compileall", "-q",
            str(backend / "visuaml" / "compiler_lab"),
            str(backend / "scripts")),
           client=client, env=env)
    invoke((sys.executable, "-m", "ruff", "check", *lint),
           client=client, env=env)
    focused = [
        "test_neural_microstep_v58.py",
        "test_neural_benchmarks_v58.py",
        "test_neural_scaling_faults_v58.py",
    ]
    tests = (
        ["backend/tests"] if args.full
        else [f"backend/tests/{name}" for name in focused]
    )
    invoke((sys.executable, "-m", "pytest", "-q", *tests),
           client=client, env=env)
    # Avoid creating output until correctness preflight succeeds.
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = ("--scaling",) if args.scaling else ()
    memory = ("--memory",) if args.memory else ()
    run = "--benchmark" if args.benchmark else "--smoke"
    benchmark_file = output_dir / "neural-v58-benchmark.json"
    invoke((
        sys.executable, "backend/scripts/benchmark_neural_vm_v58.py",
        "--output", str(benchmark_file),
        "--repeats", str(args.repeats),
        "--warmups", str(args.warmups),
        *(() if args.benchmark else ("--smoke",)),
        *mode, *memory,
    ), client=client, env=env)
    invoke((
        sys.executable, "backend/scripts/summarize_neural_benchmarks_v58.py",
        "--input", str(benchmark_file),
        "--output-json", str(output_dir / "neural-v58-scorecard.json"),
        "--output-csv", str(output_dir / "neural-v58-scorecard.csv"),
    ), client=client, env=env)
    if args.roundtrip:
        invoke((
            sys.executable, "-m", "pytest", "-q",
            "backend/tests/test_neural_symbolic_roundtrip_v58.py",
        ), client=client, env=env)
        roundtrip_file = output_dir / "neural-v58-roundtrips.json"
        invoke((
            sys.executable,
            "backend/scripts/benchmark_neural_symbolic_roundtrip_v58.py",
            "--output", str(roundtrip_file),
            *(("--holdout",) if args.roundtrip_holdout else ()),
        ), client=client, env=env)
        invoke((
            sys.executable,
            "backend/scripts/verify_neural_roundtrip_evidence_v58.py",
            "--input", str(roundtrip_file),
        ), client=client, env=env)
        invoke((
            sys.executable,
            "backend/scripts/summarize_neural_roundtrips_v58.py",
            "--input", str(roundtrip_file),
            "--output", str(output_dir / "neural-v58-roundtrip-scorecard.json"),
        ), client=client, env=env)
        if args.representation_cost:
            invoke((
                sys.executable,
                "backend/scripts/benchmark_representation_cost_v58.py",
                "--input", str(roundtrip_file),
                "--output", str(output_dir / "neural-v58-representation-cost.json"),
                "--repeats", str(args.repeats),
                "--warmups", str(args.warmups),
            ), client=client, env=env)
    if args.faults:
        invoke((
            sys.executable, "backend/scripts/benchmark_faults_v58.py",
            "--output", str(output_dir / "neural-v58-faults.json"),
        ), client=client, env=env)
    print(f"Local v58 verification and {run} completed: {output_dir}")


if __name__ == "__main__":
    main()
