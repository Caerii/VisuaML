"""Capture, open, and compare portable direct-call WASM replay bundles."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import struct
import sys

from visuaml.replay._codec import canonical
from visuaml.replay.shards import compare_shards
from visuaml.replay.journal_index import build_index, compare_indexes, verify_index
from visuaml.replay.pipeline import compare_observations, read_observations


EXAMPLE = Path(__file__).with_name("assets") / "passage-report.json"


def demo(output):
    """Export the shipped passage checkpoint from the compiled runner."""
    from visuaml.replay import PAGE_BYTES, check_report, make_bundle, open_bundle

    report = json.loads(EXAMPLE.read_text())
    bundle = report["bundle"]
    replay, checkpoint = open_bundle(bundle)
    initial = bytearray(PAGE_BYTES)
    struct.pack_into("<6i", initial, 65512, 7, 1, 0, 0, -1, 0)
    created = make_bundle(replay, 65512, initial, 2000, bundle["retired_source_steps"], runner="compiled")
    if created != bundle:
        raise ValueError("Compiled capture differs from the shipped reference checkpoint")
    traced = check_report(dict(report, bundle=created))
    final = replay.summary(replay.run(replay.restore(checkpoint), runner="compiled"))
    if final != {key: traced[key] for key in final}:
        raise ValueError("Compiled continuation differs from the reference")
    if output:
        output.write_bytes(canonical(dict(report, bundle=created))+b"\n")
    returned = next(row["stack"][-1] for row in traced["trace"] if row["depth"] == 0)
    print(f"Captured inside the Rust helper. It returned material {returned}.")
    print(f"Reference verified {len(traced['trace'])} remaining source steps. Compiled resume matched the final state.")
    if output:
        print(f"Report written to {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="visuaml-replay")
    try:
        release = version("visuaml-replay")
    except PackageNotFoundError:
        release = "source checkout"
    parser.add_argument("--version", action="version", version=f"%(prog)s {release}")
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture", help="Capture a guest at a source instruction boundary")
    capture.add_argument("wasm", type=Path)
    capture.add_argument("output", type=Path)
    capture.add_argument("--memory", type=Path)
    capture.add_argument("--argument", type=int, default=0)
    capture.add_argument("--fuel", type=int, default=5000)
    capture.add_argument("--after-step", type=int, required=True)
    capture.add_argument("--runner", choices=("reference", "compiled"), default="reference")
    demo_command = sub.add_parser("demo", help="Try the compiled-to-reference passage handoff")
    demo_command.add_argument("--output", type=Path, help="Save the runnable report")
    resume = sub.add_parser("resume", help="Resume a checkpoint and show the final state without a trace")
    resume.add_argument("bundle", type=Path)
    resume.add_argument("--runner", choices=("reference", "compiled"), default="reference")
    window = sub.add_parser("inspect", help="Reconstruct a short source trace from a report or checkpoint")
    window.add_argument("bundle", type=Path)
    window.add_argument("--after-step", type=int, help="Start after this absolute retired source step")
    window.add_argument("--steps", type=int, default=8, help="Maximum trace rows (1–256, default 8)")
    window.add_argument("--runner", choices=("reference", "compiled"), default="reference",
                        help="Runner used to reach the start of the trace")
    window.add_argument("--output", type=Path, help="Save the window for the browser viewer")
    opening = sub.add_parser("open", help="Resume a portable bundle")
    opening.add_argument("bundle", type=Path)
    opening.add_argument("--output", type=Path, help="Save executable result as JSON")
    packing = sub.add_parser("pack", help="Package a checkpoint and reproduction recipe")
    packing.add_argument("bundle", type=Path)
    packing.add_argument("output", type=Path)
    recipe_options = packing.add_mutually_exclusive_group()
    recipe_options.add_argument("--preview-local", type=int)
    recipe_options.add_argument("--expect-return", type=int, help="Record an expected i32 result")
    comparison = sub.add_parser("compare", help="Compare ordinary execution with a local.get preview candidate")
    comparison.add_argument("bundle", type=Path)
    comparison.add_argument("--preview-local", type=int, required=True)
    comparison.add_argument("--output", type=Path)
    cross = sub.add_parser("compare-onnx", help="Compare named ONNX Runtime outputs with a trusted PyTorch export")
    cross.add_argument("reference", type=Path)
    cross.add_argument("candidate", type=Path)
    cross.add_argument("bindings", type=Path, help="JSON with ordered logical output bindings and ONNX input names")
    cross.add_argument("inputs", nargs="+", type=Path)
    cross.add_argument("--rtol", type=float, default=0)
    cross.add_argument("--atol", type=float, default=0)
    pipeline = sub.add_parser("compare-pipelines", help="Find the first difference in two JSONL observation streams")
    pipeline.add_argument("reference", type=Path)
    pipeline.add_argument("candidate", type=Path)
    shards = sub.add_parser("compare-shards", help="Compare worker journals in logical request order")
    shards.add_argument("--reference", type=Path, nargs="+", required=True)
    shards.add_argument("--candidate", type=Path, nargs="+", required=True)
    indexing = sub.add_parser("index-shards", help="Build a reusable index from ordered JSONL shards")
    indexing.add_argument("shards", type=Path, nargs="+")
    indexing.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("verify-index", help="Check all stored blocks in a journal index")
    audit.add_argument("index", type=Path)
    indexed = sub.add_parser("compare-indexes", help="Compare indexed pipeline snapshots")
    indexed.add_argument("reference", type=Path)
    indexed.add_argument("candidate", type=Path)
    for command in (pipeline, shards, indexed):
        command.add_argument("--output", type=Path)
        command.add_argument("--fail-on-difference", action="store_true", help="Exit 1 when a difference is found")
        command.add_argument("--tolerances", type=Path, help="JSON mapping boundary names to [rtol, atol]")
    describe = sub.add_parser("describe", help="Inspect WASM, ONNX, or a trusted PyTorch export")
    describe.add_argument("artifact", type=Path)
    exports = sub.add_parser("compare-exports", help="Compare trusted PyTorch exports on identical inputs")
    exports.add_argument("reference", type=Path)
    exports.add_argument("candidate", type=Path)
    exports.add_argument("inputs", nargs="+", type=Path, help="One .npy file per positional tensor")
    exports.add_argument("--rtol", type=float, default=0)
    exports.add_argument("--atol", type=float, default=0)
    sub.add_parser("doctor", help="Check the installed reference runtime")
    example = sub.add_parser("example", help="Save the bundled Passage report")
    example.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-index":
            print(json.dumps(verify_index(args.index), indent=2))
            return
        if args.command == "index-shards":
            print(json.dumps(build_index(args.shards, args.output), indent=2))
            return
        if args.command in ("compare-pipelines", "compare-shards", "compare-indexes"):
            tolerances = json.loads(args.tolerances.read_text()) if args.tolerances else None
            if args.command == "compare-indexes":
                result = compare_indexes(args.reference, args.candidate, tolerances=tolerances)
            elif args.command == "compare-shards":
                result = compare_shards(args.reference, args.candidate, tolerances=tolerances)
            else:
                with args.reference.open(encoding="utf-8") as left, args.candidate.open(encoding="utf-8") as right:
                    result = compare_observations(read_observations(left), read_observations(right),
                                                  tolerances=tolerances)
            data = json.dumps(result, indent=2)
            if args.output:
                args.output.write_text(data+"\n")
            print(data)
            if args.fail_on_difference and result["first_difference"] is not None:
                raise SystemExit(1)
            return
        import numpy as np
        import torch

        from visuaml.replay import (
            REPORT_KIND, Replay, capture as capture_report, check_report, compare,
            execute_report, inspect_window, open_bundle, seal_report,
        )
        from visuaml.replay.artifacts import compare_exports, compare_onnx, describe_export, describe_onnx, describe_wasm, load_export
        torch.set_num_threads(1)
        if args.command == "describe":
            suffix = args.artifact.suffix
            if suffix == ".wasm":
                description = describe_wasm(args.artifact.read_bytes())
            elif suffix == ".onnx":
                description = describe_onnx(args.artifact.read_bytes())
            elif suffix == ".pt2":
                description = describe_export(load_export(args.artifact))
            else:
                raise ValueError("Expected .wasm, .onnx, or a trusted .pt2 export")
            print(json.dumps(description, indent=2))
            return
        if args.command == "compare-onnx":
            bindings = json.loads(args.bindings.read_text())
            inputs = [np.load(path, allow_pickle=False) for path in args.inputs]
            result = compare_onnx(load_export(args.reference), args.candidate, bindings, inputs,
                                  rtol=args.rtol, atol=args.atol)
            print(json.dumps(result, indent=2))
            return
        if args.command == "compare-exports":
            inputs = [torch.from_numpy(np.load(path, allow_pickle=False)) for path in args.inputs]
            result = compare_exports(load_export(args.reference), load_export(args.candidate),
                                     inputs, rtol=args.rtol, atol=args.atol)
            print(json.dumps(result, indent=2))
            return
        if args.command == "example":
            args.output.write_bytes(EXAMPLE.read_bytes())
            print(f"Example written to {args.output}")
            return
        if args.command == "doctor":
            result = check_report(json.loads(EXAMPLE.read_text()))
            print(f"Reference runtime ready. Bundled report verified ({len(result['trace'])} source steps).")
            return
        if args.command == "demo":
            demo(args.output)
            return
        if args.command == "capture":
            replay = Replay(args.wasm.read_bytes())
            initial = args.memory.read_bytes() if args.memory else None
            bundle = capture_report(replay, args.argument, initial, args.fuel, args.after_step, runner=args.runner)
            args.output.write_bytes(canonical(bundle)+b"\n")
            return
        bundle = json.loads(args.bundle.read_text())
        if not isinstance(bundle, dict):
            raise ValueError("Expected a JSON object")
        if args.command == "pack":
            if args.expect_return is not None:
                recipe = {"kind": "expect-return", "value": args.expect_return}
            elif args.preview_local is not None:
                recipe = {"kind": "local.get-preview", "local": args.preview_local}
            else:
                recipe = {"kind": "resume"}
            args.output.write_bytes(canonical(seal_report(bundle, recipe))+b"\n")
            return
        if args.command == "resume":
            document = bundle["bundle"] if bundle.get("report_kind") == REPORT_KIND else bundle
            replay, checkpoint = open_bundle(document)
            state = replay.run(replay.restore(checkpoint), runner=args.runner)
            print(json.dumps(replay.summary(state), indent=2))
            return
        if args.command == "inspect":
            result = inspect_window(bundle, after=args.after_step, limit=args.steps, runner=args.runner)
        elif args.command == "open":
            if bundle.get("report_kind") == REPORT_KIND:
                result = check_report(bundle)
            else:
                result = execute_report(bundle, {"kind": "resume"})
        else:
            result = compare(bundle, args.preview_local)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2))
    except (ValueError, KeyError, OSError) as error:
        print(f"visuaml-replay: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
