"""Capture, open, and compare portable direct-call WASM replay bundles."""
import argparse
import json
from pathlib import Path
import struct
import sys
from unittest.mock import patch

import torch

from visuaml.replay import (
    PAGE_BYTES, REPORT_KIND, Replay, capture as capture_report, check_report, compare,
    execute_report, make_bundle, open_bundle, seal_report,
)
from visuaml.replay.checkpoint import canonical

PACKED = "visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep"


def demo(output):
    """Export the shipped passage checkpoint from the compiled runner."""
    example = Path(__file__).resolve().parents[1] / "release/examples/suspended-passage-report.json"
    report = json.loads(example.read_text())
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


def main():
    parser = argparse.ArgumentParser(prog="visuaml_replay")
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
    opening = sub.add_parser("open", help="Resume a portable bundle")
    opening.add_argument("bundle", type=Path)
    opening.add_argument("--output", type=Path, help="Save executable result as JSON")
    packing = sub.add_parser("pack", help="Package a checkpoint and reproduction recipe")
    packing.add_argument("bundle", type=Path)
    packing.add_argument("output", type=Path)
    packing.add_argument("--preview-local", type=int)
    comparison = sub.add_parser("compare", help="Compare ordinary execution with a local.get preview candidate")
    comparison.add_argument("bundle", type=Path)
    comparison.add_argument("--preview-local", type=int, required=True)
    comparison.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
            if args.command == "demo":
                demo(args.output)
                return
            if args.command == "capture":
                replay = Replay(args.wasm.read_bytes())
                initial = args.memory.read_bytes() if args.memory else bytes(PAGE_BYTES)
                bundle = capture_report(replay, args.argument, initial, args.fuel, args.after_step, runner=args.runner)
                args.output.write_bytes(canonical(bundle)+b"\n")
                return
            bundle = json.loads(args.bundle.read_text())
            if not isinstance(bundle, dict):
                raise ValueError("Expected a JSON object")
            if args.command == "pack":
                recipe = ({"kind": "resume"} if args.preview_local is None else
                          {"kind": "local.get-preview", "local": args.preview_local})
                args.output.write_bytes(canonical(seal_report(bundle, recipe))+b"\n")
                return
            if args.command == "resume":
                document = bundle["bundle"] if bundle.get("report_kind") == REPORT_KIND else bundle
                replay, checkpoint = open_bundle(document)
                state = replay.run(replay.restore(checkpoint), runner=args.runner)
                print(json.dumps(replay.summary(state), indent=2))
                return
            if args.command == "open":
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
        print(f"visuaml_replay: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
