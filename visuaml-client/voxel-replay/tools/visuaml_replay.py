"""Capture, open, and compare portable direct-call WASM replay bundles."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

from visuaml.replay import (
    PAGE_BYTES, REPORT_KIND, Replay, capture as capture_report, check_report, compare,
    execute_report, make_bundle, seal_report,
)
from visuaml.replay.checkpoint import canonical

PACKED = "visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep"


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
            if args.command == "capture":
                replay = Replay(args.wasm.read_bytes())
                initial = args.memory.read_bytes() if args.memory else bytes(PAGE_BYTES)
                bundle = capture_report(replay, args.argument, initial, args.fuel, args.after_step)
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
