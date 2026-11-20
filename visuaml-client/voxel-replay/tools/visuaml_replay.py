"""Capture, open, and compare portable direct-call WASM replay bundles."""
import argparse
import base64
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

from fork_divergence import first_divergence
from frozen_call_checkpoint import canonical, digest, encode, identities, restore
from replay_frozen_recurrent import PACKED, wasmtime_run
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image

GRAPH = Path(__file__).resolve().parents[1] / "evidence/frozen-calls/machine.pt"
PROFILE = "visuaml-direct-call-i32-v1"
PAGE_BYTES = 65536


class Replay:
    def __init__(self, raw):
        self.raw = raw
        self.image = load_call_image(raw)
        self.graph = torch.jit.load(str(GRAPH))
        self.identity = identities(GRAPH.read_bytes(), self.image)

    def step(self, state, preview=None):
        pc = int(state.vm.pc)
        function = max(i for i, start in enumerate(self.image.starts) if start <= pc)
        source = pc-self.image.starts[function]
        opcode = self.image.functions[function][source]
        pulse = self.graph(self.image.rows, state)
        vm = pulse.state.vm
        display = pulse.state
        if preview is not None:
            if self.image.functions[function][source:source+2] != ("local.get", "local.get"):
                raise ValueError("Preview compare requires a local.get pair at the checkpoint")
            if not 0 <= preview < state.vm.locals.numel() or int(vm.status) != 0:
                raise ValueError("Preview local or source boundary is invalid")
            stack = vm.stack.clone()
            stack[int(vm.height)-1] = vm.locals[preview]
            display = display._replace(vm=vm._replace(stack=stack))
        row = dict(pc=pc, function=function, source_index=source, opcode=opcode,
                   next_pc=int(vm.pc), fuel=int(vm.fuel), status=int(vm.status),
                   depth=int(display.depth), stack=display.vm.stack[:int(vm.height)].tolist(),
                   access=pulse.access.tolist(), state_sha256=encode(display, self.identity)["checkpoint_sha256"])
        return pulse.state, row

    def finish(self, checkpoint, preview=None):
        state = restore(checkpoint, self.identity, self.image)
        rows = []
        while int(state.vm.status) == 0:
            state, row = self.step(state, preview if not rows else None)
            rows.append(row)
        return dict(trace=rows, status=int(state.vm.status), fuel=int(state.vm.fuel),
                    returned=int(state.vm.stack[0]) if int(state.vm.status) == 1 else None,
                    page_sha256=digest(state.vm.page.numpy().tobytes()))


def make_bundle(replay, argument, initial, fuel, count):
    if not 0 <= count < fuel:
        raise ValueError("Checkpoint step must be within the initial fuel allowance")
    state = initialize(replay.image, argument, initial, fuel)
    for _ in range(count):
        if int(state.vm.status) != 0:
            raise ValueError("Guest stopped before requested checkpoint")
        state, _ = replay.step(state)
    if int(state.vm.status) != 0:
        raise ValueError("Requested checkpoint is terminal")
    bundle = dict(format_version=1, profile=PROFILE, wasm=base64.b64encode(replay.raw).decode(),
                  retired_source_steps=count, checkpoint=encode(state, replay.identity))
    return dict(bundle, bundle_sha256=digest(canonical(bundle)))


def open_bundle(bundle):
    if bundle.get("format_version") != 1 or bundle.get("profile") != PROFILE:
        raise ValueError("Unsupported replay bundle")
    if type(bundle.get("retired_source_steps")) is not int or bundle["retired_source_steps"] < 0:
        raise ValueError("Invalid retired source step count")
    payload = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if digest(canonical(payload)) != bundle.get("bundle_sha256"):
        raise ValueError("Bundle integrity mismatch")
    raw = base64.b64decode(bundle["wasm"], validate=True)
    return Replay(raw), bundle["checkpoint"]


def compare(bundle, preview_local):
    replay, checkpoint = open_bundle(bundle)
    baseline = replay.finish(checkpoint)
    candidate = replay.finish(checkpoint, preview_local)
    boundary = first_divergence(baseline["trace"], candidate["trace"])
    first = None if boundary is None else dict(
        retired_source_steps=bundle["retired_source_steps"]+boundary+1,
        reference=baseline["trace"][boundary], candidate=candidate["trace"][boundary])
    return dict(format_version=1, profile=PROFILE, checkpoint_sha256=checkpoint["checkpoint_sha256"],
                preview_local=preview_local, first_difference=first, reference=baseline, candidate=candidate)


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
                bundle = make_bundle(replay, args.argument, initial, args.fuel, args.after_step)
                completed = replay.finish(bundle["checkpoint"])
                if completed["status"] == 1:
                    result, page = wasmtime_run(replay.raw, args.argument, initial)
                    assert result & 0xffffffff == completed["returned"] and digest(page) == completed["page_sha256"]
                args.output.write_bytes(canonical(bundle)+b"\n")
                return
            bundle = json.loads(args.bundle.read_text())
            if args.command == "open":
                replay, checkpoint = open_bundle(bundle)
                result = replay.finish(checkpoint)
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
