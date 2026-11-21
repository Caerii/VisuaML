"""Run a WASM test and save a runnable report on a return-value mismatch."""
import argparse
from pathlib import Path

import torch
from wasmtime import wat2wasm

from visuaml.replay import PAGE_BYTES, Replay, make_bundle, seal_report
from visuaml.replay.checkpoint import canonical

WAT = """(module
    (memory (export "memory") 1 1)
    (func (param i32) (result i32) local.get 0 i32.const 2 i32.add)
    (func (export "run") (param i32) (result i32) local.get 0 call 0))"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=int, default=43)
    parser.add_argument("--report", type=Path, default=Path("failure.json"))
    args = parser.parse_args()
    if not -2**31 <= args.expected < 2**32:
        parser.error("--expected must be an i32 value")
    torch.set_num_threads(1)
    replay = Replay(bytes(wat2wasm(WAT)))
    page = bytes(PAGE_BYTES)
    state = replay.start(40, page, 100)
    final = replay.summary(replay.run(state))
    if final["status"] == 1 and final["returned"] == args.expected & 0xffffffff:
        print("WASM test passed.")
        return 0
    bundle = make_bundle(replay, 40, page, 100, 2)
    report = seal_report(bundle, {"kind": "expect-return", "value": args.expected})
    args.report.write_bytes(canonical(report)+b"\n")
    print(f"Expected {args.expected}; actual {final['returned']}. Report saved to {args.report}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
