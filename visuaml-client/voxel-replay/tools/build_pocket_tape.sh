#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out=evidence/pocket-tape
mkdir -p "$out"
rustc guest/pocket_tape.rs --crate-type=cdylib --target=wasm32-unknown-unknown \
    -C opt-level=z -C panic=abort -C link-arg=-zstack-size=0 -C link-arg=--global-base=0 \
    -C link-arg=--initial-memory=65536 -C link-arg=--max-memory=65536 -C link-arg=--strip-all \
    -o "$out/tape.raw.wasm"
python tools/lower_cartridge.py "$out/tape.raw.wasm" guest/pocket_tape.wasm
