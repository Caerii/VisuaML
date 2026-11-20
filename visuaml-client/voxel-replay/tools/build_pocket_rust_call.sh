#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out=evidence/pocket-rust-call
mkdir -p "$out"
POCKET_RUST_CALL=1 COMPILE_RAW="$out/monitor.raw.wasm" COMPILE_WASM=guest/pocket_rust_call.wasm \
    bash tools/build_pocket_compile.sh
rustc guest/passage_helper.rs --crate-type=cdylib --target=wasm32-unknown-unknown \
    -C opt-level=z -C panic=abort -C link-arg=-zstack-size=0 -C link-arg=--global-base=0 \
    -C link-arg=--initial-memory=65536 -C link-arg=--max-memory=65536 -C link-arg=--strip-all \
    -o "$out/passable.raw.wasm"
python tools/lower_cartridge.py "$out/passable.raw.wasm" "$out/passable.wasm"
