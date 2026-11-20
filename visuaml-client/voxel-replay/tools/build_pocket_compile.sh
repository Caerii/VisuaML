#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p evidence/pocket-compile
cartridge=${COMPILE_WASM:-guest/pocket_compile.wasm}
raw=${COMPILE_RAW:-evidence/pocket-compile/compiler.raw.wasm}
profile=()
if [[ "${POCKET_SCROLL:-1}" == 1 ]]; then profile=(--cfg pocket_scroll); fi
if [[ "${POCKET_RUST_CALL:-0}" == 1 ]]; then profile+=(--cfg pocket_rust_call); fi
rustc --cfg pocket_compiler "${profile[@]}" guest/pocket_compile.rs --crate-type=cdylib --target=wasm32-unknown-unknown \
    -C opt-level=z -C panic=abort -C link-arg=-zstack-size=0 \
    -C link-arg=--global-base=0 -C link-arg=--initial-memory=65536 \
    -C link-arg=--max-memory=65536 -C link-arg=--strip-all -o "$raw"
python tools/lower_cartridge.py "$raw" "$cartridge"
