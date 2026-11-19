#!/usr/bin/env bash
set -euo pipefail
rustc="${RUSTC:-rustc}"
"$rustc" --edition=2021 --target wasm32-unknown-unknown --crate-type cdylib \
  -C opt-level=z -C panic=abort -C target-cpu=mvp -C link-arg=--strip-all \
  -C link-arg=--stack-first -C link-arg=-z -C link-arg=stack-size=0 \
  -C link-arg=--initial-memory=65536 -C link-arg=--max-memory=65536 \
  "$1" -o "$2"
