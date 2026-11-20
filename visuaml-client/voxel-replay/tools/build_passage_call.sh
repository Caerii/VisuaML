#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rustc guest/passage_call.rs --crate-type=cdylib --target=wasm32-unknown-unknown \
    -C opt-level=z -C panic=abort -C link-arg=-zstack-size=0 \
    -C link-arg=--global-base=0 -C link-arg=--initial-memory=16777216 \
    -C link-arg=--max-memory=16777216 -C link-arg=--strip-all \
    -o evidence/frozen-calls/passage.wasm
rustc guest/terrain_tick.rs --crate-type=cdylib --target=wasm32-unknown-unknown \
    -C opt-level=z -C panic=abort -C link-arg=-zstack-size=0 \
    -C link-arg=--global-base=0 -C link-arg=--initial-memory=16777216 \
    -C link-arg=--max-memory=16777216 -C link-arg=--strip-all \
    -o evidence/frozen-calls/terrain-256.wasm
