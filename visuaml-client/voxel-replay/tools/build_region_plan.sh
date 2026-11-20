#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rustc guest/plan_local_pair.rs --crate-type=cdylib --target=wasm32-unknown-unknown \
    -C opt-level=z -C panic=abort -C link-arg=-zstack-size=0 \
    -C link-arg=--global-base=0 -C link-arg=--initial-memory=65536 \
    -C link-arg=--max-memory=65536 -C link-arg=--strip-all \
    -o evidence/guest-region-plan/planner.wasm
