#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for pages in 1 16 256; do
    size=$((pages * 65536))
    rustc guest/sparse_passage.rs --crate-type=cdylib --target=wasm32-unknown-unknown \
        -C opt-level=z -C panic=abort -C link-arg=-zstack-size=0 \
        -C link-arg=--global-base=0 -C link-arg=--initial-memory="$size" \
        -C link-arg=--max-memory="$size" -C link-arg=--strip-all \
        -o "evidence/bounded-recurrent-store/passage-${pages}.wasm"
done
rustc guest/passage_with_agent.rs --crate-type=cdylib --target=wasm32-unknown-unknown \
    -C opt-level=z -C panic=abort -C link-arg=-zstack-size=0 \
    -C link-arg=--global-base=0 -C link-arg=--initial-memory=16777216 \
    -C link-arg=--max-memory=16777216 -C link-arg=--strip-all \
    -o evidence/bounded-recurrent-store/passage-with-agent.wasm
