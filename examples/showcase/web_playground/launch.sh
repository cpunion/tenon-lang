#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DIR="$(cd "$(dirname "$0")" && pwd)"
TENON="${ROOT}/target/release/tenon"
if [ ! -x "$TENON" ]; then
  (cd "$ROOT" && cargo build --release --bin tenon)
fi
mkdir -p "$DIR/dist"
"$TENON" build --target wasm32 --emit-mjs "$DIR/dist/openui.mjs" -o "$DIR/dist/openui.wasm" "$ROOT/examples/dogfood/openui_wasm_kernel.tn"
echo "Tenon Web Playground ready -> http://127.0.0.1:8791/"
cd "$DIR" && python3 -m http.server 8791 --bind 127.0.0.1
