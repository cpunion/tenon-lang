#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DIR="$(cd "$(dirname "$0")" && pwd)"
TENON="${ROOT}/target/release/tenon"
if [ ! -x "$TENON" ]; then
  echo "Building release tenon compiler..."
  (cd "$ROOT" && cargo build --release --bin tenon)
fi
echo "Compiling OpenUI + VDOM kernel to Wasm32 + ESM auto-bridge..."
"$TENON" build --target wasm32 \
  --emit-mjs "$DIR/openui.mjs" \
  -o "$DIR/openui.wasm" \
  "$ROOT/examples/dogfood/openui_wasm_kernel.tn"

PORT="${1:-8788}"
echo ""
echo "================================================================"
echo "  Tenon OpenUI + Wasm32 VDOM Studio is live at:"
echo "  -> http://127.0.0.1:${PORT}/"
echo "  Press Ctrl+C to stop."
echo "================================================================"
echo ""
(sleep 0.6 && open "http://127.0.0.1:${PORT}/" 2>/dev/null || true) &
cd "$DIR" && python3 -m http.server "$PORT" --bind 127.0.0.1
