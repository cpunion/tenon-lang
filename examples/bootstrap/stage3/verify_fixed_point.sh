#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TENON="${ROOT}/target/release/tenon"
SELF="${ROOT}/examples/bootstrap/stage3/tenon_selfhost.tn"
RT="${ROOT}/examples/bootstrap/stage3/stage3_rt.c"

echo "=== [Stage 1 -> Stage 2] Rust tenon compiles tenon_selfhost.tn ==="
"$TENON" build --opt O2 -o /tmp/tenon_s2 "$SELF"

echo "=== [Stage 2 -> Stage 3] Pure Tenon compiles ITSELF (0% Rust!) ==="
TENON_IN="$SELF" TENON_OUT=/tmp/s3.ll /tmp/tenon_s2
clang -O2 -Wno-override-module -c /tmp/s3.ll -o /tmp/s3.o
clang -O2 /tmp/s3.o "$RT" -o /tmp/tenon_s3

echo "=== [Stage 3 -> Stage 4] Self-hosted Stage-3 compiles ITSELF (0% Rust!) ==="
TENON_IN="$SELF" TENON_OUT=/tmp/s4.ll /tmp/tenon_s3
clang -O2 -Wno-override-module -c /tmp/s4.ll -o /tmp/s4.o
clang -O2 /tmp/s4.o "$RT" -o /tmp/tenon_s4

echo "=== [Fixed-Point Verification: SHA-256(Stage-3) == SHA-256(Stage-4)] ==="
shasum -a 256 /tmp/s3.ll /tmp/s4.ll
cmp /tmp/s3.ll /tmp/s4.ll
echo "  -> LLVM IR (.ll) Fixed Point: BIT-EXACT IDENTICAL!"
shasum -a 256 /tmp/s3.o /tmp/s4.o
cmp /tmp/s3.o /tmp/s4.o
echo "  -> ARM64 Machine Code (.o) Fixed Point: BIT-EXACT IDENTICAL!"

echo "=== [Stage-4 End-to-End Target Compilation + L0 Div-Zero Rejection] ==="
/tmp/tenon_s4
cat > /tmp/s4_good.tn << 'EOF'
pub type Positive = Int where self >= 1

pub enum Mode {
    Fast
    Safe
    Turbo
}

pub record Point {
    x Int
    y Int
}

pub pure fn safe_ratio(total Int, count Positive) Int {
    return total / count
}

pub fn main() Int {
    let p = Point { x = 84, y = 42 }
    let m = Turbo
    let bonus = match m {
        Fast => 1
        Safe => 2
        Turbo => 3
        _ => 0
    }
    let r = safe_ratio(p.x + p.y, bonus)
    assert r == 42
    let _ = println("Stage-4 Native ARM64 Target (type+record+enum+match+assert): " ++ int_to_text(r))
    return 0
}
EOF
TENON_IN=/tmp/s4_good.tn TENON_OUT=/tmp/s4_good.ll /tmp/tenon_s4
clang -O2 -Wno-override-module /tmp/s4_good.ll "$RT" -o /tmp/s4_good_bin
/tmp/s4_good_bin

cat > /tmp/s4_bad.tn << 'EOF'
pub pure fn bad_div(a Int, b Int) Int { return a / b }
EOF
if TENON_IN=/tmp/s4_bad.tn TENON_OUT=/tmp/s4_bad.ll /tmp/tenon_s4; then
  echo "ERROR: expected Stage-4 L0 verifier to reject unproved division by zero!"
  exit 1
else
  echo "  -> Stage-4 L0 Verifier Rejected Unproved Division by Zero (E0401) OK!"
fi

rm -f /tmp/tenon_s2 /tmp/tenon_s3 /tmp/tenon_s4 /tmp/s3.ll /tmp/s4.ll /tmp/s3.o /tmp/s4.o /tmp/s4_good.tn /tmp/s4_good.ll /tmp/s4_good_bin /tmp/s4_bad.tn /tmp/s4_bad.ll
echo "=== STAGE-3 FIXED-POINT SELF-HOSTING BOOTSTRAP: 100% VERIFIED! ==="
