#!/usr/bin/env bash
# Validate all .ns example and stdlib files
# Usage: ./test_examples.sh
set -euo pipefail

BINARY="./target/release/neuroscript"
PASS=0
FAIL=0
ERRORS=""

if [ ! -f "$BINARY" ]; then
  echo "Building release binary..."
  cargo build --release
fi

echo "=== Parsing example files ==="
for f in examples/*.ns; do
  if "$BINARY" parse "$f" >/dev/null 2>&1; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
    ERRORS="$ERRORS\n  FAIL: $f"
    echo "  FAIL: $f"
  fi
done

echo "=== Parsing stdlib files ==="
for f in stdlib/*.ns; do
  if "$BINARY" parse "$f" >/dev/null 2>&1; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
    ERRORS="$ERRORS\n  FAIL: $f"
    echo "  FAIL: $f"
  fi
done

echo ""
echo "Results: $PASS passed, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
  echo -e "\nFailed files:$ERRORS"
  exit 1
fi
