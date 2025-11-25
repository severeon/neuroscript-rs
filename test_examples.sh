#!/bin/bash
cd "$(dirname "$0")"

echo "Testing all example files..."
echo

passed=0
failed=0

# run neuroscript on all ns files in examples/

find examples/ -name "*.ns" | while read file; do
  if [ ! -f "$file" ]; then
    continue
  fi

  # Run the parser and capture both stdout and stderr
  output=$(./target/release/neuroscript "$file")
  exitcode=$?

  # Check if it contains "Parsed" (success) rather than "Parse error"
  if echo "$output" | grep -q "^Parsed"; then
    echo "✓ $file"
    ((passed++))
  else
    echo "✗ $file"
    # Show the error
    echo "$output" | grep "Parse error"
    ((failed++))
  fi
done

echo
echo "Passed: $passed"
echo "Failed: $failed"
