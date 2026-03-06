#!/usr/bin/env bash
# build-fix-context.sh — Build a shared context file for fix-queue agents
#
# Produces: .claude/context/fix-context.md
# Contains: project structure, key patterns, type signatures, fix log,
# and enough detail that agents can start coding immediately.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$PROJECT_DIR/.claude/context/fix-context.md"

mkdir -p "$(dirname "$OUT")"

cat > "$OUT" <<'HEADER'
# Fix Agent Context

You are fixing a specific issue in neuroscript-rs. DO NOT explore the codebase broadly.
Read only the files mentioned in your task. Start coding immediately after verifying the issue exists.

## Project: NeuroScript Compiler (Rust → PyTorch)

Architecture: .ns source → pest grammar → AST → IR → validator → shape inference → optimizer → codegen → Python

HEADER

# Key file map with line counts
{
  echo "## Key Files (line counts)"
  echo '```'
  for f in src/lib.rs src/main.rs src/interfaces.rs src/ir.rs src/desugar.rs \
           src/grammar/mod.rs src/grammar/ast.rs src/grammar/error.rs \
           src/validator/core.rs src/validator/cycles.rs src/validator/shapes.rs \
           src/validator/symbol_table.rs src/validator/bindings.rs \
           src/contract_resolver.rs src/unroll.rs \
           src/shape/algebra.rs src/shape/inference.rs \
           src/codegen/generator.rs src/codegen/forward.rs \
           src/codegen/instantiation.rs src/codegen/utils.rs \
           src/optimizer/core.rs \
           src/stdlib.rs src/stdlib_registry.rs src/wasm.rs \
           src/package/registry.rs src/package/security.rs; do
    if [[ -f "$PROJECT_DIR/$f" ]]; then
      lines=$(wc -l < "$PROJECT_DIR/$f" | tr -d ' ')
      printf "  %-45s %s lines\n" "$f" "$lines"
    fi
  done
  echo '```'
  echo ""
} >> "$OUT"

# Key type signatures from interfaces.rs (just the enum/struct names)
{
  echo "## Core IR Types (src/interfaces.rs)"
  echo '```rust'
  grep -E '^pub (enum|struct) ' "$PROJECT_DIR/src/interfaces.rs" | head -25
  echo '```'
  echo ""
} >> "$OUT"

# Error types
{
  echo "## Error Types"
  echo '```rust'
  grep -E '^\s*(pub )?(enum |struct ).*(Error|Warning)' "$PROJECT_DIR/src/interfaces.rs" 2>/dev/null | head -10
  echo '```'
  echo ""
} >> "$OUT"

# Validation pipeline from lib.rs
{
  echo "## Validation Pipeline (src/lib.rs)"
  echo '```rust'
  sed -n '/^pub fn validate/,/^}/p' "$PROJECT_DIR/src/lib.rs" 2>/dev/null
  echo '```'
  echo ""
} >> "$OUT"

# Build commands
{
  echo "## Build & Test Commands"
  echo '```bash'
  echo 'cargo check           # Fast syntax check'
  echo 'cargo build --release # Full build'
  echo 'cargo test            # All tests'
  echo 'cargo test <module>   # e.g., cargo test shape_inference'
  echo 'cargo test <name>     # Single test by name'
  echo './target/release/neuroscript validate <file.ns>'
  echo './target/release/neuroscript compile <file.ns>'
  echo '```'
  echo ""
} >> "$OUT"

# Comments syntax reminder
{
  echo "## NeuroScript Syntax Notes"
  echo "- Comments: \`#\` inline, \`///\` doc comments. NOT \`//\`"
  echo "- Impl refs: \`core,nn/Linear\` (comma separator, not dot)"
  echo "- Run \`source ~/.venv_ai/bin/activate\` before Python tests"
  echo ""
} >> "$OUT"

# Fix log (what's already been done)
{
  echo "## Prior Fixes"
  if [[ -f "$PROJECT_DIR/docs/reviews/fix-log.md" ]]; then
    tail -30 "$PROJECT_DIR/docs/reviews/fix-log.md"
  else
    echo "(none yet)"
  fi
  echo ""
} >> "$OUT"

# Recent git activity
{
  echo "## Recent Commits"
  echo '```'
  cd "$PROJECT_DIR"
  git log --oneline -10 2>/dev/null
  echo '```'
} >> "$OUT"

echo "Generated: $OUT ($(wc -l < "$OUT" | tr -d ' ') lines)" >&2
