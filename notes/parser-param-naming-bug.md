# Parser Bug: Parameter Names Incorrectly Rejected

> **Note**: This document describes a historical bug that existed in the old handwritten lexer/parser.
> The issue was completely resolved when the codebase migrated to a Pest PEG grammar on January 6, 2026.
> All test cases described below now parse correctly.

## Status

✅ **RESOLVED** - Fixed by migration to Pest grammar on January 6, 2026 (commit 0f51cb2)

**Resolution Date**: January 6, 2026
**Fix**: Complete rewrite of parser using Pest PEG grammar
**Verified**: January 28, 2026 - All test cases now pass successfully

## Summary

The parser incorrectly rejects valid parameter names in neuron definitions when they contain common patterns like `dim`, or start with keyword prefixes like `in_`. This forces developers to use awkward workarounds and significantly degrades developer experience.

## Problem Statement

Parameter names in neuron definitions should be completely independent from:

1. Shape dimension variables (e.g., `dim`, `batch`, `seq`)
2. Port keywords (e.g., `in:`, `out:`)
3. Language keywords (e.g., `from`, `impl`)

However, the parser currently **conflates these namespaces**, treating parameter identifiers as if they were dimension variables or keywords.

## Reproduction

### Test Case 1: Parameters with "dim"

```neuroscript
# SHOULD WORK but currently FAILS
neuron Linear(in_dim, out_dim):
  in: [*, in_dim]
  out: [*, out_dim]
  impl: core,nn/Linear
```

**Error:**

```
Expected COMMENT or endpoint, found unexpected input
```

**Attempted variations that also FAIL:**

- `dim_in`, `dim_out` - fails
- `i_dim`, `o_dim` - fails
- `indim`, `outdim` - fails

### Test Case 2: Parameters starting with "in"

```neuroscript
# SHOULD WORK but currently FAILS
neuron Embedding(in_features, out_features):
  in: [*, in_features]
  out: [*, out_features]
  impl: core,nn/Embedding
```

**Error:**

```
Expected COMMENT or endpoint, found unexpected input
```

**Attempted variations that also FAIL:**

- `input_size`, `output_size` - fails
- `in_channels`, `out_channels` - fails

### Test Case 3: Context contamination

```neuroscript
# Match expression earlier in file
neuron Processor:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> match:
      [*, d] where d > 512: Linear(d, 512) -> out
      [*, d]: Identity() -> out

# Later neuron definition FAILS even with simple params
neuron Linear(a, b):  # FAILS if 'a' or 'b' resemble bound variables
  in: [*, a]
  out: [*, b]
  impl: core,nn/Linear
```

### Test Case 4: Works (but awkward)

```neuroscript
# Only simple, non-conflicting names work
neuron Linear(n, m):
  in: [*, n]
  out: [*, m]
  impl: core,nn/Linear
```

## Expected Behavior

**All of these should parse successfully:**

```neuroscript
neuron Linear(in_dim, out_dim):
  in: [*, in_dim]
  out: [*, out_dim]
  impl: core,nn/Linear

neuron Conv2d(in_channels, out_channels, kernel_size):
  in: [*, in_channels, h, w]
  out: [*, out_channels, h_out, w_out]
  impl: core,nn/Conv2d

neuron Projection(dim_in, dim_out):
  in: [batch, seq, dim_in]
  out: [batch, seq, dim_out]
  impl: core,nn/Linear

neuron Embedding(vocab_size, embedding_dim):
  in: [batch, seq]
  out: [batch, seq, embedding_dim]
  impl: core,nn/Embedding
```

**Parameters are NOT:**

- Shape dimension variables (those appear in `[...]` brackets)
- Port names (those appear as `in:`, `out:`, `in left:`)
- Keywords (those are `neuron`, `graph`, `impl`, `match`, etc.)

**Parameters ARE:**

- Neuron configuration values passed at instantiation time
- Used to compute shapes (e.g., `[*, in_dim]`)
- Completely separate namespace from dimension variables

## Root Cause Analysis

### Likely Location: `src/parser/core.rs`

The parser appears to be:

1. **Not properly scoping parameter identifiers** - Parameters defined in neuron headers (`neuron Foo(param1, param2)`) should be in a separate namespace from dimension variables bound in match expressions.

2. **Keyword/identifier confusion** - The lexer/parser may be treating parameter names that contain keywords as special tokens rather than identifiers.

3. **Context leakage** - Dimension variables bound in match expressions (e.g., `[*, d] where d > 512`) appear to pollute the parsing context for subsequent neuron definitions.

### Specific Areas to Investigate

```rust
// src/parser/core.rs

// 1. Parameter parsing in neuron headers
fn parse_neuron_params(&mut self) -> Result<Vec<String>> {
    // Are parameters being added to wrong scope?
    // Are they being validated against keywords incorrectly?
}

// 2. Shape expression parsing
fn parse_shape(&mut self) -> Result<Shape> {
    // Are dimension variables being tracked separately from parameters?
}

// 3. Match expression parsing
fn parse_match(&mut self) -> Result<MatchExpr> {
    // Are bound dimensions (e.g., 'd' in '[*, d]') leaking out of scope?
}

// 4. Identifier/keyword discrimination
fn parse_identifier(&mut self) -> Result<String> {
    // Is this checking against keywords too aggressively?
}
```

### Likely Fix

The parser needs to maintain **separate symbol tables** for:

1. **Neuron parameters** - e.g., `in_dim`, `out_dim`, `num_heads`
2. **Dimension variables** - e.g., `batch`, `seq`, `dim` (in shape expressions)
3. **Port references** - e.g., `in`, `out`, `fork.left`
4. **Match-bound variables** - e.g., `d`, `s` (locally scoped to match arms)

These should be **non-overlapping namespaces** with proper scoping rules.

## Impact

### User Experience

- ❌ Forces awkward parameter names (`n`, `m` instead of `in_dim`, `out_dim`)
- ❌ Confusing error messages with no clear guidance
- ❌ Inconsistent with ML/PyTorch conventions (`in_features`, `out_features`)
- ❌ Breaks code that looks perfectly valid
- ❌ Makes the language feel fragile and unpredictable

### Developer Productivity

- Wasted hours trying different parameter names
- Requires workarounds and "tribal knowledge"
- Discourages contribution to stdlib/examples
- Makes onboarding difficult

### Language Design

- Violates principle of least surprise
- Creates artificial constraints with no semantic justification
- Undermines trust in the parser

## Test Cases for Fix

Once fixed, all these should parse and validate:

```neuroscript
# Test 1: Standard ML naming conventions
neuron Linear(in_features, out_features):
  in: [*, in_features]
  out: [*, out_features]
  impl: core,nn/Linear

# Test 2: Dimension suffix pattern
neuron MLP(in_dim, hidden_dim, out_dim):
  in: [*, in_dim]
  out: [*, out_dim]
  graph:
    in -> Linear(in_dim, hidden_dim) -> GELU() -> Linear(hidden_dim, out_dim) -> out

# Test 3: Dimension prefix pattern
neuron Projection(dim_in, dim_out):
  in: [batch, seq, dim_in]
  out: [batch, seq, dim_out]
  impl: core,nn/Linear

# Test 4: Channel naming (CNNs)
neuron Conv2d(in_channels, out_channels, kernel_size):
  in: [*, in_channels, h, w]
  out: [*, out_channels, h_out, w_out]
  impl: core,nn/Conv2d

# Test 5: Size/dimension mixed
neuron Embedding(vocab_size, embedding_dim, max_len):
  in: [batch, seq]
  out: [batch, seq, embedding_dim]
  impl: core,nn/Embedding

# Test 6: After match expression (context isolation)
neuron Processor:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> match:
      [*, d] where d > 512: Linear(d, 512) -> out
      [*, d]: Identity() -> out

# This should work - 'in_dim' param is separate from 'd' match var
neuron Linear(in_dim, out_dim):
  in: [*, in_dim]
  out: [*, out_dim]
  impl: core,nn/Linear
```

## Priority

**P0 - Critical**

This bug affects:

- All user code with standard parameter naming
- Standard library development
- Example quality and clarity
- First impressions of the language
- PyTorch/ML convention alignment

## Success Criteria

✅ All parameter names in test cases parse successfully
✅ No regressions in existing valid code
✅ Clear error messages for actual syntax errors
✅ Parameter namespace is properly isolated
✅ Match-bound variables don't leak scope
✅ Documentation updated with parameter naming guidelines

## Related Files

- `src/parser/core.rs` - Main parser implementation
- `src/lexer/core.rs` - Tokenization and keyword handling
- `src/interfaces.rs` - IR definitions (NeuronDef, Parameter types)
- `tests/integration_tests.rs` - Add regression tests
- `examples/primitives/*.ns` - Currently use workarounds
- `examples/tutorials/*.ns` - Currently use workarounds

## Notes

This bug was discovered during the examples directory reorganization (Jan 28, 2026) when creating clean, documented example files. The workaround of using simple names like `n`, `m` works but significantly degrades code quality and readability.

The fact that `neuron Linear(in_dim, out_dim)` works in some contexts (old examples) but fails in others (after match expressions) suggests **context leakage** rather than a fundamental design constraint.

## Investigation Trail

Attempted parameter names (all rejected):

1. `in_dim`, `out_dim`
2. `i_dim`, `o_dim`
3. `dim_in`, `dim_out`
4. `indim`, `outdim`
5. `in_features`, `out_features` (failed: starts with `in`)
6. `input_size`, `output_size` (failed: starts with `in`)
7. `from_size`, `to_size` (failed: `from` might be keyword)
8. `n`, `m` ✅ (works but poor naming)

This systematic exploration reveals the parser is treating identifiers with certain substrings as special tokens.

---

## Verification (January 28, 2026)

All parameter naming patterns now work correctly with the Pest grammar:

```bash
# Test results with current parser (neuroscript v2 with Pest)
$ ./target/release/neuroscript parse test_all_param_variations.ns
✓ Successfully parsed test_all_param_variations.ns
  Test1 (in_dim, out_dim) ✅
  Test2 (dim_in, dim_out) ✅
  Test3 (i_dim, o_dim) ✅
  Test4 (indim, outdim) ✅
  Test5 (in_features, out_features) ✅
  Test6 (input_size, output_size) ✅
  Test7 (in_channels, out_channels) ✅
  Test8 (vocab_size, embedding_dim) ✅
  Test9 (from_size, to_size) ✅
```

## How the Pest Grammar Fixed This

The Pest PEG grammar (`src/grammar/neuroscript.pest`) correctly handles parameter naming through:

1. **Atomic keyword rules with boundary checking**:
   ```pest
   keyword_in = @{ "in" ~ !ident_cont }
   keyword_out = @{ "out" ~ !ident_cont }
   ```
   The `!ident_cont` ensures that "in" only matches as a keyword when NOT followed by an identifier continuation character.

2. **Proper identifier rule**:
   ```pest
   ident = @{
       !keyword ~ (ASCII_ALPHA | "_") ~ ident_cont*
   }
   ```
   The negative lookahead `!keyword` checks if the ENTIRE identifier is a keyword, not just a prefix.

3. **Separate namespaces**: Parameters, dimension variables, and keywords are properly scoped in the AST builder (`src/grammar/ast.rs`).

## Migration Details

- **Commit**: `0f51cb2` - "Migrate to Pest grammar exclusively and remove legacy lexer/parser"
- **Date**: January 6, 2026
- **Changes**: Complete removal of handwritten lexer (`src/lexer/`) and parser (`src/parser/`)
- **Replacement**: Pest PEG grammar with proper keyword boundary checking
