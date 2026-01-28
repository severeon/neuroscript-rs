# Parameter Naming Bug - RESOLVED ✅

**Status**: Fixed in Pest grammar migration (January 6, 2026)
**Verification**: January 28, 2026
**Test File**: `tests/param_naming_regression.ns`

## Summary

The parameter naming bug that prevented common ML naming conventions (like `in_features`, `out_features`, `in_dim`, `out_dim`) has been **completely resolved** by the migration to the Pest PEG grammar.

## What Was Fixed

The old handwritten lexer/parser incorrectly rejected parameter names that:
- Started with keywords (`in_features`, `out_channels`)
- Contained keyword substrings (`dim_in`, `dim_out`)
- Appeared after match expressions with bound dimension variables

## How It Was Fixed

**Commit**: `0f51cb2` - "Migrate to Pest grammar exclusively and remove legacy lexer/parser"
**Date**: January 6, 2026

The Pest grammar uses proper keyword boundary checking:
```pest
keyword_in = @{ "in" ~ !ident_cont }
ident = @{ !keyword ~ (ASCII_ALPHA | "_") ~ ident_cont* }
```

This ensures:
- "in" only matches as a keyword when NOT followed by `_`, letters, or digits
- "in_dim" is correctly recognized as an identifier, not a keyword
- Parameters are in a separate namespace from dimension variables

## Verification Results

All test cases from the original bug report now pass:

```bash
✓ in_dim, out_dim
✓ dim_in, dim_out
✓ in_features, out_features
✓ input_size, output_size
✓ in_channels, out_channels
✓ vocab_size, embedding_dim
✓ from_size, to_size
✓ Context isolation (match expressions don't affect later neurons)
```

## Regression Test

A comprehensive regression test has been added at:
```
tests/param_naming_regression.ns
```

This test covers all previously problematic parameter naming patterns and can be run with:
```bash
./target/release/neuroscript validate tests/param_naming_regression.ns
```

## For Future Reference

If you encounter parameter naming issues in the future:
1. Check the Pest grammar rules in `src/grammar/neuroscript.pest`
2. Verify keyword definitions include `!ident_cont` boundary checking
3. Run the regression test to ensure no regressions: `./target/release/neuroscript validate tests/param_naming_regression.ns`

## Related Files

- **Bug Report**: `notes/parser-param-naming-bug.md` (historical)
- **Regression Test**: `tests/param_naming_regression.ns` (active)
- **Grammar**: `src/grammar/neuroscript.pest` (implementation)
- **AST Builder**: `src/grammar/ast.rs` (parameter handling)
