# NeuroScript pest Grammar - Review Document

## Summary

I've successfully created a complete pest PEG grammar for NeuroScript that parses all tested example files. The grammar is located in `src/grammar/neuroscript.pest` and is ~250 lines.

## Test Results

✅ **All 22 grammar tests passing:**
- Basic tokens (integers, identifiers, shapes)
- Use statements
- Simple and complex neuron definitions
- All numbered examples (01-10)
- Critical examples: residual.ns, 22-xor.ns, 27-addition.ns
- Multiple connection styles
- Match expressions
- Let/set bindings

## Grammar Features

### Successfully Implemented

1. **Lexical Elements**
   - All keywords (neuron, use, in, out, impl, graph, match, where, let, set, etc.)
   - Operators (arrows, arithmetic, comparison, assignment)
   - Literals (integers, floats, strings with backticks, booleans)
   - Identifiers with keyword exclusion

2. **Shapes and Dimensions**
   - Literals: `[512, 256]`
   - Named dimensions: `[batch, seq, dim]`
   - Wildcards: `[*, 512]`
   - Variadic: `[*batch, seq, dim]`
   - Expressions: `[dim * 4, seq - 1]`

3. **Port Specifications** (Flexible - supports all current styles)
   - Inline single: `in: [shape]`
   - Inline named: `in left: [shape]`
   - Indented multi-port:
     ```
     in:
       left: [shape]
       right: [shape]
     ```

4. **Implementation References**
   - Standard: `impl: core,nn/Linear`
   - External: `impl: external(provider=`lmstudio`, model=`qwen`)`

5. **Graph Connections** (Supports all 3 styles found in examples)
   - Inline: `in -> Linear(512) -> out`
   - Indented without arrows:
     ```
     in ->
       Linear(512)
       GELU()
       out
     ```
   - Indented with arrows:
     ```
     in ->
       Linear(512) ->
       GELU() ->
       out
     ```
   - Multiple connections in same graph (properly delimited)

6. **Match Expressions**
   - Patterns with wildcards and captures
   - Guards with boolean expressions
   - Multi-arm matching

7. **Bindings**
   - Let sections (lazy)
   - Set sections (eager)

## Key Design Decisions

### 1. Indentation Handling Strategy

**Decision:** Grammar ignores physical indentation; AST builder will handle it.

**Rationale:**
- pest doesn't natively support Python-style INDENT/DEDENT tokens
- Cleaner separation of concerns: grammar defines syntax, AST validates structure
- Allows grammar to focus on logical structure rather than whitespace mechanics

**Implementation Plan:**
- Grammar uses `:` + `NEWLINE` to signal block starts
- AST builder will:
  - Track indentation levels while traversing parse tree
  - Validate consistent indentation
  - Report indentation errors with proper source spans

### 2. Flexible Port Syntax

**Decision:** Support both inline and indented port definitions.

**Why:** All current examples use inline style (`in: [shape]`), but the parser supports indented multi-port. Keeping both ensures backward compatibility.

**Note:** We can standardize later if desired, but would require migrating examples.

### 3. Connection Pipeline Styles

**Decision:** Support three connection styles found in examples:
1. Inline with explicit arrows: `a -> b -> c`
2. Indented without arrows: `a ->\n  b\n  c` (implicit arrows)
3. Indented with arrows: `a ->\n  b ->\n  c ->\n  d`

**Technical Challenge:** Distinguishing when an indented pipeline ends and a new connection begins.

**Solution:** Use negative lookahead to detect new connections:
```pest
indented_pipeline_item = {
    (!(ref_endpoint ~ arrow ~ NEWLINE) ~ endpoint ~ arrow ~ NEWLINE)
  | (!(ref_endpoint ~ arrow ~ NEWLINE) ~ endpoint ~ NEWLINE)
}
```

This stops consuming endpoints when we see a pattern like `identifier ->` at the start of a line (new connection).

### 4. Keyword vs Identifier Handling

**Decision:** `in` and `out` are context-sensitive - keywords in some positions, identifiers in others.

**Example:**
```neuroscript
in -> out   # Both are port references (identifiers)

in:         # Keyword introducing port section
  x: [dim]
```

**Implementation:** The grammar rule `ref_endpoint` explicitly allows `keyword_in | keyword_out` as alternatives to `ident`.

## Grammar Structure Overview

```
program
├── use_stmt*
└── neuron_def*
    ├── params?
    └── neuron_section+
        ├── in_section
        ├── out_section  
        ├── impl_section
        ├── let_section
        ├── set_section
        └── graph_section
            └── connection+
                ├── endpoint (source)
                ├── arrow
                └── connection_tail
                    ├── endpoint+ (inline or indented)
                    └── match_expr?
```

## Files Created

1. **`src/grammar/neuroscript.pest`** (250 lines)
   - Complete PEG grammar definition
   - Well-commented sections
   - All NeuroScript features

2. **`src/grammar/mod.rs`** (150 lines)
   - pest parser integration
   - 22 comprehensive tests
   - Test macro for example files

3. **Modified `Cargo.toml`**
   - Added `pest = "2.7"`
   - Added `pest_derive = "2.7"`

4. **Modified `src/lib.rs`**
   - Added `pub mod grammar;`

## Current Limitations

1. **No indentation validation yet**
   - Grammar parses structure correctly
   - Need AST builder to validate indent consistency
   - This is expected and per the plan

2. **Parse trees only**
   - Grammar produces pest `Pair` trees
   - Need AST builder to convert to IR types
   - Next phase of work

3. **Error messages**
   - Currently pest's default error messages
   - Will improve when wrapping with miette in AST builder

## Next Steps (Recommended)

### Phase 2: AST Builder (Week 2)

1. **Create `src/grammar/ast.rs`**
   - Convert pest `Pair<Rule>` trees to IR types
   - Implement `build_program()` as entry point
   - Handle all grammar rules

2. **Create `src/grammar/indent.rs`**
   - Track indentation state during AST building
   - Validate consistent indentation
   - Report errors with source spans

3. **Create `src/grammar/error.rs`**
   - Convert pest errors to existing `ParseError` types
   - Preserve miette diagnostic features
   - Maintain error message quality

4. **Testing strategy:**
   - Start with simple examples (Linear, GELU)
   - Add tests for each IR type
   - Validate against existing parser output
   - Ensure all 70+ integration tests pass

### Phase 3: Integration (Week 3)

1. Compare outputs between old and new parsers
2. Run full test suite
3. Benchmark performance
4. Address any discrepancies

## Questions for Review

### Grammar Review

1. **Is the grammar structure clear and maintainable?**
   - Each section well-commented
   - Logical grouping of rules
   - Appropriate level of detail?

2. **Port syntax flexibility:**
   - Keep supporting both inline and indented?
   - Or standardize on one (requires example migration)?
   - Current: supports both for compatibility

3. **Connection styles:**
   - All three styles (inline, indented-no-arrows, indented-with-arrows) needed?
   - Current: all supported because found in examples

4. **Keyword handling:**
   - `in`/`out` as context-sensitive keywords acceptable?
   - Alternative: use different keyword for port references?

### Next Phase Approval

5. **Proceed with AST builder implementation?**
   - Start with basic structure (Program, NeuronDef, etc.)
   - Add indentation validation
   - Target: simple examples parsing to IR by end of Week 2

6. **Testing approach:**
   - Parallel implementation (old parser still active)?
   - Feature flag for gradual migration?
   - Or direct replacement after validation?

## Files for Your Review

Please review:
1. **`src/grammar/neuroscript.pest`** - The complete grammar
2. **`src/grammar/mod.rs`** - Tests demonstrating coverage
3. This document - Design decisions and next steps

Let me know if you'd like any changes to the grammar before I proceed with the AST builder!
