# PR Comment Audit

**Date:** 2026-03-05
**PRs Analyzed:** 15 (#25-39)
**Total Review Items Found:** 113
**Still Open:** 48 | **Resolved:** 56 | **Deferred:** 2 | **Cannot Determine:** 7

---

## Summary of Open Issues

### High Severity

| ID | PR | Summary | File | Status |
|----|-----|---------|------|--------|
| PR32-1 | #32 | Hardcoded local deploy path in package.json | website/package.json | still_open |
| PR32-2 | #32 | Local file dependency for docusaurus-llms-generator | website/package.json | still_open |

### Medium Severity

| ID | PR | Summary | File | Status |
|----|-----|---------|------|--------|
| PR39-13 | #39 | No semantic validation that @wrap target accepts Neuron-typed first param | src/desugar.rs | still_open |
| PR39-20 | #39 | `__sequential__` can collide with user-defined neuron names | src/validator/core.rs | still_open |
| PR39-33 | #39 | No integration test for @wrap inside an unroll block | -- | still_open |
| PR34-57 | #34 | `Others` in @repeat(copy) misclassified as "not new" | src/codegen/forward.rs | still_open |
| PR34-58 | #34 | eprintln! in production codegen for @reduce fallback | src/codegen/forward.rs | still_open |
| PR34-59 | #34 | Two distinct dim-to-Python conversion paths (context-aware vs not) | src/codegen/forward.rs | still_open |
| PR34-29 | #34 | `Others` -> `Dim::Wildcard` loses rank-collapsing semantics | src/interfaces.rs | still_open |
| PR34-30 | #34 | `to_shape()` drops Binding expression constraint | src/interfaces.rs | still_open |
| PR34-39 | #34 | InvalidReshape/InvalidAnnotation lack miette source spans | src/interfaces.rs | still_open |
| PR34-21 | #34 | No validation that @reduce target dims reachable from source | src/validator/ | still_open |
| PR34-51 | #34 | Rank-delta fallback silently generates wrong code for non-trailing reductions | src/codegen/forward.rs | still_open |
| PR34-55 | #34 | @reduce doesn't validate rank actually decreases | src/validator/ | still_open |
| PR34-53 | #34 | `literal_product` uses i64 instead of BigUint | src/validator/symbol_table.rs | still_open |
| PR34-60 | #34 | No test for @reduce/fat-arrow inside match arm pipeline | -- | still_open |
| PR38-1 | #38 | Mobile deep-link does not show detail panel | website/.../NeuronGenealogy/index.js | still_open |
| PR37-1 | #37 | Einsum stdlib docstring says "1-3 inputs" but only 2 ports exist | stdlib/primitives/Einsum.ns | still_open |
| PR36-2 | #36 | No CI step for WASM build target | CI config | still_open |
| PR35-2 | #35 | Test generates invalid Python (`batch` undefined at runtime) | src/codegen/tests.rs | still_open |
| PR33-9 | #33 | ALiBi absolute-distance bias wrong for causal/decoder attention | neuroscript_runtime/.../embeddings.py | still_open |
| PR31-4 | #31 | WaveNetBlock doc says "causal" but padding is symmetric | stdlib/WaveNetBlock.ns | still_open |
| PR31-8 | #31 | ConstScale docs describe Scale instead of ConstScale | website/docs/.../constScale.md | still_open |
| PR28-4 | #28 | Contract resolution shape matching uses only defaults, not call-site args | src/contract_resolver.rs | still_open |
| PR25-1 | #25 | Hardcoded stdlib include_str! list needs build.rs automation | src/stdlib.rs | still_open |

### Low Severity

| ID | PR | Summary | File | Status |
|----|-----|---------|------|--------|
| PR39-14 | #39 | Fragile grammar test assertions (break-on-first-match) | src/grammar/tests.rs | still_open |
| PR39-15 | #39 | `__sequential__` tracking may classify composites as primitives | src/codegen/instantiation.rs | still_open |
| PR39-22 | #39 | Nested @wrap silently dropped | src/desugar.rs | still_open |
| PR39-26 | #39 | Silent Value::Call fallback in build_wrap_endpoint | src/grammar/ast.rs | still_open |
| PR39-29 | #39 | test_parse_wrap_inline_pipeline only checks non-empty | src/grammar/tests.rs | still_open |
| PR39-32 | #39 | Missing @wrap kwargs test | src/grammar/tests.rs | still_open |
| PR39-34 | #39 | `__sequential__` bypasses arity/shape validation | src/validator/symbol_table.rs | still_open |
| PR39-35 | #39 | Spurious primitive insertion for binding-reference args | src/codegen/instantiation.rs | still_open |
| PR39-36 | #39 | Fragile token skipping in build_wrap_endpoint | src/grammar/ast.rs | still_open |
| PR39-28 | #39 | WrapExpr.id reused by desugared Call | src/desugar.rs | still_open |
| PR34-20 | #34 | 1598-line plan document still committed | docs/plans/ | still_open |
| PR34-49 | #34 | `others` globally reserved keyword (breaking change) | src/grammar/neuroscript.pest | still_open |
| PR34-10 | #34 | Fragile positional next() calls in build_transform_annotation | src/grammar/ast.rs | still_open |
| PR34-28 | #34 | process_destination function very large (750+ lines) | src/codegen/forward.rs | still_open |
| PR34-41 | #34 | No test for @repeat(copy) with all-wildcard source | -- | still_open |
| PR34-62 | #34 | @reduce after top-level If source unsupported | src/codegen/forward.rs | still_open |
| PR34-63 | #34 | id + 1000 offset in test helper is fragile | src/validator/tests/reshape.rs | still_open |
| PR34-64 | #34 | @repeat expand branch duplication | src/codegen/forward.rs | still_open |
| PR38-2 | #38 | Unused `listRef` in NeuronList.js | website/.../NeuronList.js | still_open |
| PR38-3 | #38 | Missing `localSearch` useEffect dependency | website/.../NeuronControls.js | still_open |
| PR38-4 | #38 | Full file source shown for multi-neuron files | website/.../NeuronSourceCode.js | still_open |
| PR38-6 | #38 | Email exposed in funding.json | website/static/funding.json | still_open |
| PR37-3 | #37 | LearnedPool/@reduce(AttentionPool) example unvalidated | tutorial docs | still_open |
| PR36-1 | #36 | collect_calls ignores TransformAnnotation::Neuron names | src/wasm.rs | still_open |
| PR35-1 | #35 | Doc comment missing overflow path for shape_literal_product | src/validator/symbol_table.rs | still_open |
| PR35-3 | #35 | Test missing integer division assertion | src/codegen/tests.rs | still_open |
| PR27-1 | #27 | `strip_preamble` uses `contains` instead of `ends_with` | src/codegen/generator.rs | still_open |
| PR27-3 | #27 | `modules_for_primitives` silently ignores External variants | src/stdlib_registry.rs | still_open |
| PR26-1 | #26 | No error boundary around NeuroEditor | website/src/pages/index.js | still_open |
| PR25-5 | #25 | stdlib clone overhead per WASM compilation | src/wasm.rs | still_open |
| PR33-8 | #33 | Dropblock naming should be DropBlock (capital B) | neuroscript_runtime/.../regularization.py | still_open |
| PR33-10 | #33 | SpecAugment per-batch Python loop performance (TODO remains) | neuroscript_runtime/.../regularization.py | still_open |
| PR33-11 | #33 | DilatedConv missing constructor validation | neuroscript_runtime/.../convolutions.py | still_open |
| PR33-13 | #33 | ALiBi categorized under embeddings instead of attention | neuroscript_runtime/.../embeddings.py | still_open |
| PR31-7 | #31 | WaveNetBlock simplified gating not documented | stdlib/WaveNetBlock.ns | still_open |
| PR31-11 | #31 | Python tests validate reference impls, not compiled output | -- | still_open |
| PR30-2 | #30 | Missing version info in tutorial docs | website/docs/ | still_open |
| PR28-6 | #28 | find_call_sites only handles positional args, not kwargs | src/contract_resolver.rs | still_open |

---

## Detailed Findings by PR

### PR #39: feat: hyper-connections and @wrap annotation (17 comments)

**Total items: 36** | Open: 13 | Resolved in PR: 21 | Resolved later: 0 | Can't determine: 2

#### Open Items

- **[PR39-13]** (medium) No semantic validation that @wrap target accepts Neuron-typed first param. `@wrap(Linear, 512): attn` desugars to `Linear(attn, 512)` -- the validator accepts it and codegen emits broken Python.
  - *Original:* "the validator accepts it and codegen emits broken Python"
  - *Current state:* No validation check for first-param `: Neuron` annotation in desugar.rs. Author deferred: "Tracked for follow-up."
  - *File:* src/desugar.rs

- **[PR39-20]** (medium) `__sequential__` can collide with user-defined neuron names. No validation prevents user-defined `__dunder__` names.
  - *Original:* "Nothing prevents a user from writing `neuron __sequential__:`"
  - *Current state:* No rejection of `__`-prefixed names in validator/core.rs. Author deferred.
  - *File:* src/validator/core.rs

- **[PR39-33]** (medium) No integration test for @wrap inside an unroll block. The ordering fix (expand_unrolls before desugar_wraps) is critical but untested.
  - *Original:* "A test nesting @wrap inside an unroll template would prevent regression."
  - *Current state:* No such test exists.

- **[PR39-14]** (low) Grammar tests use `break` on first match, so inner assertions on nested wraps never run.
  - *File:* src/grammar/tests.rs:522-619

- **[PR39-15]** (low) `__sequential__` tracking in instantiation.rs may classify composite neurons as primitives via the else branch.
  - *File:* src/codegen/instantiation.rs:139-140

- **[PR39-22]** (low) Nested @wrap (e.g., `@wrap(A): -> @wrap(B): x`) is silently dropped. Documented limitation.
  - *File:* src/desugar.rs

- **[PR39-26]** (low) Silent Value::Call fallback in build_wrap_endpoint should be an explicit ParseError.
  - *File:* src/grammar/ast.rs

- **[PR39-29]** (low) `test_parse_wrap_inline_pipeline` only asserts `!connections.is_empty()` -- no check for Wrap endpoint presence.
  - *File:* src/grammar/tests.rs:604-619

- **[PR39-32]** (low) No test exercises keyword arguments in @wrap calls (e.g., `@wrap(Wrapper, kw=val):`).
  - *File:* src/grammar/tests.rs

- **[PR39-34]** (low) `__sequential__` bypasses arity/shape validation via wildcard port return in symbol_table.
  - *File:* src/validator/symbol_table.rs:293-295

- **[PR39-35]** (low) Possible spurious primitive insertion for binding-reference args in `__sequential__` instantiation.
  - *File:* src/codegen/instantiation.rs:139-140

- **[PR39-36]** (low) Fragile token skipping in build_wrap_endpoint -- three consecutive `inner.next()` calls skip by position without checking rule type.
  - *File:* src/grammar/ast.rs

- **[PR39-28]** (low) Desugared Call inherits the Wrap's id, so both share an id in debug dumps. Cosmetic.
  - *File:* src/desugar.rs:90

#### Resolved During PR (21 items)
PR39-1 (validate() doc comment), PR39-2 (unused param removed), PR39-3 (HCWidth init bias), PR39-4 (assertions added), PR39-5 (error instead of drop), PR39-6 (lazy instantiation comment), PR39-7 (.contiguous()), PR39-8 (alphabetical order), PR39-9 (plan document removed), PR39-10 (einsum transpose bug), PR39-11 (panic->error), PR39-12 (unwrap->ok_or_else), PR39-16 (dup of PR39-3), PR39-17 (or_insert_with documented), PR39-18 (span captured), PR39-19 (assert->raise), PR39-21 (Value::Call recursion), PR39-24 (Tuple doesn't contain Endpoint), PR39-25 (__wrap naming), PR39-27 (HCDepth init), PR39-30 (comment accuracy)

---

### PR #34: feat: add fat arrow (=>) shape transform operator (35 comments)

**Total items: 64** | Open: 19 | Resolved in PR: 37 | Resolved later: 4 | Can't determine: 4

#### Open Items

- **[PR34-57]** (medium) `Others` in @repeat(copy) misclassified as "not new" via `_ => false` fallback. `expand(-1)` for an `Others` slot has undefined semantics.
  - *Current state:* The `_ => false` fallback still present in forward.rs.
  - *File:* src/codegen/forward.rs

- **[PR34-58]** (medium) `eprintln!` warning in production codegen for @reduce fallback. Should be a proper diagnostic, not raw stderr.
  - *Current state:* Line 1076 still uses `eprintln!`.
  - *File:* src/codegen/forward.rs:1076

- **[PR34-59]** (medium) Two distinct dim-to-Python conversion paths -- one context-aware, one not. If param names appear in repeat dims, the non-context-aware path emits raw name instead of `self.name`.
  - *File:* src/codegen/forward.rs

- **[PR34-29]** (medium) `ReshapeDim::Others` maps to `Dim::Wildcard` in `to_shape()`, losing rank-collapsing semantics. TODO comment exists at interfaces.rs:325.
  - *File:* src/interfaces.rs:338

- **[PR34-30]** (medium) `to_shape()` drops Binding expression constraint -- `dh=dim/heads` becomes `Dim::Named("dh")`. Shape solver can't verify divisibility. TODO at interfaces.rs:324.
  - *File:* src/interfaces.rs

- **[PR34-39]** (medium) `InvalidReshape` and `InvalidAnnotation` error variants lack miette source spans. Author deferred: "Will add Span in a separate PR."
  - *File:* src/interfaces.rs:683-689

- **[PR34-21]** (medium) No validation that @reduce target dims are reachable from source. `=> @reduce(mean) [b, c, x]` where `x` isn't a source dim generates undefined Python name.
  - *File:* src/validator/

- **[PR34-51]** (medium) Rank-delta fallback for @reduce assumes trailing dimensions. Non-trailing reductions generate wrong indices. Only emits eprintln warning.
  - *Current state:* Lines 1071-1082 document limitation, eprintln at 1076.
  - *File:* src/codegen/forward.rs:1065-1084

- **[PR34-55]** (medium) @reduce doesn't validate that target shape has fewer dimensions than source.
  - *File:* src/validator/

- **[PR34-53]** (medium) `literal_product` uses i64 with `checked_mul` instead of BigUint. Inconsistent with rest of shape system which uses BigUint to avoid overflow.
  - *File:* src/validator/symbol_table.rs:597-621

- **[PR34-60]** (medium) No test for @reduce/fat-arrow inside match arm pipeline.

- **[PR34-20]** (low) 1598-line plan document still committed at docs/plans/2026-02-26-fat-arrow-shape-transforms.md.

- **[PR34-49]** (low) `others` globally reserved as keyword. Prevents use as identifier anywhere in .ns files. Undocumented breaking change.
  - *File:* src/grammar/neuroscript.pest:42,106

- **[PR34-10]** (low) Fragile positional `next()` calls in build_transform_annotation. Author declined: "follows existing pattern."
  - *File:* src/grammar/ast.rs

- **[PR34-28]** (low) process_destination now 750+ lines. Author acknowledged as follow-up.
  - *File:* src/codegen/forward.rs:377-1226

- **[PR34-41]** (low) No test for @repeat(copy) with all-wildcard source shape.

- **[PR34-62]** (low) @reduce after top-level If source remains unsupported. Documented limitation.
  - *File:* src/codegen/forward.rs

- **[PR34-63]** (low) `id + 1000` offset in test helper to avoid ID collision is fragile.
  - *File:* src/validator/tests/reshape.rs

- **[PR34-64]** (low) @repeat expand branches functionally identical -- could be unified.
  - *File:* src/codegen/forward.rs

#### Resolved During PR (37 items)
PR34-1 through PR34-9, PR34-12 through PR34-19, PR34-24 through PR34-25, PR34-27, PR34-31 through PR34-34, PR34-37 through PR34-38, PR34-40, PR34-43 through PR34-48, PR34-50, PR34-52, PR34-61

#### Resolved After PR Merge (4 items)
PR34-6 (port compatibility), PR34-11 (element-count validation), PR34-26 (compile-time check), PR34-35 (shape skip narrowed)

---

### PR #38: Modernize neuron genealogy page (5 comments)

**Total items: 6** | Open: 4 | Resolved: 0 | Can't determine: 2

#### Open Items

- **[PR38-1]** (medium) Mobile deep-link (`/neuron-genealogy#TransformerBlock`) sets selectedNeuron but never calls `setMobileShowDetail(true)`. Detail panel stays hidden.
  - *File:* website/src/components/NeuronGenealogy/index.js:93-110

- **[PR38-2]** (low) Unused `listRef` created with `useRef(null)` but never read.
  - *File:* website/src/components/NeuronGenealogy/NeuronList.js:5,30

- **[PR38-3]** (low) Missing `localSearch` dependency in useEffect array. Will trigger eslint warning.
  - *File:* website/src/components/NeuronGenealogy/NeuronControls.js:29

- **[PR38-4]** (low) Source code display shows entire .ns file, not just selected neuron definition. MetaNeurons.ns shows all 16 definitions.
  - *File:* website/src/components/NeuronGenealogy/NeuronSourceCode.js

---

### PR #37: docs: add einops and einsum tutorial (9 comments)

**Total items: 6** | Open: 1 | Resolved: 0 | Deferred: 1 | Can't determine: 3

#### Open Items

- **[PR37-1]** (medium) Einsum.ns docstring says "Supports 1-3 input tensors" but the neuron only has 2 input ports (`a` and `b`).
  - *File:* stdlib/primitives/Einsum.ns:16

#### Deferred

- **[PR37-2]** (low) `@repeat(copy)` naming discrepancy with PyTorch behavior. Author said: "grammar/language design decision, not a docs fix."

---

### PR #36: fix: handle Endpoint::Reshape in WASM build (1 comment)

**Total items: 2** | Open: 2 | Resolved: 0

#### Open Items

- **[PR36-2]** (medium) No CI step for `cargo check --target wasm32-unknown-unknown` to catch WASM breakage.

- **[PR36-1]** (low) `collect_calls` ignores `TransformAnnotation::Neuron` names in Reshape endpoints.
  - *File:* src/wasm.rs:205

---

### PR #35: fix: address code review issues from fat arrow (1 comment)

**Total items: 3** | Open: 3 | Resolved: 0

#### Open Items

- **[PR35-2]** (medium) `test_codegen_fat_arrow_reshape_in_match_arm` has wildcard input but reshape target uses named `batch`. Generated Python would have undefined `batch` at runtime. Test only checks `.contains(".reshape(")`.
  - *File:* src/codegen/tests.rs:1007-1110

- **[PR35-1]** (low) Doc comment for `shape_literal_product` says "Returns None if any dim is not a Literal" but omits the `checked_mul` overflow None path.
  - *File:* src/validator/symbol_table.rs:595-596

- **[PR35-3]** (low) Test exercises `ReshapeDim::Binding { expr: BinOp::Div }` but never asserts that `//` (integer division) appears in output.
  - *File:* src/codegen/tests.rs:1102-1109

---

### PR #33: feat: add 8 Stage 1 stdlib primitives (3 comments)

**Total items: 14** | Open: 5 | Resolved later: 8 | Resolved in PR: 0

#### Open Items

- **[PR33-9]** (medium) ALiBi uses absolute-distance bias, which is only correct for bidirectional/encoder attention. For decoder-style transformers it incorrectly penalizes attending to earlier tokens. No documentation notes this limitation.
  - *File:* neuroscript_runtime/primitives/embeddings.py:502

- **[PR33-8]** (low) `class Dropblock` should be `class DropBlock` (capital B) for consistency with DropPath, DropConnect.
  - *File:* neuroscript_runtime/primitives/regularization.py:195

- **[PR33-10]** (low) SpecAugment per-batch Python loop has 128 CPU/GPU sync points per forward. TODO comment remains at line 395.
  - *File:* neuroscript_runtime/primitives/regularization.py:395

- **[PR33-11]** (low) DilatedConv missing constructor validation (`in_channels > 0`, etc.).
  - *File:* neuroscript_runtime/primitives/convolutions.py:224

- **[PR33-13]** (low) ALiBi categorized under embeddings.py instead of attention -- operates on attention scores, not embeddings.
  - *File:* neuroscript_runtime/primitives/embeddings.py

#### Resolved After Merge (8 items)
PR33-1 (SpecAugment off-by-one), PR33-2 (masks shared across batch), PR33-3 (ALiBi 3D check), PR33-4 (head count validation), PR33-5 (extra_repr), PR33-6 (Crop interface), PR33-7 (gamma clamp), PR33-12 (Clone attribute naming), PR33-14 (bias in extra_repr)

---

### PR #32: chore: remove duplicate primitive docs (1 comment)

**Total items: 2** | Open: 2 | Resolved: 0

#### Open Items

- **[PR32-1]** (high) Hardcoded local deploy path: `"deploy": "cp -r build/* /Users/tquick/projects/neuroscript-docs/"`. No other contributor can run `npm run deploy`.
  - *File:* website/package.json

- **[PR32-2]** (high) Local `file:../../docusaurus-llms-generator` dependency. `npm install` fails for anyone else.
  - *File:* website/package.json

---

### PR #31: stdlib phase5 - extended-architectures (8 comments)

**Total items: 11** | Open: 4 | Resolved later: 5 | Can't determine: 1

#### Open Items

- **[PR31-4]** (medium) WaveNetBlock doc comment says "dilated causal convolution" but padding `dilation * (kernel_size - 1) / 2` is symmetric (non-causal).
  - *File:* stdlib/WaveNetBlock.ns:2

- **[PR31-8]** (medium) ConstScale docs page describes Scale instead. Parameter listed as `dim` with shape `[*, dim]` instead of scalar `factor`.
  - *File:* website/docs/primitives/Operations/constScale.md

- **[PR31-7]** (low) WaveNetBlock simplified gating (`tanh * sigmoid` on same conv output) differs from original WaveNet (separate weight matrices). Not documented.
  - *File:* stdlib/WaveNetBlock.ns

- **[PR31-11]** (low) Python tests validate reference implementations, not compiled output. No end-to-end codegen validation tests.

#### Resolved After Merge (5 items)
PR31-1 (Linear->Conv1d), PR31-2 (dilation parameter), PR31-3 (odd-kernel docs), PR31-5 (test assertions restored), PR31-6 (Conformer architecture fixed)

---

### PR #30: docs: higher-order neurons tutorial (2 comments)

**Total items: 3** | Open: 1 | Resolved later: 1 | Can't determine: 1

#### Open Items

- **[PR30-2]** (low) No version badges or implementation status notes in tutorial pages.

---

### PR #29: docs: add compiler reference (3 comments)

**Total items: 2** | Open: 0 | Resolved in PR: 1 | Can't determine: 1

All actionable items resolved.

---

### PR #28: feat: higher-order neurons, named unrolls (4 comments)

**Total items: 7** | Open: 2 | Resolved later: 4 | Can't determine: 1

#### Open Items

- **[PR28-4]** (medium) Contract resolution shape matching uses only parameter defaults, not call-site arguments. Neurons without defaults may fail even when call site provides concrete values.
  - *File:* src/contract_resolver.rs

- **[PR28-6]** (low) `find_call_sites` only handles positional args, not kwargs. Documented as intentional but limits future use.
  - *File:* src/contract_resolver.rs

#### Resolved After Merge (4 items)
PR28-1 (multi-endpoint pipelines), PR28-2 (Neuron-typed param validation), PR28-3 (optimizer recurses into If), PR28-5 (lib.rs doc comment)

---

### PR #27: feat: add --bundle flag (2 comments)

**Total items: 3** | Open: 2 | Resolved: 0 | Can't determine: 1

#### Open Items

- **[PR27-1]** (low) `strip_preamble` uses `contains("\"\"\"")` instead of `ends_with` for closing triple-quotes. Fragile.
  - *File:* src/codegen/generator.rs:69

- **[PR27-3]** (low) `modules_for_primitives` silently ignores `ImplRef::External` variants in bundle mode. No warning or documentation.
  - *File:* src/stdlib_registry.rs:~758

---

### PR #26: feat: GPT-2 live demo (2 comments)

**Total items: 3** | Open: 1 | Can't determine: 2

#### Open Items

- **[PR26-1]** (low) No error boundary around NeuroEditor on front page. WASM compiler failure shows no user feedback.
  - *File:* website/src/pages/index.js

---

### PR #25: fix: embed stdlib in WASM (3 comments)

**Total items: 5** | Open: 2 | Resolved later: 2 | Can't determine: 1

#### Open Items

- **[PR25-1]** (medium) 82+ `include_str!()` calls hardcoded in stdlib.rs. Must be manually updated when stdlib files change. No build.rs automation.
  - *File:* src/stdlib.rs:156+

- **[PR25-5]** (low) `merge_programs()` clones entire stdlib on every WASM compilation. OnceLock caches parsed result but clone happens per call.
  - *File:* src/wasm.rs

#### Resolved After Merge (2 items)
PR25-2 (`fullSource` variable renamed away), PR25-4 (OnceLock caching implemented)

---

## Statistics

| PR | Total | Open | Resolved in PR | Resolved Later | Can't Determine |
|----|------:|-----:|---------------:|---------------:|----------------:|
| #39 | 36 | 13 | 21 | 0 | 2 |
| #38 | 6 | 4 | 0 | 0 | 2 |
| #37 | 6 | 1 | 0 | 0 | 4 |
| #36 | 2 | 2 | 0 | 0 | 0 |
| #35 | 3 | 3 | 0 | 0 | 0 |
| #34 | 64 | 19 | 37 | 4 | 4 |
| #33 | 14 | 5 | 0 | 8 | 0 |
| #32 | 2 | 2 | 0 | 0 | 0 |
| #31 | 11 | 4 | 0 | 5 | 1 |
| #30 | 3 | 1 | 0 | 1 | 1 |
| #29 | 2 | 0 | 1 | 0 | 1 |
| #28 | 7 | 2 | 0 | 4 | 1 |
| #27 | 3 | 2 | 0 | 0 | 1 |
| #26 | 3 | 1 | 0 | 0 | 2 |
| #25 | 5 | 2 | 0 | 2 | 1 |
| **Total** | **167** | **61** | **59** | **24** | **20** |

---

## Analysis

### Resolution Rate
- **59 of 167 items (35%)** were resolved within the PR itself
- **24 items (14%)** were resolved in subsequent work
- **61 items (37%)** remain open
- **20 items (12%)** could not be determined without running the code

### Hotspot Files (most open items)
1. `src/codegen/forward.rs` -- 8 open items (complexity, @reduce fallback, dim conversion paths)
2. `src/grammar/ast.rs` -- 4 open items (fragile token skipping, test weakness)
3. `src/interfaces.rs` -- 3 open items (Others semantics, Binding constraints, missing spans)
4. `src/desugar.rs` -- 3 open items (@wrap validation, nested @wrap, id reuse)
5. `website/package.json` -- 2 open items (hardcoded paths, local dependency)

### Patterns
- **Deferred validation** is the most common open category -- items where semantic checks were acknowledged but postponed (PR39-13, PR34-21, PR34-55, PR28-4)
- **Missing test coverage** for edge cases is the second most common (PR34-41, PR34-60, PR39-32, PR39-33, PR35-2, PR35-3)
- **Documentation inaccuracies** in stdlib and website docs tend not to get follow-up PRs (PR31-4, PR31-8, PR33-9, PR37-1)
- **Website/infra issues** (PR32-1, PR32-2) are high severity but likely low-traffic

### Highest Priority Actions
1. **Fix hardcoded paths in website/package.json** (PR32-1, PR32-2) -- blocks any contributor
2. **Add @wrap target validation** (PR39-13) -- generates broken Python for common misuse
3. **Fix @reduce fallback** (PR34-51, PR34-58) -- silently generates wrong code with only eprintln warning
4. **Add WASM CI step** (PR36-2) -- prevents regression class that already broke once
5. **Fix test that generates invalid Python** (PR35-2) -- test passes but generated code would crash

---

*Generated by analyzing 167 review items across 15 PRs (#25-#39). Items verified against current codebase on main branch (commit e56d0e2).*
