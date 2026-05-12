# Changelog

All notable changes to NeuroScript will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.2] - Unreleased

Sprint 4 internal refactors plus the Manifold-Constrained Hyper-Connection (mHC)
primitive family and supporting runtime. Some pieces in this release line
originated on `main` (mHC family via PR #169); the WedLM-DS model design that
was bundled with that PR is **not** part of this release line.

### Added — Manifold-Constrained Hyper-Connection (mHC)

- **mHC primitive family** — `ManifoldHyperConnect` (composite), `HyperExpand`,
  `HyperCollapse`, `LearnableResidual` with Birkhoff-polytope-constrained
  routing for parameter-efficient fine-tuning (`stdlib/ManifoldHyperConnect.ns`,
  `stdlib/LearnableResidual.ns`)
- **MoE / routing primitives** — `MoERouter` (softmax top-k) and
  `SigmoidMoERouter` (DeepSeek-V3-style sigmoid with auxiliary-loss-free load
  balancing) (`stdlib/MoERouter.ns`, `stdlib/SigmoidMoERouter.ns`)
- **Diffusion / multi-token primitives** — `DenoisingHead` (MLM-style head for
  masked diffusion) and `MultiTokenPredictionHead` (predicts N future tokens)
  (`stdlib/DenoisingHead.ns`, `stdlib/MultiTokenPredictionHead.ns`)
- **MLA + SSM primitives** — `MultiHeadLatentAttention` (KV-compressed MLA,
  DeepSeek-V2/V3) and `MambaBlock` (selective SSM structural placeholder)
  (`stdlib/MultiHeadLatentAttention.ns`, `stdlib/MambaBlock.ns`)
- **Python runtime** — `neuroscript_runtime/primitives/diffusion.py`,
  `routing.py`, `ssm.py`; expanded `connections.py` and `attention.py` with
  the new primitive implementations and Sinkhorn-Knopp utility
- **mHC training/eval scripts** — `examples/train_mhc_adapter.py` (Qwen2.5-7B /
  Phi-3-mini fine-tuning with mHC vs. LoRA at matched param budgets,
  HuggingFaceTB/smoltalk dataset, MPS-friendly) and `examples/demo_mhc.py`
  (parameter-comparison demo)
- **Wasteland real-world examples** — `examples/real_world/wasteland_briefing_v1.ns`,
  `v2.ns`, `v3_mhc.ns`, `wasteland_coder_v1.ns`, `wasteland_hybrid_v1.ns`,
  `wasteland_moe_v1.ns`
- **Website docs** — `website/docs/examples/mhc-transformer.mdx`,
  Connections primitive docs for `ManifoldHyperConnect`, `HyperExpand`,
  `HyperCollapse`
- Stdlib registry expanded from 72 to 77 primitives

### Added — Sprint 4 Language & Tooling

- **Diagnostic codes for `ValidationError`** with structured miette
  diagnostics; wildcard-constraint documentation (#75, #165 — Hal, PR #168)
- **`Binding::new()` factory + endpoint walker rename** — `Binding` literals
  use a factory constructor; `call_to_result` renamed to `endpoint_to_result`
  for accuracy (#162, #163 — Sonny, PR #167)
- **`EndpointVisitor` trait** unifying endpoint walkers across codegen,
  validator, and passes (#126 — Chappie, merged into _dev locally)
- **`contract_resolver.rs` decomposed** from a single 1,740-line file into a
  submodule directory: `call_sites.rs`, `detection.rs`, `matching.rs`,
  `resolution.rs`, `mod.rs`, plus a dedicated `tests.rs` (#127 — Bishop,
  merged into _dev locally)
- **Dimension-solving deduplicated** — validator now delegates shape
  compatibility to `ShapeInferenceEngine` rather than re-implementing the
  algebra (#130 — Roy, merged into _dev locally)
- **CI: self-hosted runner workflow** for targeting `aibox`

### Added — Orchestration / Agent System

- Persona memory files: `CLAUDE.ava.md`, `CLAUDE.roy.md`, `CLAUDE.vision.md`
- `9bd4be1` onboarded neuroscript-rs to the cross-project orchestration system

### Changed

- Stdlib registry exports updated to include `DenoisingHead`,
  `MultiTokenPredictionHead`, `SigmoidMoERouter`, `MoERouter`, `MambaBlock`,
  `LearnableResidual`, `MultiHeadLatentAttention`
- `train_mhc_adapter.py` default base model is `Qwen/Qwen2.5-7B`
- Removed unused `_gen` param in codegen, addressed an unreachable check, and
  added a `SYNC` marker comment on the `ValidationError` variant-count
  constant (#155, #159, #160 — Samantha, PR #166)

### Fixed

- `routing.py`: `SigmoidMoERouter.forward` and `MoERouter.forward` replaced
  O(tokens × experts × k) nested loops with `index_add_` pattern (still an
  O(num_experts) Python loop; true zero-loop vectorization requires padding
  or scatter kernels)
- `__init__.py`: removed `sinkhorn_knopp` from public `__all__` (still
  importable as an internal utility)
- `MambaBlock.ns`: doc comment clarifies the runtime is a structural
  placeholder; production use should substitute the `mamba-ssm` fused kernel
- mHC training on MPS: dtype, API, and double-residual fixes

## [0.6.1] - 2026-03-07

Sprint 3: 12 issues resolved by 9 AI agents across 3 batches. See [Agent Scoreboard](docs/AGENT-SCOREBOARD.md).

### Added

- **Logical operators `&&`/`||` in guard expressions** with correct precedence (OR < AND < comparison) (#121, PR #144 — Vision)
- **Source spans for `MutualLazyRecursion` errors** — `Binding` struct carries `Option<SourceSpan>`, miette diagnostics show source context (#117, PR #145 — Ava)
- **Wildcard `*` matches multiple leading dimensions** — `[batch, seq, dim]` now unifies with `[*, in_dim]`, unblocking stdlib attention neurons (#119, PR #149 — Roy)
- **Variadic element-wise shape unification** with abstract-shape guard to avoid stdlib false positives (#118, PR #147 — Dolores)
- Regression test for embeddings.ns fat arrow syntax (#120, PR #146 — TARS)
- CLI binary existence assertion with helpful error message (#116, PR #148 — Sonny)
- Agent Scoreboard with persona roster and sprint history
- GitHub Issues-centric sprint workflow with planner/implementer agent definitions
- `draft-pr-guard.yml` GitHub Actions workflow to block merging draft PRs
- `CONTRIBUTING.md` guide for external contributors
- `LICENSE` (MIT)
- `.github/FUNDING.yml` for sponsorship

### Changed

- **`process_destination` decomposed** from 850-line monolith into 10 focused per-variant handlers (#128, PR #150 — Bishop)
- **Non-test `unwrap()` calls eliminated** (~150 → 0) — codegen uses `?` propagation, AST builder uses `.expect()` with grammar guarantees (#129, PR #151 — Chappie)
- README rewritten for public audience with badges, quick-start, and feature overview (#140)
- CONTRIBUTING.md clarifies that CLAUDE.md doubles as AI assistant configuration
- Retired `backlog-worker.md` agent in favor of `planner.md` + `implementer.md`

### Fixed

- **ValidationError PartialEq safety** — replaced `unreachable!()` with `_ => false` fallback (#114, PR #142 — Samantha)
- **Hardcoded agent paths** replaced with `$(git rev-parse --show-toplevel)` (#125, PR #148 — Sonny)
- **Website `file://` dependency** — `docusaurus-llms-generator` moved to `optionalDependencies` with runtime fallback (#124, PR #146 — TARS)
- Documented non-recursive Tuple/Call endpoint cases in `compute_reachability` (#115, PR #142 — Samantha)

## [0.6.0] - 2026-03-07

### Fixed

- **Lazy instantiation consistency** — unified two divergent codegen paths (Ref and Call) for `@lazy` bindings so both resolve args identically (#108)
- **Validator no longer mutates IR** — reachability computation moved to a dedicated optimizer pass, keeping the validator read-only (#107)
- **Snake_case acronym handling** — `ReLU` now generates `relu_1` instead of `re_l_u_1` (#106)
- **PortMismatch error messages** — restored port names and shape details in validation errors (#105)
- **Double validation output** — non-spanned errors no longer printed twice (#103)
- **Silent error dropping** — all validation errors now rendered regardless of ordering (#102)
- **build.rs stdlib tracking** — emit `rerun-if-changed` per `.ns` file instead of top-level directory (#101)
- **wait-for-review.sh race condition** — track pending state to avoid accepting stale check results

### Added

- Comprehensive single-page language reference (`docs/language-reference.md`) (#104)
- `conventions.md` living document for cross-sprint knowledge capture
- Knowledge capture design doc (`docs/plans/2026-03-07-knowledge-capture-design.md`)

### Infrastructure

- `wait-for-review.sh` script for CI polling with macOS notifications (#99)
- `backlog-worker` agent definition for sprint automation (#99)

## [0.5.0] - 2026-03-07

### Fixed

- Mutual `@lazy` recursion detection in binding validation (#84)
- Miette `Diagnostic` derive on `ValidationError` with source spans (#93)
- Source spans on `ReshapeExpr` and `TransformAnnotation` (#91)
- String injection risk — `sanitize_python_ident()` applied to all user strings in codegen (#89)
- `Value::String` escaping in generated Python (backslashes, quotes, newlines, tabs) (#95)
- Sequential pseudo-neuron recognized in validator (#87)
- Embeddings.ns updated to fat arrow reshape syntax (#83)

### Added

- `MultiHeadAttention.ns` composite neuron implementation (#90)
- Expanded `MetaNeurons.ns` with 6 routing/composition patterns (#94)
- End-to-end codegen snapshot test with adversarial identifiers (#96)
- Backlog-worker agent and wait-for-review CI script (#99)

### Changed

- Unified dim-to-Python conversion into single `value_to_python_dim` function (#97)
- Auto-discover stdlib files via `build.rs` instead of manual `include_str!` (#92)
- Simplified CI workflows to direct self-hosted jobs (#82)

## [0.4.0] - 2026-03-06

### Added

- Self-hosted runner with GitHub-hosted fallback (#81)
- Comprehensive review findings: 13 critical fixes, 24 warnings resolved (#80)
- Shape system hardened: all 4 original critical issues fixed
- Table-driven stdlib registry (#73)
- Dimension bindings tracked in `shapes_compatible` (#78)
- `Dim::Global` evaluation and unification (#72)
- Unroll expansion limit (#71)
- WASM check CI workflow (#70)
- `visitor.rs` shared endpoint walker (#80)
- Compile-time contract resolution `match(param):` (#80)

[0.6.0]: https://github.com/severeon/neuroscript-rs/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/severeon/neuroscript-rs/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/severeon/neuroscript-rs/releases/tag/v0.4.0
