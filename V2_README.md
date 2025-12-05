# NeuroScript v2 Planning Documents

This directory contains comprehensive planning documents for NeuroScript v2 redesign.

---

## Quick Links

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **[V2_SUMMARY.md](V2_SUMMARY.md)** | Executive overview with key decisions | 5 min |
| **[NEUROSCRIPT_V2_PLAN.md](NEUROSCRIPT_V2_PLAN.md)** | Complete architecture plan (13 parts) | 30 min |
| **[RECURSION_RESEARCH.md](RECURSION_RESEARCH.md)** | Recursion strategies and implementation | 15 min |
| **[PARSER_COMPARISON.md](PARSER_COMPARISON.md)** | Parser library comparison with examples | 10 min |

---

## Start Here

### 5-Minute Overview
Read **[V2_SUMMARY.md](V2_SUMMARY.md)** for:
- Key changes from v1
- Architecture diagram
- Three main questions answered (recursion, first-class neurons, modularity)
- Timeline and next steps

### Decision-Maker's Guide
If you need to make decisions about v2, read:
1. **V2_SUMMARY.md** - Overall direction *(5 min)*
2. **PARSER_COMPARISON.md** - Why pest over hand-written *(10 min)*
3. **NEUROSCRIPT_V2_PLAN.md** Part 13 - Open questions *(5 min)*

### Implementer's Guide
If you're going to build v2, read in order:
1. **V2_SUMMARY.md** - Big picture *(5 min)*
2. **NEUROSCRIPT_V2_PLAN.md** - Full plan *(30 min)*
3. **RECURSION_RESEARCH.md** - How to handle recursion *(15 min)*
4. **PARSER_COMPARISON.md** - Parser implementation *(10 min)*

---

## Key Decisions Made

### ✅ Confirmed Choices

1. **Parser**: pest (PEG grammar)
   - Declarative, maintainable, good errors
   - See: PARSER_COMPARISON.md

2. **IR**: Three-level (HIR → MIR → LIR)
   - Clean backend separation
   - Enables multi-target compilation

3. **Backends**: PyTorch, JAX, ONNX (extensible)
   - Backend trait for third-party targets

4. **Recursion**: Multi-strategy
   - Static (Phase 1 MVP)
   - Dynamic (Phase 2)
   - Structural (Phase 3)
   - See: RECURSION_RESEARCH.md

5. **Type System**: Hindley-Milner + constraints
   - First-class neuron types
   - Shape inference built-in

6. **File Limit**: 300 lines (enforced by CI)
   - Forces modular design

### ❓ Open Questions

Need to decide before implementation:

1. **Higher-order neuron syntax** (explicit vs. inferred types)
2. **Dynamic recursion syntax** (built-in vs. library)
3. **Backend selection** (compile-time flag vs. all-at-once)
4. **Module import syntax** (Python-style vs. Rust-style)

See: NEUROSCRIPT_V2_PLAN.md Part 13

---

## Document Summaries

### [V2_SUMMARY.md](V2_SUMMARY.md)
**Quick executive overview**

- What's changing from v1
- Architecture at a glance
- Three key questions answered:
  1. How to handle recursion efficiently?
  2. How to make neurons first-class?
  3. How to stay modular with 300-line limit?
- Timeline and milestones
- Success metrics

**Read this first.**

---

### [NEUROSCRIPT_V2_PLAN.md](NEUROSCRIPT_V2_PLAN.md)
**Comprehensive architecture plan**

13-part detailed plan covering:
1. Lessons from v1
2. Recursion patterns
3. Multi-backend architecture
4. Parser library selection
5. First-class neurons
6. Modular file structure (300 lines)
7. Development phases (7 phases, 24 weeks)
8. Technical decisions
9. Migration from v1
10. Success metrics
11. Risk mitigation
12. Research questions
13. Open questions

**Deep dive for implementers.**

---

### [RECURSION_RESEARCH.md](RECURSION_RESEARCH.md)
**Recursion strategies deep dive**

7 sections covering:
1. Three types of neural recursion (static, dynamic, structural)
2. Weight sharing strategies
3. Implementation strategies for v2
4. Optimization techniques
5. Backend considerations (PyTorch, JAX, ONNX)
6. Recommendations by phase
7. Example: Universal Transformer

Includes:
- Code examples for each recursion type
- PyTorch lowering strategies
- ACT (Adaptive Computation Time) implementation
- TreeLSTM example

**Essential for understanding recursion design.**

---

### [PARSER_COMPARISON.md](PARSER_COMPARISON.md)
**Parser library evaluation**

Compares 4 libraries with code examples:
1. **pest** (PEG) - ⭐⭐⭐⭐⭐ Recommended
2. **lalrpop** (LALR) - ⭐⭐⭐⭐ Good alternative
3. **nom** (Combinators) - ⭐⭐⭐ Too low-level
4. **tree-sitter** (GLR) - ⭐⭐⭐⭐ Future LSP

Each section includes:
- Example grammar/code for same test case
- Pros and cons
- Performance comparison
- Verdict

**Justifies pest choice with evidence.**

---

## Timeline Overview

```
Phase 0: Infrastructure        [Weeks 1-2]   ← Set up pest, HIR
Phase 1: Type System          [Weeks 3-5]   ← Inference + shapes
Phase 2: MIR + Recursion      [Weeks 6-8]   ← Static recursion
Phase 3: PyTorch Backend      [Weeks 9-11]  ← MVP milestone
Phase 4: Multi-Backend        [Weeks 12-14] ← JAX + ONNX
Phase 5: Advanced Features    [Weeks 15-18] ← Higher-order neurons
Phase 6: Optimization         [Weeks 19-21] ← Production quality
Phase 7: Tooling              [Weeks 22-24] ← LSP + REPL
```

**Milestones**:
- **Week 11**: MVP (PyTorch-only, static recursion)
- **Week 14**: Multi-backend (PyTorch, JAX, ONNX)
- **Week 18**: Full features (higher-order, dynamic recursion)
- **Week 24**: Production-ready (LSP, optimizations)

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                     NeuroScript v2                       │
└──────────────────────────────────────────────────────────┘

    Source (.ns files)
         │
         ▼
    ┌─────────┐
    │  pest   │  Grammar-based parsing
    │ Parser  │  (declarative, maintainable)
    └────┬────┘
         │
         ▼
    ┌─────────────┐
    │  High IR    │  • Neuron types (first-class)
    │   (HIR)     │  • Generics and type variables
    └────┬────────┘  • Symbolic shapes
         │
         │ Type checking + Shape inference
         ▼
    ┌─────────────┐
    │  Mid IR     │  • SSA form
    │   (MIR)     │  • Control flow graph
    └────┬────────┘  • Tensor operations
         │
         │ Optimizations (inline, DCE, CSE, fusion)
         ▼
    ┌────┴────┐
    │ Backend │
    │Selection│
    └────┬────┘
         │
    ┌────┴────┬─────────┬─────────┐
    ▼         ▼         ▼         ▼
┌────────┐ ┌──────┐ ┌──────┐ ┌────────┐
│PyTorch │ │ JAX  │ │ ONNX │ │ Custom │
│  (LIR) │ │(LIR) │ │(LIR) │ │ (LIR)  │
└───┬────┘ └──┬───┘ └──┬───┘ └───┬────┘
    │         │        │         │
    ▼         ▼        ▼         ▼
  .py       .py     .onnx      ???
```

---

## Next Actions

### Immediate (Before Implementation)
1. ✅ Review planning documents
2. ❓ Answer open questions (syntax decisions)
3. ❓ Get team buy-in on architecture
4. ❓ Decide on timeline (full 24 weeks or MVP first?)

### Week 1 (Infrastructure)
1. Create `neuroscript-v2/` repository
2. Set up Cargo project with pest dependency
3. Write basic pest grammar for neuron definitions
4. Implement HIR type definitions
5. Add miette error infrastructure

### Week 2 (Parser)
1. Complete pest grammar (all syntax)
2. Implement parser (pest → HIR)
3. Add span tracking for errors
4. Port 10 core examples from v1
5. Set up snapshot testing

### Month 1 Goal
**Deliverable**: Can parse v1 examples into HIR with good error messages

---

## Success Criteria

### Must-Have (MVP - Week 11)
- ✅ Parse all v1 examples
- ✅ Type check with clear error messages
- ✅ Generate correct PyTorch code
- ✅ Support static recursion
- ✅ All files ≤300 lines (CI enforced)

### Should-Have (Multi-Backend - Week 14)
- ✅ 3 backends working (PyTorch, JAX, ONNX)
- ✅ Cross-backend validation
- ✅ Same semantics across targets

### Nice-to-Have (Production - Week 24)
- ✅ Higher-order neurons
- ✅ Dynamic recursion (ACT)
- ✅ LSP server (<100ms latency)
- ✅ REPL with type display
- ✅ Documentation generator

---

## Questions?

- **Technical**: See NEUROSCRIPT_V2_PLAN.md Part 13 (Open Questions)
- **Recursion**: See RECURSION_RESEARCH.md
- **Parser**: See PARSER_COMPARISON.md
- **Overview**: See V2_SUMMARY.md

**Ready to start?** Create `neuroscript-v2/` repo and begin Phase 0!

---

## File Manifest

```
neuroscript-rs/
├── V2_README.md              ← You are here
├── V2_SUMMARY.md             ← Start here (5 min read)
├── NEUROSCRIPT_V2_PLAN.md    ← Full plan (30 min read)
├── RECURSION_RESEARCH.md     ← Recursion deep dive (15 min)
└── PARSER_COMPARISON.md      ← Parser choice (10 min)
```

**Total reading time**: ~60 minutes for complete understanding

---

*Last updated: 2025-12-05*
*Status: Planning phase - ready for review*
