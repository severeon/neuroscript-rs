## Your Task

1. Extend IR for arm metadata (src/interfaces.rs)
  - Add is_reachable: bool to MatchArm (default true).
  - Update Display/Clone/etc. for new field.
2. Mark unreachable arms in validator (src/validator/core.rs)
  - In validate_match_expression(), after detecting UnreachableMatchArm, set
arms[j].is_reachable = false.
  - Collect arms but don't error on shadowing (warn or optional).
3. Create optimizer pass (src/optimizer/mod.rs & core.rs (new module))
  - New pub fn optimize_matches(program: &mut Program).
  - For each composite neuron: traverse Graph, find Endpoint::Match, prune arms where
!arm.is_reachable.
  - Return PrunedArms count for logging.
4. Integrate optimizer before codegen (src/main.rs or src/codegen/mod.rs)
  - Call optimize_matches(&mut program) after validation, before generate_pytorch.
  - Log: "Pruned X dead arms from Y matches".
5. Update codegen for pruned arms (src/codegen/forward.rs:200-300 (approx))
  - In generate_shape_check()/if chains: skip arms where !arm.is_reachable.
  - Ensure else: raise only if no reachable catch-all.
6. Add CLI flag (src/main.rs)
- --optimize or --dead-elim to toggle (default on).
- Print summary: "Dead branch elimination: pruned X arms".
1. Unit tests (src/optimizer/mod.rs + src/codegen/)
  - Test shadowing: [*, d], [*, 512] → prunes 2nd.
  - Test guards: no prune (guards make reachable).
  - Test codegen: fewer if/elif.
  - Roundtrip: parse → validate → optimize → codegen → verify Python.
2. Integration tests (test_examples.sh + examples/)
  - Add match-heavy example (e.g., examples/match_demo.ns).
  - Assert pruned vs unpruned --codegen output diff.
3. Update MVP todo (mvp-todo.md)
  - Mark [x] Implement dead branch elimination.


# Dead Branch Elimination: Key Files & Lines

## 1. IR Changes
  - **src/interfaces.rs:129-134** - `MatchArm` struct: Add `is_reachable: bool = true;`

## 2. Validator Marking
  - **src/validator/core.rs:868-904** - `validate_match_expression()`: Set `arms[j].is_reachable =
false` on shadowing.
  - **src/validator/core.rs:930-993** - `pattern_subsumes()`: Reuse for marking (no changes
needed).

## 3. New Optimizer
  - **src/optimizer/mod.rs** (new module): `optimize_matches(program: &mut Program)` - Traverse graphs,
prune `!arm.is_reachable`.

## 4. Codegen Skip
  - **src/codegen/442-513** - `Endpoint::Match` in `process_destination()`: Skip `if
!arm.is_reachable { continue; }`.

## 5. Integration
  - **src/main.rs** (CLI): Add `--optimize` flag, call optimizer post-validate.
  - **src/codegen/mod.rs (generate_pytorch)** - `generate_pytorch()`: Run optimizer before emitting.

## Tests
  - **src/validator/core.rs tests** (~lines 996+): Add shadowing cases.

1. Extend IR for arm metadata (src/interfaces.rs)
  - Add is_reachable: bool to MatchArm (default true).
  - Update Display/Clone/etc. for new field.
  
<note>**src/interfaces.rs:129-134** - `MatchArm` struct: Add `is_reachable: bool = true;`</note>
