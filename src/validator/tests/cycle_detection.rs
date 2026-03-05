use super::fixtures::*;
use crate::interfaces::*;

#[test]
fn test_simple_cycle() {
    // Cycles in NeuroScript occur through named Ref endpoints (context bindings).
    // Two named nodes feeding into each other: a -> b -> a
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("A", wildcard(), wildcard())
        .with_simple_neuron("B", wildcard(), wildcard())
        .with_composite(
            "Composite",
            vec![
                connection(ref_endpoint("a"), ref_endpoint("b")),
                connection(ref_endpoint("b"), ref_endpoint("a")),
            ],
            None,
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(e, ValidationError::CycleDetected { .. })
    });
}

#[test]
fn test_cycle_through_unpacked_ports() {
    let mut program = ProgramBuilder::new()
        .with_multi_port_neuron(
            "Fork",
            vec![default_port(wildcard())],
            vec![port("a", wildcard()), port("b", wildcard())],
        )
        .with_simple_neuron("A", wildcard(), wildcard())
        .with_composite(
            "Composite",
            vec![
                connection(ref_endpoint("a"), call_endpoint("Fork")),
                connection(
                    call_endpoint("Fork"),
                    tuple_endpoint(vec!["main", "skip"]),
                ),
                connection(ref_endpoint("main"), ref_endpoint("a")),
            ],
            None,
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(e, ValidationError::CycleDetected { .. })
    });
}

#[test]
fn test_no_cycle_valid_residual() {
    let mut program = ProgramBuilder::new()
        .with_multi_port_neuron(
            "Fork",
            vec![default_port(wildcard())],
            vec![port("a", wildcard()), port("b", wildcard())],
        )
        .with_multi_port_neuron(
            "Add",
            vec![port("left", wildcard()), port("right", wildcard())],
            vec![default_port(wildcard())],
        )
        .with_simple_neuron("Process", wildcard(), wildcard())
        .with_composite(
            "Residual",
            vec![
                connection(ref_endpoint("in"), call_endpoint("Fork")),
                connection(
                    call_endpoint("Fork"),
                    tuple_endpoint(vec!["main", "skip"]),
                ),
                connection(ref_endpoint("main"), call_endpoint("Process")),
                connection(call_endpoint("Process"), ref_endpoint("processed")),
                connection(
                    tuple_endpoint(vec!["processed", "skip"]),
                    call_endpoint("Add"),
                ),
                connection(call_endpoint("Add"), ref_endpoint("out")),
            ],
            Some(10),
        )
        .build();

    assert_validation_ok(&mut program);
}

#[test]
fn test_two_independent_cycles_both_reported() {
    // Two independent cycles through named refs: a->b->a and c->d->c
    // Both should be reported (not just the first one)
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("A", wildcard(), wildcard())
        .with_simple_neuron("B", wildcard(), wildcard())
        .with_simple_neuron("C", wildcard(), wildcard())
        .with_simple_neuron("D", wildcard(), wildcard())
        .with_composite(
            "Composite",
            vec![
                // Cycle 1: a -> b -> a
                connection(ref_endpoint("a"), ref_endpoint("b")),
                connection(ref_endpoint("b"), ref_endpoint("a")),
                // Cycle 2: c -> d -> c
                connection(ref_endpoint("c"), ref_endpoint("d")),
                connection(ref_endpoint("d"), ref_endpoint("c")),
            ],
            None,
        )
        .build();

    let result = crate::validator::Validator::validate(&mut program);
    assert!(result.is_err(), "Expected validation errors");
    let errors = result.unwrap_err();
    let cycle_count = errors
        .iter()
        .filter(|e| matches!(e, ValidationError::CycleDetected { .. }))
        .count();
    assert!(
        cycle_count >= 2,
        "Expected at least 2 cycle errors, got {}. Errors: {:?}",
        cycle_count,
        errors
    );
}

#[test]
fn test_no_false_positive_when_non_cyclic_node_feeds_into_cycle() {
    // Graph: d->b, b->c->b (cycle). D feeds into the cycle but is not part of it.
    // Should report exactly 1 cycle (b->c->b), NOT a false positive for d
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("A", wildcard(), wildcard())
        .with_simple_neuron("B", wildcard(), wildcard())
        .with_simple_neuron("C", wildcard(), wildcard())
        .with_simple_neuron("D", wildcard(), wildcard())
        .with_composite(
            "Composite",
            vec![
                connection(ref_endpoint("b"), ref_endpoint("c")),
                connection(ref_endpoint("c"), ref_endpoint("b")),
                // d feeds into b but is NOT part of the cycle
                connection(ref_endpoint("d"), ref_endpoint("b")),
            ],
            None,
        )
        .build();

    let result = crate::validator::Validator::validate(&mut program);
    assert!(result.is_err(), "Expected validation error for b->c->b cycle");
    let errors = result.unwrap_err();
    let cycle_errors: Vec<_> = errors
        .iter()
        .filter(|e| matches!(e, ValidationError::CycleDetected { .. }))
        .collect();

    // Should have exactly 1 cycle error, not a false positive involving d
    assert_eq!(
        cycle_errors.len(),
        1,
        "Expected exactly 1 cycle error, got {}. Errors: {:?}",
        cycle_errors.len(),
        cycle_errors
    );

    // Verify the cycle does not include d
    for error in &cycle_errors {
        if let ValidationError::CycleDetected { cycle, .. } = error {
            assert!(
                !cycle.iter().any(|n| n.contains("d")),
                "Cycle should not include d (false positive). Cycle: {:?}",
                cycle
            );
        }
    }
}
