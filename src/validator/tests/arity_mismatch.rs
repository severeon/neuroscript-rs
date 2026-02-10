use super::fixtures::*;
use crate::interfaces::*;

#[test]
fn test_arity_mismatch_call_to_call() {
    let mut program = ProgramBuilder::new()
        .with_multi_port_neuron(
            "TwoOut",
            vec![default_port(wildcard())],
            vec![port("a", wildcard()), port("b", wildcard())],
        )
        .with_simple_neuron("OneIn", wildcard(), wildcard())
        .with_composite(
            "Composite",
            vec![connection(call_endpoint("TwoOut"), call_endpoint("OneIn"))],
            Some(10),
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(
            e,
            ValidationError::ArityMismatch {
                expected: 1,
                got: 2,
                ..
            }
        )
    });
}

#[test]
fn test_implicit_fork_single_to_two() {
    // Single-output source -> (a, b) should now be valid (implicit fork)
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("OneOut", wildcard(), wildcard())
        .with_composite(
            "Composite",
            vec![connection(
                call_endpoint("OneOut"),
                tuple_endpoint(vec!["a", "b"]),
            )],
            Some(10),
        )
        .build();

    assert_validation_ok(&mut program);
}

#[test]
fn test_implicit_fork_single_to_many() {
    // Single-output source -> (a, b, c, d, e) should be valid (implicit fork)
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("OneOut", wildcard(), wildcard())
        .with_composite(
            "Composite",
            vec![connection(
                call_endpoint("OneOut"),
                tuple_endpoint(vec!["a", "b", "c", "d", "e"]),
            )],
            Some(10),
        )
        .build();

    assert_validation_ok(&mut program);
}

#[test]
fn test_arity_mismatch_multi_to_wrong_count() {
    // Two-output source -> (a, b, c) should still error (2 != 3)
    let mut program = ProgramBuilder::new()
        .with_multi_port_neuron(
            "TwoOut",
            vec![default_port(wildcard())],
            vec![port("a", wildcard()), port("b", wildcard())],
        )
        .with_composite(
            "Composite",
            vec![connection(
                call_endpoint("TwoOut"),
                tuple_endpoint(vec!["x", "y", "z"]),
            )],
            Some(10),
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(
            e,
            ValidationError::ArityMismatch {
                expected: 3,
                got: 2,
                ..
            }
        )
    });
}

#[test]
fn test_explicit_fork_still_works() {
    // Fork -> (a, b) with Fork defined as 2-output should still work
    let mut program = ProgramBuilder::new()
        .with_multi_port_neuron(
            "Fork",
            vec![default_port(wildcard())],
            vec![port("a", wildcard()), port("b", wildcard())],
        )
        .with_composite(
            "Composite",
            vec![
                connection(ref_endpoint("in"), call_endpoint("Fork")),
                connection(call_endpoint("Fork"), tuple_endpoint(vec!["a", "b"])),
            ],
            Some(10),
        )
        .build();

    assert_validation_ok(&mut program);
}

#[test]
fn test_single_element_tuple_unpacking() {
    // Single-output source -> (a) is a 1:1 match, should be valid (standard path, not implicit fork)
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("OneOut", wildcard(), wildcard())
        .with_composite(
            "Composite",
            vec![connection(
                call_endpoint("OneOut"),
                tuple_endpoint(vec!["a"]),
            )],
            Some(10),
        )
        .build();

    assert_validation_ok(&mut program);
}

#[test]
fn test_arity_mismatch_tuple_to_call() {
    let mut program = ProgramBuilder::new()
        .with_multi_port_neuron(
            "TwoIn",
            vec![port("left", wildcard()), port("right", wildcard())],
            vec![default_port(wildcard())],
        )
        .with_multi_port_neuron(
            "Fork",
            vec![default_port(wildcard())],
            vec![port("a", wildcard()), port("b", wildcard())],
        )
        .with_composite(
            "Composite",
            vec![
                connection(ref_endpoint("in"), call_endpoint("Fork")),
                connection(call_endpoint("Fork"), tuple_endpoint(vec!["a", "b"])),
                connection(tuple_endpoint(vec!["a"]), call_endpoint("TwoIn")),
            ],
            Some(10),
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(
            e,
            ValidationError::ArityMismatch {
                expected: 2,
                got: 1,
                ..
            }
        )
    });
}
