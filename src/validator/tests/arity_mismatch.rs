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

#[test]
fn test_variadic_port_accepts_two_inputs() {
    // (a, b) -> VariadicNeuron() should be valid when VariadicNeuron has variadic input
    let mut program = ProgramBuilder::new()
        .with_multi_port_neuron(
            "VariadicNeuron",
            vec![variadic_port("inputs", wildcard())],
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
                connection(tuple_endpoint(vec!["a", "b"]), call_endpoint("VariadicNeuron")),
                connection(call_endpoint("VariadicNeuron"), ref_endpoint("out")),
            ],
            Some(10),
        )
        .build();

    assert_validation_ok(&mut program);
}

#[test]
fn test_variadic_port_accepts_three_inputs() {
    // (a, b, c) -> VariadicNeuron() should be valid
    let mut program = ProgramBuilder::new()
        .with_multi_port_neuron(
            "VariadicNeuron",
            vec![variadic_port("inputs", wildcard())],
            vec![default_port(wildcard())],
        )
        .with_multi_port_neuron(
            "Fork3",
            vec![default_port(wildcard())],
            vec![port("a", wildcard()), port("b", wildcard()), port("c", wildcard())],
        )
        .with_composite(
            "Composite",
            vec![
                connection(ref_endpoint("in"), call_endpoint("Fork3")),
                connection(call_endpoint("Fork3"), tuple_endpoint(vec!["a", "b", "c"])),
                connection(
                    tuple_endpoint(vec!["a", "b", "c"]),
                    call_endpoint("VariadicNeuron"),
                ),
                connection(call_endpoint("VariadicNeuron"), ref_endpoint("out")),
            ],
            Some(10),
        )
        .build();

    assert_validation_ok(&mut program);
}

#[test]
fn test_non_variadic_still_enforces_arity() {
    // (a, b, c) -> NonVariadic() with 2 inputs should still fail
    let mut program = ProgramBuilder::new()
        .with_multi_port_neuron(
            "TwoIn",
            vec![port("left", wildcard()), port("right", wildcard())],
            vec![default_port(wildcard())],
        )
        .with_multi_port_neuron(
            "Fork3",
            vec![default_port(wildcard())],
            vec![port("a", wildcard()), port("b", wildcard()), port("c", wildcard())],
        )
        .with_composite(
            "Composite",
            vec![
                connection(ref_endpoint("in"), call_endpoint("Fork3")),
                connection(call_endpoint("Fork3"), tuple_endpoint(vec!["a", "b", "c"])),
                connection(
                    tuple_endpoint(vec!["a", "b", "c"]),
                    call_endpoint("TwoIn"),
                ),
                connection(call_endpoint("TwoIn"), ref_endpoint("out")),
            ],
            Some(10),
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(
            e,
            ValidationError::ArityMismatch {
                expected: 2,
                got: 3,
                ..
            }
        )
    });
}

#[test]
fn test_variadic_port_accepts_single_input() {
    // a -> VariadicNeuron() should also be valid (single input to variadic)
    let mut program = ProgramBuilder::new()
        .with_multi_port_neuron(
            "VariadicNeuron",
            vec![variadic_port("inputs", wildcard())],
            vec![default_port(wildcard())],
        )
        .with_simple_neuron("OneOut", wildcard(), wildcard())
        .with_composite(
            "Composite",
            vec![
                connection(ref_endpoint("in"), call_endpoint("OneOut")),
                connection(call_endpoint("OneOut"), call_endpoint("VariadicNeuron")),
                connection(call_endpoint("VariadicNeuron"), ref_endpoint("out")),
            ],
            Some(10),
        )
        .build();

    assert_validation_ok(&mut program);
}

#[test]
fn test_variadic_output_port_rejected() {
    // Variadic output ports are not supported
    let mut program = ProgramBuilder::new()
        .with_multi_port_neuron(
            "BadNeuron",
            vec![default_port(wildcard())],
            vec![variadic_port("outputs", wildcard())],
        )
        .with_composite(
            "Composite",
            vec![
                connection(ref_endpoint("in"), call_endpoint("BadNeuron")),
                connection(call_endpoint("BadNeuron"), ref_endpoint("out")),
            ],
            Some(10),
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(e, ValidationError::Custom(msg) if msg.contains("Variadic output ports"))
    });
}

#[test]
fn test_variadic_port_requires_explicit_name() {
    // A variadic port named "default" should be rejected
    let mut program = ProgramBuilder::new()
        .with_multi_port_neuron(
            "BadVariadic",
            vec![variadic_port("default", wildcard())],
            vec![default_port(wildcard())],
        )
        .with_composite(
            "Composite",
            vec![
                connection(ref_endpoint("in"), call_endpoint("BadVariadic")),
                connection(call_endpoint("BadVariadic"), ref_endpoint("out")),
            ],
            Some(10),
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(e, ValidationError::Custom(msg) if msg.contains("explicit name"))
    });
}
