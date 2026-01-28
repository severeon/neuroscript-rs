use super::fixtures::*;
use crate::interfaces::*;

#[test]
fn test_simple_cycle() {
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("A", wildcard(), wildcard())
        .with_simple_neuron("B", wildcard(), wildcard())
        .with_composite(
            "Composite",
            vec![
                connection(call_endpoint("A"), call_endpoint("B")),
                connection(call_endpoint("B"), call_endpoint("A")),
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
                connection(call_endpoint("A"), call_endpoint("Fork")),
                connection(
                    call_endpoint("Fork"),
                    tuple_endpoint(vec!["main", "skip"]),
                ),
                connection(ref_endpoint("main"), call_endpoint("A")),
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
