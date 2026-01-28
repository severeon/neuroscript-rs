use super::fixtures::*;
use crate::interfaces::*;

#[test]
fn test_shape_mismatch_literal() {
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("Out512", wildcard(), shape_512())
        .with_simple_neuron("In256", shape_256(), wildcard())
        .with_composite(
            "Composite",
            vec![connection(call_endpoint("Out512"), call_endpoint("In256"))],
            Some(10),
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(e, ValidationError::PortMismatch { .. })
    });
}

#[test]
fn test_shape_mismatch_multi_dim() {
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("Out512", wildcard(), shape_batch_512())
        .with_simple_neuron("In256", shape_batch_256(), wildcard())
        .with_composite(
            "Composite",
            vec![connection(call_endpoint("Out512"), call_endpoint("In256"))],
            Some(10),
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(e, ValidationError::PortMismatch { .. })
    });
}

#[test]
fn test_shape_match_exact() {
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("Out512", wildcard(), shape_512())
        .with_simple_neuron("In512", shape_512(), wildcard())
        .with_composite(
            "Composite",
            vec![connection(call_endpoint("Out512"), call_endpoint("In512"))],
            Some(10),
        )
        .build();

    assert_validation_ok(&mut program);
}
