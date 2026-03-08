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

// ========================================
// Issue #119: 3D shapes vs [*, in_dim] wildcard patterns
// ========================================

#[test]
fn test_issue_119_3d_shape_into_wildcard_star_linear() {
    // Bug: A neuron outputting [batch, seq, dim] flowing into Linear
    // with input [*, in_dim] fails validation with PortMismatch.
    // The wildcard * should match multiple leading dimensions,
    // following PyTorch nn.Linear semantics.
    let shape_3d = Shape::new(vec![
        Dim::Named("batch".to_string()),
        Dim::Named("seq".to_string()),
        Dim::Named("dim".to_string()),
    ]);
    let star_in_dim = Shape::new(vec![Dim::Wildcard, Dim::Named("in_dim".to_string())]);
    let star_out_dim = Shape::new(vec![Dim::Wildcard, Dim::Named("out_dim".to_string())]);

    let mut program = ProgramBuilder::new()
        .with_simple_neuron("Projection", shape_3d.clone(), shape_3d)
        .with_simple_neuron("Linear", star_in_dim, star_out_dim)
        .with_composite(
            "Composite",
            vec![connection(
                call_endpoint("Projection"),
                call_endpoint("Linear"),
            )],
            Some(10),
        )
        .build();

    // This should pass: [batch, seq, dim] is compatible with [*, in_dim]
    assert_validation_ok(&program);
}
