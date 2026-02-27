use super::fixtures::*;
use crate::interfaces::*;

// ========== Tests using ProgramBuilder (IR-level) ==========

// Helper to build a composite neuron with proper wildcard shapes that
// won't conflict with reshape output shapes during port compatibility checks
fn build_reshape_program(
    name: &str,
    reshape_ep: Endpoint,
    extra_neurons: Vec<(&str, Shape, Shape)>,
) -> Program {
    // Create a second reshape endpoint with a distinct id for the source connection,
    // so each connection has a unique endpoint (matching real parsed programs).
    let reshape_ep_source = match &reshape_ep {
        Endpoint::Reshape(r) => Endpoint::Reshape(ReshapeExpr {
            dims: r.dims.clone(),
            annotation: r.annotation.clone(),
            id: r.id + 1000, // Offset to avoid id collision
        }),
        other => other.clone(),
    };
    let mut builder = ProgramBuilder::new();
    for (n, in_shape, out_shape) in extra_neurons {
        builder = builder.with_simple_neuron(n, in_shape, out_shape);
    }
    builder
        .with_composite_ports(
            name,
            vec![default_port(shape_two_wildcard())],
            vec![default_port(shape_two_wildcard())],
            vec![
                connection(ref_endpoint("in"), reshape_ep),
                connection(reshape_ep_source, ref_endpoint("out")),
            ],
            Some(10),
        )
        .build()
}

#[test]
fn test_validate_reshape_basic() {
    // A simple reshape endpoint in a composite neuron should pass validation
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Named("batch".to_string()),
            ReshapeDim::Named("dim".to_string()),
        ],
        annotation: None,
        id: 100,
    });

    let mut program = build_reshape_program("ReshapeTest", reshape_ep, vec![]);
    assert_validation_ok(&mut program);
}

#[test]
fn test_validate_reshape_with_reduce_mean() {
    // Reshape with @reduce(mean) annotation should pass
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Named("batch".to_string()),
            ReshapeDim::Named("dim".to_string()),
        ],
        annotation: Some(TransformAnnotation::Reduce(TransformStrategy::Intrinsic(
            "mean".to_string(),
        ))),
        id: 101,
    });

    let mut program = build_reshape_program("ReduceTest", reshape_ep, vec![]);
    assert_validation_ok(&mut program);
}

#[test]
fn test_validate_reshape_with_reduce_sum() {
    // Reshape with @reduce(sum) annotation should pass
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Named("batch".to_string()),
            ReshapeDim::Named("dim".to_string()),
        ],
        annotation: Some(TransformAnnotation::Reduce(TransformStrategy::Intrinsic(
            "sum".to_string(),
        ))),
        id: 102,
    });

    let mut program = build_reshape_program("ReduceSumTest", reshape_ep, vec![]);
    assert_validation_ok(&mut program);
}

#[test]
fn test_validate_reshape_with_repeat_copy() {
    // Reshape with @repeat(copy) annotation should pass
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Named("batch".to_string()),
            ReshapeDim::Named("dim".to_string()),
        ],
        annotation: Some(TransformAnnotation::Repeat(TransformStrategy::Intrinsic(
            "copy".to_string(),
        ))),
        id: 103,
    });

    let mut program = build_reshape_program("RepeatTest", reshape_ep, vec![]);
    assert_validation_ok(&mut program);
}

#[test]
fn test_validate_reshape_invalid_reduce_intrinsic() {
    // Reshape with an invalid @reduce intrinsic should fail
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Named("batch".to_string()),
            ReshapeDim::Named("dim".to_string()),
        ],
        annotation: Some(TransformAnnotation::Reduce(TransformStrategy::Intrinsic(
            "invalid_op".to_string(),
        ))),
        id: 104,
    });

    let mut program = build_reshape_program("InvalidReduceTest", reshape_ep, vec![]);
    assert_validation_error(&mut program, |e| {
        matches!(
            e,
            ValidationError::InvalidAnnotation {
                annotation,
                reason,
                ..
            } if annotation.contains("@reduce") && reason.contains("invalid_op")
        )
    });
}

#[test]
fn test_validate_reshape_invalid_repeat_intrinsic() {
    // Reshape with an invalid @repeat intrinsic should fail
    // mean is valid for reduce but not repeat
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Named("batch".to_string()),
            ReshapeDim::Named("dim".to_string()),
        ],
        annotation: Some(TransformAnnotation::Repeat(TransformStrategy::Intrinsic(
            "mean".to_string(),
        ))),
        id: 105,
    });

    let mut program = build_reshape_program("InvalidRepeatTest", reshape_ep, vec![]);
    assert_validation_error(&mut program, |e| {
        matches!(
            e,
            ValidationError::InvalidAnnotation {
                annotation,
                reason,
                ..
            } if annotation.contains("@repeat") && reason.contains("mean")
        )
    });
}

#[test]
fn test_validate_reshape_with_neuron_annotation_existing() {
    // Reshape with a neuron annotation referencing an existing neuron should pass
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Named("batch".to_string()),
            ReshapeDim::Named("dim".to_string()),
        ],
        annotation: Some(TransformAnnotation::Reduce(TransformStrategy::Neuron {
            name: "PoolNeuron".to_string(),
            args: vec![],
            kwargs: vec![],
        })),
        id: 106,
    });

    let mut program = build_reshape_program(
        "NeuronAnnotationTest",
        reshape_ep,
        vec![("PoolNeuron", wildcard(), wildcard())],
    );
    assert_validation_ok(&mut program);
}

#[test]
fn test_validate_reshape_with_neuron_annotation_missing() {
    // Reshape with a neuron annotation referencing a missing neuron should fail
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Named("batch".to_string()),
            ReshapeDim::Named("dim".to_string()),
        ],
        annotation: Some(TransformAnnotation::Reduce(TransformStrategy::Neuron {
            name: "MissingPoolNeuron".to_string(),
            args: vec![],
            kwargs: vec![],
        })),
        id: 107,
    });

    let mut program = build_reshape_program("MissingNeuronAnnotationTest", reshape_ep, vec![]);
    assert_validation_error(&mut program, |e| {
        matches!(
            e,
            ValidationError::MissingNeuron { name, .. } if name == "MissingPoolNeuron"
        )
    });
}

#[test]
fn test_validate_reshape_with_others_dim() {
    // Reshape with Others (wildcard) dim should pass
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Others,
            ReshapeDim::Named("dim".to_string()),
        ],
        annotation: None,
        id: 108,
    });

    let mut program = build_reshape_program("OthersTest", reshape_ep, vec![]);
    assert_validation_ok(&mut program);
}

#[test]
fn test_validate_reshape_with_literal_dim() {
    // Reshape with literal dimensions should pass
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Named("batch".to_string()),
            ReshapeDim::Literal(64),
        ],
        annotation: None,
        id: 109,
    });

    let mut program = build_reshape_program("LiteralDimTest", reshape_ep, vec![]);
    assert_validation_ok(&mut program);
}

// ========== Tests using source parsing ==========

#[test]
fn test_validate_fat_arrow_parsed_basic() {
    let source = r#"
neuron ReshapeTest(dim, heads):
  in: [batch, seq, dim]
  out: [batch, heads, seq, dim]
  graph:
    in => [batch, seq, heads, dh] -> out
"#;
    let mut program = crate::parse(source).unwrap();
    let result = crate::validate(&mut program);
    assert!(
        result.is_ok(),
        "basic fat arrow should validate: {:?}",
        result
    );
}

#[test]
fn test_validate_fat_arrow_parsed_chained() {
    let source = r#"
neuron ReshapeChain(dim, heads):
  in: [batch, seq, dim]
  out: [batch, heads, seq, dh]
  graph:
    in => [batch, seq, heads, dh] => [batch, heads, seq, dh] -> out
"#;
    let mut program = crate::parse(source).unwrap();
    let result = crate::validate(&mut program);
    assert!(
        result.is_ok(),
        "chained fat arrow should validate: {:?}",
        result
    );
}

#[test]
fn test_validate_fat_arrow_parsed_with_reduce() {
    let source = r#"
neuron Pool(dim):
  in: [batch, seq, dim]
  out: [batch, dim]
  graph:
    in => @reduce(mean) [batch, dim] -> out
"#;
    let mut program = crate::parse(source).unwrap();
    let result = crate::validate(&mut program);
    assert!(
        result.is_ok(),
        "fat arrow with @reduce(mean) should validate: {:?}",
        result
    );
}

#[test]
fn test_validate_fat_arrow_all_reduce_intrinsics() {
    // Test all valid reduce intrinsics
    for intrinsic in &["mean", "sum", "min", "max", "prod", "logsumexp"] {
        let source = format!(
            r#"
neuron Pool(dim):
  in: [batch, seq, dim]
  out: [batch, dim]
  graph:
    in => @reduce({}) [batch, dim] -> out
"#,
            intrinsic
        );
        let mut program = crate::parse(&source).unwrap();
        let result = crate::validate(&mut program);
        assert!(
            result.is_ok(),
            "@reduce({}) should validate: {:?}",
            intrinsic,
            result
        );
    }
}

#[test]
fn test_validate_reshape_empty_dims() {
    // Reshape with empty dims (=> []) should be rejected
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![],
        annotation: None,
        id: 200,
    });

    let mut program = build_reshape_program("EmptyReshapeTest", reshape_ep, vec![]);
    assert_validation_error(&mut program, |e| {
        matches!(
            e,
            ValidationError::InvalidReshape {
                message,
                ..
            } if message.contains("at least one dimension")
        )
    });
}
