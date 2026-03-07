use super::fixtures::*;
use crate::interfaces::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Monotonic counter for generating unique endpoint IDs in test helpers,
/// mirroring the real AST builder's `next_id()` approach.
static TEST_ID_COUNTER: AtomicUsize = AtomicUsize::new(10_000);

fn next_test_id() -> usize {
    TEST_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

// ========== Tests using ProgramBuilder (IR-level) ==========

fn shape_three_wildcard() -> Shape {
    Shape::new(vec![Dim::Wildcard, Dim::Wildcard, Dim::Wildcard])
}

fn shape_named_3d() -> Shape {
    Shape::new(vec![
        Dim::Named("batch".to_string()),
        Dim::Named("seq".to_string()),
        Dim::Named("dim".to_string()),
    ])
}

fn shape_named_2d() -> Shape {
    Shape::new(vec![
        Dim::Named("batch".to_string()),
        Dim::Named("dim".to_string()),
    ])
}

// Helper to build a composite neuron for @reduce tests: 3-dim input, 2-dim output
// so the reduce properly decreases rank. Uses named dims so unreachable-dim checks pass.
fn build_reduce_reshape_program(
    name: &str,
    reshape_ep: Endpoint,
    extra_neurons: Vec<(&str, Shape, Shape)>,
) -> Program {
    let reshape_ep_source = match &reshape_ep {
        Endpoint::Reshape(r) => Endpoint::Reshape(ReshapeExpr {
            dims: r.dims.clone(),
            annotation: r.annotation.clone(),
            id: next_test_id(),
            span: None,
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
            vec![default_port(shape_named_3d())],
            vec![default_port(shape_named_2d())],
            vec![
                connection(ref_endpoint("in"), reshape_ep),
                connection(reshape_ep_source, ref_endpoint("out")),
            ],
            Some(10),
        )
        .build()
}

// Helper to build a composite neuron with proper wildcard shapes that
// won't conflict with reshape output shapes during port compatibility checks
fn build_reshape_program(
    name: &str,
    reshape_ep: Endpoint,
    extra_neurons: Vec<(&str, Shape, Shape)>,
) -> Program {
    build_reshape_program_with_ports(
        name,
        reshape_ep,
        extra_neurons,
        vec![default_port(shape_two_wildcard())],
        vec![default_port(shape_two_wildcard())],
    )
}

// Helper that allows specifying custom input/output port shapes
fn build_reshape_program_with_ports(
    name: &str,
    reshape_ep: Endpoint,
    extra_neurons: Vec<(&str, Shape, Shape)>,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
) -> Program {
    // Create a second reshape endpoint with a distinct id for the source connection,
    // so each connection has a unique endpoint (matching real parsed programs).
    let reshape_ep_source = match &reshape_ep {
        Endpoint::Reshape(r) => Endpoint::Reshape(ReshapeExpr {
            dims: r.dims.clone(),
            annotation: r.annotation.clone(),
            id: next_test_id(),
            span: None,
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
            inputs,
            outputs,
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
        span: None,
    });

    let mut program = build_reshape_program("ReshapeTest", reshape_ep, vec![]);
    assert_validation_ok(&mut program);
}

#[test]
fn test_validate_reshape_with_reduce_mean() {
    // Reshape with @reduce(mean) annotation should pass when rank decreases (3 -> 2)
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Named("batch".to_string()),
            ReshapeDim::Named("dim".to_string()),
        ],
        annotation: Some(TransformAnnotation::Reduce(TransformStrategy::Intrinsic(
            "mean".to_string(),
        ), None)),
        id: 101,
        span: None,
    });

    let mut program = build_reduce_reshape_program("ReduceTest", reshape_ep, vec![]);
    assert_validation_ok(&mut program);
}

#[test]
fn test_validate_reshape_with_reduce_sum() {
    // Reshape with @reduce(sum) annotation should pass when rank decreases (3 -> 2)
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Named("batch".to_string()),
            ReshapeDim::Named("dim".to_string()),
        ],
        annotation: Some(TransformAnnotation::Reduce(TransformStrategy::Intrinsic(
            "sum".to_string(),
        ), None)),
        id: 102,
        span: None,
    });

    let mut program = build_reduce_reshape_program("ReduceSumTest", reshape_ep, vec![]);
    assert_validation_ok(&mut program);
}

#[test]
fn test_validate_reshape_reduce_must_decrease_rank() {
    // @reduce with same rank as source should fail (2 dims -> 2 dims)
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Named("batch".to_string()),
            ReshapeDim::Named("dim".to_string()),
        ],
        annotation: Some(TransformAnnotation::Reduce(TransformStrategy::Intrinsic(
            "mean".to_string(),
        ), None)),
        id: next_test_id(),
        span: None,
    });

    // build_reshape_program uses shape_two_wildcard() = [*, *] (2 dims) as input
    let mut program = build_reshape_program("ReduceNoRankDecrease", reshape_ep, vec![]);
    assert_validation_error(&mut program, |e| {
        matches!(
            e,
            ValidationError::InvalidAnnotation {
                reason,
                ..
            } if reason.contains("@reduce must decrease rank")
        )
    });
}

#[test]
fn test_validate_reshape_reduce_increasing_rank_fails() {
    // @reduce with more dims than source should fail (2 dims -> 3 dims)
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Named("batch".to_string()),
            ReshapeDim::Named("seq".to_string()),
            ReshapeDim::Named("dim".to_string()),
        ],
        annotation: Some(TransformAnnotation::Reduce(TransformStrategy::Intrinsic(
            "sum".to_string(),
        ), None)),
        id: next_test_id(),
        span: None,
    });

    // build_reshape_program uses shape_two_wildcard() = [*, *] (2 dims) as input
    let mut program = build_reshape_program("ReduceRankIncrease", reshape_ep, vec![]);
    assert_validation_error(&mut program, |e| {
        matches!(
            e,
            ValidationError::InvalidAnnotation {
                reason,
                ..
            } if reason.contains("@reduce must decrease rank")
        )
    });
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
        ), None)),
        id: 103,
        span: None,
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
        ), None)),
        id: 104,
        span: None,
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
        ), None)),
        id: 105,
        span: None,
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
    // Uses 3-dim input -> 2-dim target so @reduce properly decreases rank
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Named("batch".to_string()),
            ReshapeDim::Named("dim".to_string()),
        ],
        annotation: Some(TransformAnnotation::Reduce(TransformStrategy::Neuron {
            name: "PoolNeuron".to_string(),
            args: vec![],
            kwargs: vec![],
        }, None)),
        id: 106,
        span: None,
    });

    let mut program = build_reduce_reshape_program(
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
        }, None)),
        id: 107,
        span: None,
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
        span: None,
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
        span: None,
    });

    let mut program = build_reshape_program("LiteralDimTest", reshape_ep, vec![]);
    assert_validation_ok(&mut program);
}

// ========== Element-count preservation tests ==========

/// Build a composite neuron with literal-dimensioned ports so element-count
/// checks can fire (unlike wildcard shapes used by build_reshape_program).
fn build_literal_reshape_program(
    name: &str,
    in_shape: Shape,
    out_shape: Shape,
    reshape_ep: Endpoint,
) -> Program {
    let reshape_ep_source = match &reshape_ep {
        Endpoint::Reshape(r) => Endpoint::Reshape(ReshapeExpr {
            dims: r.dims.clone(),
            annotation: r.annotation.clone(),
            id: next_test_id(),
            span: None,
        }),
        other => other.clone(),
    };
    ProgramBuilder::new()
        .with_composite_ports(
            name,
            vec![default_port(in_shape)],
            vec![default_port(out_shape)],
            vec![
                connection(ref_endpoint("in"), reshape_ep.clone()),
                connection(reshape_ep_source, ref_endpoint("out")),
            ],
            Some(10),
        )
        .build()
}

#[test]
fn test_validate_reshape_element_count_mismatch() {
    // Source shape [2, 6] (12 elements) reshaped to [3, 5] (15 elements) should fail
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Literal(3),
            ReshapeDim::Literal(5),
        ],
        annotation: None,
        id: 300,
        span: None,
    });

    let in_shape = Shape::new(vec![Dim::Literal(2), Dim::Literal(6)]);
    let out_shape = Shape::new(vec![Dim::Literal(3), Dim::Literal(5)]);
    let mut program = build_literal_reshape_program(
        "ElementMismatchTest",
        in_shape,
        out_shape,
        reshape_ep,
    );
    assert_validation_error(&mut program, |e| {
        matches!(
            e,
            ValidationError::PortMismatch {
                context,
                ..
            } if context.contains("element count mismatch")
        )
    });
}

#[test]
fn test_validate_reshape_element_count_preserved() {
    // Source shape [2, 6] (12 elements) reshaped to [3, 4] (12 elements) should pass
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![
            ReshapeDim::Literal(3),
            ReshapeDim::Literal(4),
        ],
        annotation: None,
        id: 301,
        span: None,
    });

    let in_shape = Shape::new(vec![Dim::Literal(2), Dim::Literal(6)]);
    let out_shape = Shape::new(vec![Dim::Literal(3), Dim::Literal(4)]);
    let mut program = build_literal_reshape_program(
        "ElementPreservedTest",
        in_shape,
        out_shape,
        reshape_ep,
    );
    assert_validation_ok(&mut program);
}

#[test]
fn test_validate_reshape_annotated_skips_element_check() {
    // @reduce(mean) intentionally changes element count, so [2, 6] => [12] should pass
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![ReshapeDim::Literal(12)],
        annotation: Some(TransformAnnotation::Reduce(TransformStrategy::Intrinsic(
            "mean".to_string(),
        ), None)),
        id: 302,
        span: None,
    });

    let in_shape = Shape::new(vec![Dim::Literal(2), Dim::Literal(6)]);
    let out_shape = Shape::new(vec![Dim::Literal(12)]);
    let mut program = build_literal_reshape_program(
        "AnnotatedSkipTest",
        in_shape,
        out_shape,
        reshape_ep,
    );
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
fn test_validate_reduce_unreachable_dim() {
    // @reduce target with a dim not in input/output shapes or params should fail
    let source = r#"
neuron BadPool:
  in: [b, c, h, w]
  out: [b, c]
  graph:
    in => @reduce(mean) [b, c, x] -> out
"#;
    let mut program = crate::parse(source).unwrap();
    let result = crate::validate(&mut program);
    assert!(
        result.is_err(),
        "@reduce with unreachable dim 'x' should fail validation"
    );
    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| {
            matches!(
                e,
                ValidationError::InvalidAnnotation {
                    reason,
                    ..
                } if reason.contains("x") && reason.contains("not defined")
            )
        }),
        "expected error about unreachable dim 'x', got: {:?}",
        errors
    );
}

#[test]
fn test_validate_reduce_dim_from_param_is_ok() {
    // @reduce target dim that matches a param name should be allowed
    let source = r#"
neuron ParamPool(x):
  in: [b, c, h, w]
  out: [b, x]
  graph:
    in => @reduce(mean) [b, x] -> out
"#;
    let mut program = crate::parse(source).unwrap();
    let result = crate::validate(&mut program);
    assert!(
        result.is_ok(),
        "@reduce with dim from param should pass: {:?}",
        result
    );
}

#[test]
fn test_validate_reduce_binding_checks_rhs_not_lhs() {
    // Binding h=dim/heads: h is new, dim and heads must be known.
    // Using unknown dim "zz" in the RHS should fail.
    let source = r#"
neuron BadBinding:
  in: [b, c, h, w]
  out: [b, c]
  graph:
    in => @reduce(mean) [b, h=zz/c] -> out
"#;
    let mut program = crate::parse(source).unwrap();
    let result = crate::validate(&mut program);
    assert!(
        result.is_err(),
        "@reduce binding with unknown RHS dim 'zz' should fail"
    );
    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| {
            matches!(
                e,
                ValidationError::InvalidAnnotation { reason, .. }
                    if reason.contains("zz") && reason.contains("not defined")
            )
        }),
        "expected error about unreachable dim 'zz', got: {:?}",
        errors
    );
}

#[test]
fn test_validate_reduce_binding_with_known_rhs_is_ok() {
    // Binding h=dim/heads where dim and heads are known should pass.
    // h is a new name introduced by the binding — not checked against known_dims.
    let source = r#"
neuron GoodBinding(heads):
  in: [b, c, dim]
  out: [b, c]
  graph:
    in => @reduce(mean) [b, h=dim/heads] -> out
"#;
    let mut program = crate::parse(source).unwrap();
    let result = crate::validate(&mut program);
    assert!(
        result.is_ok(),
        "@reduce binding with known RHS dims should pass: {:?}",
        result
    );
}

#[test]
fn test_validate_reduce_expr_unknown_dim() {
    // Expression dim h*w where h is unknown should fail.
    let source = r#"
neuron BadExpr:
  in: [b, c, w]
  out: [b, c]
  graph:
    in => @reduce(mean) [b, unknown_dim*w] -> out
"#;
    let mut program = crate::parse(source).unwrap();
    let result = crate::validate(&mut program);
    assert!(
        result.is_err(),
        "@reduce expr with unknown dim should fail"
    );
    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| {
            matches!(
                e,
                ValidationError::InvalidAnnotation { reason, .. }
                    if reason.contains("unknown_dim") && reason.contains("not defined")
            )
        }),
        "expected error about unreachable dim 'unknown_dim', got: {:?}",
        errors
    );
}

#[test]
fn test_validate_reshape_empty_dims() {
    // Reshape with empty dims (=> []) should be rejected
    let reshape_ep = Endpoint::Reshape(ReshapeExpr {
        dims: vec![],
        annotation: None,
        id: 200,
        span: None,
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
