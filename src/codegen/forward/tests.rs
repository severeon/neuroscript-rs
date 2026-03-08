use super::*;

#[test]
fn test_shape_check_literals() {
    let program = Program::new();
    let ctx = InferenceContext::new();
    let gen = CodeGenerator::new(&program, ctx);

    let shape = Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]);

    let result = generate_shape_check(&gen, &shape, None, "x");
    assert_eq!(result.condition, "x.ndim == 2 and x.shape[1] == 512");
    assert!(result.bindings.is_empty());
    assert!(result.guard_condition.is_none());
}

#[test]
fn test_shape_check_with_capture() {
    let program = Program::new();
    let ctx = InferenceContext::new();
    let gen = CodeGenerator::new(&program, ctx);

    let shape = Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]);

    let result = generate_shape_check(&gen, &shape, None, "x");
    assert_eq!(result.condition, "x.ndim == 2");
    assert_eq!(result.bindings, vec!["d = x.shape[1]"]);
    assert!(result.guard_condition.is_none());
}

#[test]
fn test_shape_check_with_guard() {
    let program = Program::new();
    let ctx = InferenceContext::new();
    let gen = CodeGenerator::new(&program, ctx);

    let shape = Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]);

    let guard = Value::BinOp {
        op: BinOp::Gt,
        left: Box::new(Value::Name("d".to_string())),
        right: Box::new(Value::Int(512)),
    };

    let result = generate_shape_check(&gen, &shape, Some(&guard), "x");
    assert_eq!(result.condition, "x.ndim == 2");
    assert_eq!(result.bindings, vec!["d = x.shape[1]"]);
    assert_eq!(result.guard_condition, Some("d > 512".to_string()));
}

#[test]
fn test_shape_check_with_logical_guard() {
    let program = Program::new();
    let ctx = InferenceContext::new();
    let gen = CodeGenerator::new(&program, ctx);

    let shape = Shape::new(vec![Dim::Wildcard, Dim::Wildcard, Dim::Named("d".to_string())]);

    // Guard: d >= 128 && d < 512
    let guard = Value::BinOp {
        op: BinOp::And,
        left: Box::new(Value::BinOp {
            op: BinOp::Ge,
            left: Box::new(Value::Name("d".to_string())),
            right: Box::new(Value::Int(128)),
        }),
        right: Box::new(Value::BinOp {
            op: BinOp::Lt,
            left: Box::new(Value::Name("d".to_string())),
            right: Box::new(Value::Int(512)),
        }),
    };

    let result = generate_shape_check(&gen, &shape, Some(&guard), "x");
    assert_eq!(result.condition, "x.ndim == 3");
    assert_eq!(result.bindings, vec!["d = x.shape[2]"]);
    assert_eq!(
        result.guard_condition,
        Some("d >= 128 and d < 512".to_string())
    );
}

#[test]
fn test_shape_check_with_or_guard() {
    let program = Program::new();
    let ctx = InferenceContext::new();
    let gen = CodeGenerator::new(&program, ctx);

    let shape = Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]);

    // Guard: d == 256 || d == 512
    let guard = Value::BinOp {
        op: BinOp::Or,
        left: Box::new(Value::BinOp {
            op: BinOp::Eq,
            left: Box::new(Value::Name("d".to_string())),
            right: Box::new(Value::Int(256)),
        }),
        right: Box::new(Value::BinOp {
            op: BinOp::Eq,
            left: Box::new(Value::Name("d".to_string())),
            right: Box::new(Value::Int(512)),
        }),
    };

    let result = generate_shape_check(&gen, &shape, Some(&guard), "x");
    assert_eq!(result.condition, "x.ndim == 2");
    assert_eq!(result.bindings, vec!["d = x.shape[1]"]);
    assert_eq!(
        result.guard_condition,
        Some("d == 256 or d == 512".to_string())
    );
}

#[test]
fn test_shape_check_with_compound_logical_guard() {
    let program = Program::new();
    let ctx = InferenceContext::new();
    let gen = CodeGenerator::new(&program, ctx);

    let shape = Shape::new(vec![
        Dim::Wildcard,
        Dim::Named("s".to_string()),
        Dim::Named("d".to_string()),
    ]);

    // Guard: s > 1 && d >= 64 && d <= 2048
    let guard = Value::BinOp {
        op: BinOp::And,
        left: Box::new(Value::BinOp {
            op: BinOp::And,
            left: Box::new(Value::BinOp {
                op: BinOp::Gt,
                left: Box::new(Value::Name("s".to_string())),
                right: Box::new(Value::Int(1)),
            }),
            right: Box::new(Value::BinOp {
                op: BinOp::Ge,
                left: Box::new(Value::Name("d".to_string())),
                right: Box::new(Value::Int(64)),
            }),
        }),
        right: Box::new(Value::BinOp {
            op: BinOp::Le,
            left: Box::new(Value::Name("d".to_string())),
            right: Box::new(Value::Int(2048)),
        }),
    };

    let result = generate_shape_check(&gen, &shape, Some(&guard), "x");
    assert_eq!(result.condition, "x.ndim == 3");
    assert_eq!(result.bindings, vec!["s = x.shape[1]", "d = x.shape[2]"]);
    assert_eq!(
        result.guard_condition,
        Some("s > 1 and d >= 64 and d <= 2048".to_string())
    );
}
