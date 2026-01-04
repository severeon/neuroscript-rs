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
