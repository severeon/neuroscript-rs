use super::*;

#[test]
fn test_value_to_python_primitives() {
    assert_eq!(value_to_python_impl(&Value::Int(42)), "42");
    assert_eq!(value_to_python_impl(&Value::Float(4.5)), "4.5");
    // Integer-valued floats must preserve the decimal point in Python output
    assert_eq!(value_to_python_impl(&Value::Float(1.0)), "1.0");
    assert_eq!(
        value_to_python_impl(&Value::String("hello".to_string())),
        "\"hello\""
    );
    assert_eq!(value_to_python_impl(&Value::Bool(true)), "True");
    assert_eq!(value_to_python_impl(&Value::Bool(false)), "False");
    assert_eq!(value_to_python_impl(&Value::Name("dim".to_string())), "dim");
}

#[test]
fn test_value_to_python_binop() {
    let binop = Value::BinOp {
        op: BinOp::Mul,
        left: Box::new(Value::Name("dim".to_string())),
        right: Box::new(Value::Int(4)),
    };
    assert_eq!(value_to_python_impl(&binop), "dim * 4");
}

#[test]
fn test_snake_case() {
    assert_eq!(snake_case_impl("Linear"), "linear");
    assert_eq!(snake_case_impl("GELU"), "g_e_l_u");
    assert_eq!(snake_case_impl("LayerNorm"), "layer_norm");
    assert_eq!(
        snake_case_impl("MultiHeadAttention"),
        "multi_head_attention"
    );
}

#[test]
fn test_has_captured_dimensions() {
    let mut params = HashSet::new();
    params.insert("dim".to_string());

    // Parameter reference - not captured
    assert!(!has_captured_dimensions_impl(
        &Value::Name("dim".to_string()),
        &params
    ));

    // Non-parameter reference - captured
    assert!(has_captured_dimensions_impl(
        &Value::Name("d".to_string()),
        &params
    ));

    // BinOp with captured
    let binop = Value::BinOp {
        op: BinOp::Mul,
        left: Box::new(Value::Name("d".to_string())),
        right: Box::new(Value::Int(4)),
    };
    assert!(has_captured_dimensions_impl(&binop, &params));
}

#[test]
fn test_endpoint_key_unique_per_call() {
    let call1 = Endpoint::Call {
        name: "Linear".to_string(),
        args: vec![Value::Int(512), Value::Int(256)],
        kwargs: vec![],
        id: 0,
        frozen: false,
    };

    let call2 = Endpoint::Call {
        name: "Linear".to_string(),
        args: vec![Value::Int(512), Value::Int(256)],
        kwargs: vec![],
        id: 1,
        frozen: false,
    };

    // Different ids should produce different keys (each call gets its own module)
    assert_ne!(endpoint_key_impl(&call1), endpoint_key_impl(&call2));

    // Same id should produce same key
    let call3 = Endpoint::Call {
        name: "Linear".to_string(),
        args: vec![Value::Int(512), Value::Int(256)],
        kwargs: vec![],
        id: 0,
        frozen: false,
    };
    assert_eq!(endpoint_key_impl(&call1), endpoint_key_impl(&call3));
}
