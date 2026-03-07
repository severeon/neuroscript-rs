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

#[test]
fn test_sanitize_python_ident_defense_in_depth() {
    // Normal identifiers pass through unchanged
    assert_eq!(sanitize_python_ident("dim"), "dim");
    assert_eq!(sanitize_python_ident("Linear"), "Linear");
    assert_eq!(sanitize_python_ident("my_var"), "my_var");

    // Python keywords get trailing underscore
    assert_eq!(sanitize_python_ident("class"), "class_");
    assert_eq!(sanitize_python_ident("import"), "import_");
    assert_eq!(sanitize_python_ident("lambda"), "lambda_");

    // Invalid chars become underscores
    assert_eq!(sanitize_python_ident("a-b"), "a_b");
    assert_eq!(sanitize_python_ident("x.y"), "x_y");

    // Leading digit gets underscore prefix
    assert_eq!(sanitize_python_ident("3layer"), "_3layer");

    // Empty string
    assert_eq!(sanitize_python_ident(""), "_empty");
}

#[test]
fn test_value_to_python_sanitizes_names() {
    // Value::Name with a Python keyword should be sanitized
    assert_eq!(
        value_to_python_impl(&Value::Name("class".to_string())),
        "class_"
    );

    // Value::Global with special chars should be sanitized
    assert_eq!(
        value_to_python_impl(&Value::Global("my-global".to_string())),
        "my_global"
    );

    // Value::Call with kwargs keys should be sanitized
    let call = Value::Call {
        name: "Linear".to_string(),
        args: vec![],
        kwargs: vec![("class".to_string(), Value::Int(1))],
    };
    assert_eq!(value_to_python_impl(&call), "Linear(class_=1)");
}

#[test]
fn test_value_to_python_string_escaping() {
    // Simple string unchanged
    assert_eq!(
        value_to_python_impl(&Value::String("hello".to_string())),
        "\"hello\""
    );

    // Double quotes escaped
    assert_eq!(
        value_to_python_impl(&Value::String("say \"hi\"".to_string())),
        "\"say \\\"hi\\\"\""
    );

    // Backslashes escaped
    assert_eq!(
        value_to_python_impl(&Value::String("path\\to\\file".to_string())),
        "\"path\\\\to\\\\file\""
    );

    // Newlines escaped
    assert_eq!(
        value_to_python_impl(&Value::String("line1\nline2".to_string())),
        "\"line1\\nline2\""
    );

    // Tabs escaped
    assert_eq!(
        value_to_python_impl(&Value::String("col1\tcol2".to_string())),
        "\"col1\\tcol2\""
    );

    // Carriage return escaped
    assert_eq!(
        value_to_python_impl(&Value::String("cr\rhere".to_string())),
        "\"cr\\rhere\""
    );

    // Combined: quotes, backslash, newline
    assert_eq!(
        value_to_python_impl(&Value::String("a\"b\\c\n".to_string())),
        "\"a\\\"b\\\\c\\n\""
    );

    // Null byte uses \x00 (not \0) to avoid octal ambiguity
    assert_eq!(
        value_to_python_impl(&Value::String("a\0b".to_string())),
        "\"a\\x00b\""
    );

    // Null byte before digit — \0 + '1' would be octal \01 in Python
    assert_eq!(
        value_to_python_impl(&Value::String("\01".to_string())),
        "\"\\x001\""
    );
}
