use super::*;

#[test]
fn test_module_instantiation_simple() {
    let program = Program::new();
    let ctx = InferenceContext::new();
    let mut gen = CodeGenerator::new(&program, ctx);

    // Simple connection with a Call
    let connections = vec![Connection {
        source: Endpoint::Ref(PortRef::new("in")),
        destination: Endpoint::Call {
            name: "Linear".to_string(),
            args: vec![Value::Int(512), Value::Int(256)],
            kwargs: vec![],
            id: 0,
        },
    }];

    let mut output = String::new();
    generate_module_instantiations(&mut gen, &mut output, &[], &[], &connections).unwrap();

    assert!(output.contains("self.linear_0 = Linear(512, 256)"));
}

#[test]
fn test_module_deduplication() {
    let program = Program::new();
    let ctx = InferenceContext::new();
    let mut gen = CodeGenerator::new(&program, ctx);

    // Two calls with same signature should deduplicate
    let connections = vec![
        Connection {
            source: Endpoint::Ref(PortRef::new("in")),
            destination: Endpoint::Call {
                name: "Linear".to_string(),
                args: vec![Value::Int(512), Value::Int(256)],
                kwargs: vec![],
                id: 0,
            },
        },
        Connection {
            source: Endpoint::Ref(PortRef::new("x")),
            destination: Endpoint::Call {
                name: "Linear".to_string(),
                args: vec![Value::Int(512), Value::Int(256)],
                kwargs: vec![],
                id: 0,
            },
        },
    ];

    let mut output = String::new();
    generate_module_instantiations(&mut gen, &mut output, &[], &[], &connections).unwrap();

    // Should only have one instantiation
    assert_eq!(output.matches("Linear(512, 256)").count(), 1);
}

#[test]
fn test_lazy_instantiation_marker() {
    let program = Program::new();
    let ctx = InferenceContext::new();
    let mut gen = CodeGenerator::new(&program, ctx);

    // Call with captured dimension
    let connections = vec![Connection {
        source: Endpoint::Ref(PortRef::new("in")),
        destination: Endpoint::Call {
            name: "Linear".to_string(),
            args: vec![Value::Name("d".to_string()), Value::Int(512)],
            kwargs: vec![],
            id: 0,
        },
    }];

    let mut output = String::new();
    generate_module_instantiations(&mut gen, &mut output, &[], &[], &connections).unwrap();

    // Should generate lazy instantiation marker
    assert!(output.contains("self._linear_0 = None"));
    assert!(output.contains("# Lazy instantiation"));
}
