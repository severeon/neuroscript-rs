use super::fixtures::*;
use crate::interfaces::*;
use crate::validator::Validator;

/// Helper to build a neuron with context bindings for testing
fn neuron_with_bindings(name: &str, bindings: Vec<Binding>) -> NeuronDef {
    NeuronDef {
        name: name.to_string(),
        params: vec![],
        inputs: vec![default_port(wildcard())],
        outputs: vec![default_port(wildcard())],
        body: NeuronBody::Graph {
            context_bindings: bindings,
            context_unrolls: vec![],
            connections: vec![connection(ref_endpoint("in"), ref_endpoint("out"))],
        },
        max_cycle_depth: None,
        doc: None,
    }
}

fn lazy_binding(name: &str, call_name: &str, args: Vec<Value>) -> Binding {
    Binding {
        name: name.to_string(),
        call_name: call_name.to_string(),
        args,
        kwargs: vec![],
        scope: Scope::Instance { lazy: true },
        frozen: false,
        unroll_group: None,
    }
}

fn lazy_binding_with_kwargs(
    name: &str,
    call_name: &str,
    args: Vec<Value>,
    kwargs: Vec<Kwarg>,
) -> Binding {
    Binding {
        name: name.to_string(),
        call_name: call_name.to_string(),
        args,
        kwargs,
        scope: Scope::Instance { lazy: true },
        frozen: false,
        unroll_group: None,
    }
}

fn eager_binding(name: &str, call_name: &str, args: Vec<Value>) -> Binding {
    Binding {
        name: name.to_string(),
        call_name: call_name.to_string(),
        args,
        kwargs: vec![],
        scope: Scope::Instance { lazy: false },
        frozen: false,
        unroll_group: None,
    }
}

fn is_mutual_recursion(e: &ValidationError) -> bool {
    matches!(e, ValidationError::MutualLazyRecursion { .. })
}

#[test]
fn test_mutual_lazy_recursion_detected() {
    // a references b, b references a => cycle: a -> b -> a
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("SomeNeuron", wildcard(), wildcard())
        .with_neuron(
            "TestNeuron",
            neuron_with_bindings(
                "TestNeuron",
                vec![
                    lazy_binding("a", "SomeNeuron", vec![Value::Name("b".to_string())]),
                    lazy_binding("b", "SomeNeuron", vec![Value::Name("a".to_string())]),
                ],
            ),
        )
        .build();

    assert_validation_error(&mut program, is_mutual_recursion);
}

#[test]
fn test_independent_lazy_bindings_pass() {
    // Two @lazy bindings that don't reference each other should pass
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("NeuronA", wildcard(), wildcard())
        .with_simple_neuron("NeuronB", wildcard(), wildcard())
        .with_neuron(
            "TestNeuron",
            neuron_with_bindings(
                "TestNeuron",
                vec![
                    lazy_binding("a", "NeuronA", vec![Value::Int(42)]),
                    lazy_binding("b", "NeuronB", vec![Value::Int(99)]),
                ],
            ),
        )
        .build();

    assert_validation_ok(&mut program);
}

#[test]
fn test_three_way_lazy_cycle_detected() {
    // a -> b -> c -> a (three-way cycle)
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("N", wildcard(), wildcard())
        .with_neuron(
            "TestNeuron",
            neuron_with_bindings(
                "TestNeuron",
                vec![
                    lazy_binding("a", "N", vec![Value::Name("b".to_string())]),
                    lazy_binding("b", "N", vec![Value::Name("c".to_string())]),
                    lazy_binding("c", "N", vec![Value::Name("a".to_string())]),
                ],
            ),
        )
        .build();

    assert_validation_error(&mut program, is_mutual_recursion);
}

#[test]
fn test_self_recursive_lazy_not_double_reported_as_mutual() {
    // A single @lazy binding that calls itself should only produce the
    // existing self-recursion error, not a mutual recursion error.
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("TestNeuron", wildcard(), wildcard())
        .with_neuron(
            "TestNeuron",
            neuron_with_bindings(
                "TestNeuron",
                vec![lazy_binding("a", "TestNeuron", vec![Value::Int(1)])],
            ),
        )
        .build();

    let result = Validator::validate(&mut program);
    match &result {
        Ok(()) => {} // self-recursion with args is allowed
        Err(errors) => {
            assert!(
                !errors.iter().any(is_mutual_recursion),
                "Self-recursive @lazy binding should not trigger mutual recursion error: {:?}",
                errors
            );
        }
    }
}

#[test]
fn test_mixed_lazy_and_eager_bindings_no_false_cycle() {
    // Only @lazy bindings participate in mutual recursion detection.
    // An eager binding referencing a lazy binding should not trigger
    // a mutual recursion error.
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("N", wildcard(), wildcard())
        .with_neuron(
            "TestNeuron",
            neuron_with_bindings(
                "TestNeuron",
                vec![
                    lazy_binding("a", "N", vec![Value::Name("b".to_string())]),
                    eager_binding("b", "N", vec![Value::Name("a".to_string())]),
                ],
            ),
        )
        .build();

    let result = Validator::validate(&mut program);
    match &result {
        Ok(()) => {}
        Err(errors) => {
            assert!(
                !errors.iter().any(is_mutual_recursion),
                "Mixed lazy/eager bindings should not trigger mutual recursion: {:?}",
                errors
            );
        }
    }
}

#[test]
fn test_sibling_cycles_sharing_node_both_reported() {
    // a references both b and c; b references a; c references a.
    // This creates two sibling cycles: a->b->a and a->c->a.
    // Both should be reported.
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("N", wildcard(), wildcard())
        .with_neuron(
            "TestNeuron",
            neuron_with_bindings(
                "TestNeuron",
                vec![
                    lazy_binding(
                        "a",
                        "N",
                        vec![
                            Value::Name("b".to_string()),
                            Value::Name("c".to_string()),
                        ],
                    ),
                    lazy_binding("b", "N", vec![Value::Name("a".to_string())]),
                    lazy_binding("c", "N", vec![Value::Name("a".to_string())]),
                ],
            ),
        )
        .build();

    let result = Validator::validate(&mut program);
    let errors = result.unwrap_err();
    let mutual_count = errors.iter().filter(|e| is_mutual_recursion(e)).count();
    assert_eq!(
        mutual_count, 2,
        "Expected 2 sibling cycles reported, got {}: {:?}",
        mutual_count, errors
    );
}

#[test]
fn test_three_bindings_only_two_form_cycle() {
    // a -> b -> a forms a cycle, c is independent (no cycle)
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("N", wildcard(), wildcard())
        .with_neuron(
            "TestNeuron",
            neuron_with_bindings(
                "TestNeuron",
                vec![
                    lazy_binding("a", "N", vec![Value::Name("b".to_string())]),
                    lazy_binding("b", "N", vec![Value::Name("a".to_string())]),
                    lazy_binding("c", "N", vec![Value::Int(42)]),
                ],
            ),
        )
        .build();

    let result = Validator::validate(&mut program);
    let errors = result.unwrap_err();
    let mutual_count = errors.iter().filter(|e| is_mutual_recursion(e)).count();
    assert_eq!(
        mutual_count, 1,
        "Expected exactly 1 cycle, got {}: {:?}",
        mutual_count, errors
    );
}

#[test]
fn test_cycle_via_binop_reference() {
    // a references b through a BinOp expression: a = N(b + 1)
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("N", wildcard(), wildcard())
        .with_neuron(
            "TestNeuron",
            neuron_with_bindings(
                "TestNeuron",
                vec![
                    lazy_binding(
                        "a",
                        "N",
                        vec![Value::BinOp {
                            op: BinOp::Add,
                            left: Box::new(Value::Name("b".to_string())),
                            right: Box::new(Value::Int(1)),
                        }],
                    ),
                    lazy_binding("b", "N", vec![Value::Name("a".to_string())]),
                ],
            ),
        )
        .build();

    assert_validation_error(&mut program, is_mutual_recursion);
}

#[test]
fn test_cycle_via_kwargs_reference() {
    // a references b through a kwarg: a = N(key=b)
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("N", wildcard(), wildcard())
        .with_neuron(
            "TestNeuron",
            neuron_with_bindings(
                "TestNeuron",
                vec![
                    lazy_binding_with_kwargs(
                        "a",
                        "N",
                        vec![],
                        vec![("key".to_string(), Value::Name("b".to_string()))],
                    ),
                    lazy_binding("b", "N", vec![Value::Name("a".to_string())]),
                ],
            ),
        )
        .build();

    assert_validation_error(&mut program, is_mutual_recursion);
}

#[test]
fn test_mutual_lazy_recursion_has_source_span() {
    // Parse from source so pest captures spans, then validate.
    // The MutualLazyRecursion error should carry a non-None span
    // pointing back into the source.
    let source = r#"
neuron Helper(x):
    in: [*shape]
    out: [*shape]
    impl: core,nn/Identity

neuron MutualTest(dim):
    in: [*shape]
    out: [*shape]
    context:
        @lazy a = Helper(b)
        @lazy b = Helper(a)
    graph:
        in -> a -> out
"#;

    let mut program = crate::parse(source).expect("should parse successfully");
    let result = crate::validate(&mut program);
    assert!(result.is_err(), "Expected MutualLazyRecursion validation error");
    let errors = result.unwrap_err();
    let mutual_error = errors
        .iter()
        .find(|e| matches!(e, ValidationError::MutualLazyRecursion { .. }))
        .expect("Expected a MutualLazyRecursion error");
    assert!(
        mutual_error.span().is_some(),
        "MutualLazyRecursion error should have a source span, but span() returned None"
    );
}
