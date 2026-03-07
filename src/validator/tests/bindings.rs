use super::fixtures::*;
use crate::interfaces::*;

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

#[test]
fn test_mutual_lazy_recursion_detected() {
    // Binding `a` calls SomeNeuron with `b` as argument,
    // and binding `b` calls SomeNeuron with `a` as argument.
    // This forms a cycle: a -> b -> a
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

    assert_validation_error(&mut program, |e| match e {
        ValidationError::Custom(msg) => msg.contains("Mutual @lazy recursion detected"),
        _ => false,
    });
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

    assert_validation_error(&mut program, |e| match e {
        ValidationError::Custom(msg) => msg.contains("Mutual @lazy recursion detected"),
        _ => false,
    });
}
