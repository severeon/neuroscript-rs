use super::fixtures::*;
use crate::interfaces::SEQUENTIAL_PSEUDO_NEURON;

#[test]
fn test_empty_graph() {
    let mut program = ProgramBuilder::new()
        .with_composite("Empty", vec![], Some(10))
        .build();

    assert_validation_ok(&mut program);
}

#[test]
fn test_simple_passthrough() {
    let mut program = ProgramBuilder::new()
        .with_composite(
            "Passthrough",
            vec![connection(ref_endpoint("in"), ref_endpoint("out"))],
            Some(10),
        )
        .build();

    assert_validation_ok(&mut program);
}

#[test]
fn test_valid_pipeline() {
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("A", wildcard(), wildcard())
        .with_simple_neuron("B", wildcard(), wildcard())
        .with_composite(
            "Pipeline",
            vec![
                connection(ref_endpoint("in"), call_endpoint("A")),
                connection(call_endpoint("A"), call_endpoint("B")),
                connection(call_endpoint("B"), ref_endpoint("out")),
            ],
            Some(10),
        )
        .build();

    assert_validation_ok(&mut program);
}

#[test]
fn test_sequential_pseudo_neuron_passes_validation() {
    // The @wrap desugar pass introduces __sequential__ as a synthetic neuron.
    // The validator must recognise it without a user-defined definition.
    let mut program = ProgramBuilder::new()
        .with_composite(
            "Wrapped",
            vec![
                connection(ref_endpoint("in"), call_endpoint(SEQUENTIAL_PSEUDO_NEURON)),
                connection(call_endpoint(SEQUENTIAL_PSEUDO_NEURON), ref_endpoint("out")),
            ],
            Some(10),
        )
        .build();

    assert_validation_ok(&mut program);
}
