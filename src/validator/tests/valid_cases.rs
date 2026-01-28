use super::fixtures::*;

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
