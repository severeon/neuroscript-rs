use super::fixtures::*;
use crate::interfaces::*;

#[test]
fn test_missing_neuron_in_call() {
    let mut program = ProgramBuilder::new()
        .with_composite(
            "Composite",
            vec![connection(
                ref_endpoint("in"),
                call_endpoint("MissingNeuron"),
            )],
            Some(10),
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(
            e,
            ValidationError::MissingNeuron { name, .. } if name == "MissingNeuron"
        )
    });
}

#[test]
fn test_missing_neuron_in_match() {
    let mut program = ProgramBuilder::new()
        .with_composite(
            "Composite",
            vec![connection(
                ref_endpoint("in"),
                Endpoint::Match(MatchExpr {
                    arms: vec![MatchArm {
                        pattern: wildcard(),
                        guard: None,
                        pipeline: vec![call_endpoint("MissingInMatch")],
                        is_reachable: true,
                    }],
                    id: 0,
                }),
            )],
            Some(10),
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(
            e,
            ValidationError::MissingNeuron { name, .. } if name == "MissingInMatch"
        )
    });
}
