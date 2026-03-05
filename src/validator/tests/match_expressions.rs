use super::fixtures::*;
use crate::interfaces::*;
use crate::validator::Validator;

#[test]
fn test_match_exhaustiveness_with_catchall() {
    let mut program = ProgramBuilder::new()
        .with_composite_ports(
            "TestMatch",
            vec![default_port(shape_two_wildcard())],
            vec![default_port(Shape::new(vec![
                Dim::Wildcard,
                Dim::Literal(512),
            ]))],
            vec![connection(
                ref_endpoint("in"),
                Endpoint::Match(MatchExpr {
                    subject: MatchSubject::Implicit,
                    arms: vec![
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![Dim::Wildcard, Dim::Literal(512)])),
                            guard: None,
                            pipeline: vec![ref_endpoint("out")],
                            is_reachable: true,
                        },
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![Dim::Wildcard, named_dim("d")])),
                            guard: None,
                            pipeline: vec![ref_endpoint("out")],
                            is_reachable: true,
                        },
                    ],
                    id: 0,
                }),
            )],
            Some(10),
        )
        .build();

    assert_validation_ok(&mut program);
}

#[test]
fn test_match_exhaustiveness_without_catchall() {
    let mut program = ProgramBuilder::new()
        .with_composite_ports(
            "TestMatch",
            vec![default_port(shape_two_wildcard())],
            vec![default_port(Shape::new(vec![
                Dim::Wildcard,
                Dim::Literal(512),
            ]))],
            vec![connection(
                ref_endpoint("in"),
                Endpoint::Match(MatchExpr {
                    subject: MatchSubject::Implicit,
                    arms: vec![
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![Dim::Wildcard, Dim::Literal(512)])),
                            guard: None,
                            pipeline: vec![ref_endpoint("out")],
                            is_reachable: true,
                        },
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![Dim::Wildcard, Dim::Literal(256)])),
                            guard: None,
                            pipeline: vec![ref_endpoint("out")],
                            is_reachable: true,
                        },
                    ],
                    id: 0,
                }),
            )],
            Some(10),
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(e, ValidationError::NonExhaustiveMatch { .. })
    });
}

#[test]
fn test_match_pattern_shadowing() {
    let mut program = ProgramBuilder::new()
        .with_composite_ports(
            "TestMatch",
            vec![default_port(shape_two_wildcard())],
            vec![default_port(Shape::new(vec![
                Dim::Wildcard,
                Dim::Literal(512),
            ]))],
            vec![connection(
                ref_endpoint("in"),
                Endpoint::Match(MatchExpr {
                    subject: MatchSubject::Implicit,
                    arms: vec![
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![Dim::Wildcard, named_dim("d")])),
                            guard: None,
                            pipeline: vec![ref_endpoint("out")],
                            is_reachable: true,
                        },
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![Dim::Wildcard, Dim::Literal(512)])),
                            guard: None,
                            pipeline: vec![ref_endpoint("out")],
                            is_reachable: true,
                        },
                        MatchArm {
                            pattern: MatchPattern::Shape(shape_two_wildcard()),
                            guard: None,
                            pipeline: vec![ref_endpoint("out")],
                            is_reachable: true,
                        },
                    ],
                    id: 0,
                }),
            )],
            Some(10),
        )
        .build();

    assert_validation_ok(&mut program);

    // Verify is_reachable flags
    let neuron = program.neurons.get("TestMatch").unwrap();
    if let NeuronBody::Graph { connections, .. } = &neuron.body {
        if let Endpoint::Match(match_expr) = &connections[0].destination {
            assert!(
                match_expr.arms[0].is_reachable,
                "First arm should be reachable"
            );
            assert!(
                !match_expr.arms[1].is_reachable,
                "Second arm (specific) should be unreachable (shadowed by first)"
            );
            assert!(
                !match_expr.arms[2].is_reachable,
                "Third arm (catch-all) should be unreachable (shadowed by first)"
            );
        }
    }
}

#[test]
fn test_pattern_subsumption() {
    let general = shape_two_wildcard();
    let specific = Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]);
    assert!(Validator::pattern_subsumes(&general, &specific));

    let named = Shape::new(vec![Dim::Wildcard, named_dim("d")]);
    let literal = Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]);
    assert!(Validator::pattern_subsumes(&named, &literal));

    let lit1 = Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]);
    let lit2 = Shape::new(vec![Dim::Wildcard, Dim::Literal(256)]);
    assert!(!Validator::pattern_subsumes(&lit1, &lit2));
}

#[test]
fn test_is_catch_all_pattern() {
    // All wildcards
    let pattern1 = shape_two_wildcard();
    assert!(Validator::is_catch_all_pattern(&pattern1));

    // Named dimensions
    let pattern2 = Shape::new(vec![Dim::Wildcard, named_dim("d")]);
    assert!(Validator::is_catch_all_pattern(&pattern2));

    // Variadic without literals
    let pattern3 = Shape::new(vec![variadic_dim("shape")]);
    assert!(Validator::is_catch_all_pattern(&pattern3));

    // Has literal - not catch-all
    let pattern4 = Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]);
    assert!(!Validator::is_catch_all_pattern(&pattern4));

    // Empty - not catch-all
    let pattern5 = Shape::new(vec![]);
    assert!(!Validator::is_catch_all_pattern(&pattern5));
}

#[test]
fn test_match_inconsistent_port_count() {
    // Arm 1 resolves to 1 port (ref "out"), arm 2 resolves to 2 ports (tuple)
    let mut program = ProgramBuilder::new()
        .with_composite_ports(
            "TestMatch",
            vec![default_port(shape_two_wildcard())],
            vec![default_port(wildcard()), port("extra", wildcard())],
            vec![connection(
                ref_endpoint("in"),
                Endpoint::Match(MatchExpr {
                    subject: MatchSubject::Implicit,
                    arms: vec![
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![
                                Dim::Wildcard,
                                Dim::Literal(512),
                            ])),
                            guard: None,
                            pipeline: vec![ref_endpoint("out")],
                            is_reachable: true,
                        },
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![
                                Dim::Wildcard,
                                named_dim("d"),
                            ])),
                            guard: None,
                            pipeline: vec![tuple_endpoint(vec!["out", "extra"])],
                            is_reachable: true,
                        },
                    ],
                    id: 0,
                }),
            )],
            Some(10),
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(e, ValidationError::InconsistentArmPorts { .. })
    });
}

#[test]
fn test_match_inconsistent_port_names() {
    // Both arms produce 1 port but with different names ("out" vs "extra")
    let mut program = ProgramBuilder::new()
        .with_composite_ports(
            "TestMatch",
            vec![default_port(shape_two_wildcard())],
            vec![default_port(wildcard()), port("extra", wildcard())],
            vec![connection(
                ref_endpoint("in"),
                Endpoint::Match(MatchExpr {
                    subject: MatchSubject::Implicit,
                    arms: vec![
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![
                                Dim::Wildcard,
                                Dim::Literal(512),
                            ])),
                            guard: None,
                            pipeline: vec![ref_endpoint("out")],
                            is_reachable: true,
                        },
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![
                                Dim::Wildcard,
                                named_dim("d"),
                            ])),
                            guard: None,
                            pipeline: vec![ref_endpoint("extra")],
                            is_reachable: true,
                        },
                    ],
                    id: 0,
                }),
            )],
            Some(10),
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(e, ValidationError::InconsistentArmPorts { .. })
    });
}
