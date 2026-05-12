//! Tests for contract resolution.

use super::*;
use super::detection::has_named_match;

fn make_shape(dims: Vec<Dim>) -> Shape {
    Shape { dims }
}

fn make_port(name: &str, dims: Vec<Dim>) -> Port {
    Port {
        name: name.to_string(),
        shape: make_shape(dims),
        variadic: false,
    }
}

fn make_neuron_contract_arm(
    input_dims: Vec<Dim>,
    output_dims: Vec<Dim>,
    pipeline: Vec<Endpoint>,
) -> MatchArm {
    MatchArm {
        pattern: MatchPattern::NeuronContract(NeuronPortContract {
            input_ports: vec![("default".to_string(), make_shape(input_dims))],
            output_ports: vec![("default".to_string(), make_shape(output_dims))],
        }),
        guard: None,
        pipeline,
        is_reachable: true,
    }
}

fn ref_endpoint(name: &str) -> Endpoint {
    Endpoint::Ref(PortRef {
        node: name.to_string(),
        port: "default".to_string(),
    })
}

#[test]
fn test_has_named_match_empty() {
    let neuron = NeuronDef {
        name: "Test".to_string(),
        params: vec![],
        inputs: vec![],
        outputs: vec![],
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![],
        },
        max_cycle_depth: Some(10),
        doc: None,
    };
    assert!(!has_named_match(&neuron));
}

#[test]
fn test_has_named_match_with_implicit() {
    let neuron = NeuronDef {
        name: "Test".to_string(),
        params: vec![],
        inputs: vec![],
        outputs: vec![],
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![Connection {
                source: ref_endpoint("in"),
                destination: Endpoint::Match(MatchExpr {
                    subject: MatchSubject::Implicit,
                    arms: vec![],
                    id: 0,
                }),
            }],
        },
        max_cycle_depth: Some(10),
        doc: None,
    };
    assert!(!has_named_match(&neuron));
}

#[test]
fn test_has_named_match_with_named() {
    let neuron = NeuronDef {
        name: "Test".to_string(),
        params: vec![],
        inputs: vec![],
        outputs: vec![],
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![Connection {
                source: ref_endpoint("in"),
                destination: Endpoint::Match(MatchExpr {
                    subject: MatchSubject::Named("block".to_string()),
                    arms: vec![],
                    id: 0,
                }),
            }],
        },
        max_cycle_depth: Some(10),
        doc: None,
    };
    assert!(has_named_match(&neuron));
}

#[test]
fn test_resolve_no_contracts() {
    let mut program = Program::new();
    program.neurons.insert(
        "Simple".to_string(),
        NeuronDef {
            name: "Simple".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![],
                context_unrolls: vec![],
                connections: vec![],
            },
            max_cycle_depth: Some(10),
            doc: None,
        },
    );

    let result = resolve_neuron_contracts(&mut program);
    assert!(result.is_ok());
}

#[test]
fn test_no_matching_arm_reports_error() {
    // Create a concrete neuron with [*, dim] ports
    let mut program = Program::new();
    program.neurons.insert(
        "ConcreteBlock".to_string(),
        NeuronDef {
            name: "ConcreteBlock".to_string(),
            params: vec![],
            inputs: vec![make_port(
                "default",
                vec![Dim::Wildcard, Dim::Named("dim".to_string())],
            )],
            outputs: vec![make_port(
                "default",
                vec![Dim::Wildcard, Dim::Named("dim".to_string())],
            )],
            body: NeuronBody::Primitive(ImplRef::Source {
                source: "test".to_string(),
                path: "test".to_string(),
            }),
            max_cycle_depth: None,
            doc: None,
        },
    );

    // Create a higher-order neuron with a contract that expects 3D shapes
    program.neurons.insert(
        "HigherOrder".to_string(),
        NeuronDef {
            name: "HigherOrder".to_string(),
            params: vec![Param {
                name: "block".to_string(),
                default: None,
                type_annotation: Some(ParamType::Neuron),
            }],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![],
                context_unrolls: vec![],
                connections: vec![Connection {
                    source: ref_endpoint("in"),
                    destination: Endpoint::Match(MatchExpr {
                        subject: MatchSubject::Named("block".to_string()),
                        arms: vec![make_neuron_contract_arm(
                            // Expects 3D input: [*, seq, dim]
                            vec![
                                Dim::Wildcard,
                                Dim::Named("seq".to_string()),
                                Dim::Named("dim".to_string()),
                            ],
                            vec![
                                Dim::Wildcard,
                                Dim::Named("seq".to_string()),
                                Dim::Named("dim".to_string()),
                            ],
                            vec![ref_endpoint("blocks")],
                        )],
                        id: 0,
                    }),
                }],
            },
            max_cycle_depth: Some(10),
            doc: None,
        },
    );

    // Create a caller that passes ConcreteBlock (2D) to HigherOrder (expects 3D)
    program.neurons.insert(
        "Caller".to_string(),
        NeuronDef {
            name: "Caller".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![Binding::new(
                    "ho",
                    "HigherOrder",
                    vec![Value::Name("ConcreteBlock".to_string())],
                )],
                context_unrolls: vec![],
                connections: vec![],
            },
            max_cycle_depth: Some(10),
            doc: None,
        },
    );

    let result = resolve_neuron_contracts(&mut program);
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert_eq!(errors.len(), 1);
    let msg = format!("{}", errors[0]);
    assert!(
        msg.contains("No contract arm"),
        "Expected 'No contract arm' error, got: {}",
        msg
    );
}

#[test]
fn test_multi_endpoint_pipeline_splices_connections() {
    // Create a concrete neuron with matching 2D ports
    let mut program = Program::new();
    program.neurons.insert(
        "Block2D".to_string(),
        NeuronDef {
            name: "Block2D".to_string(),
            params: vec![],
            inputs: vec![make_port(
                "default",
                vec![Dim::Wildcard, Dim::Named("dim".to_string())],
            )],
            outputs: vec![make_port(
                "default",
                vec![Dim::Wildcard, Dim::Named("dim".to_string())],
            )],
            body: NeuronBody::Primitive(ImplRef::Source {
                source: "test".to_string(),
                path: "test".to_string(),
            }),
            max_cycle_depth: None,
            doc: None,
        },
    );

    // Create higher-order neuron where matching arm has 2 endpoints (multi-step)
    program.neurons.insert(
        "MultiStep".to_string(),
        NeuronDef {
            name: "MultiStep".to_string(),
            params: vec![Param {
                name: "block".to_string(),
                default: None,
                type_annotation: Some(ParamType::Neuron),
            }],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![],
                context_unrolls: vec![],
                connections: vec![Connection {
                    source: ref_endpoint("in"),
                    destination: Endpoint::Match(MatchExpr {
                        subject: MatchSubject::Named("block".to_string()),
                        arms: vec![make_neuron_contract_arm(
                            vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                            vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                            // Multi-endpoint pipeline: blocks -> out
                            vec![ref_endpoint("blocks"), ref_endpoint("out")],
                        )],
                        id: 0,
                    }),
                }],
            },
            max_cycle_depth: Some(10),
            doc: None,
        },
    );

    // Caller passes Block2D to MultiStep
    program.neurons.insert(
        "Caller".to_string(),
        NeuronDef {
            name: "Caller".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![Binding::new(
                    "ms",
                    "MultiStep",
                    vec![Value::Name("Block2D".to_string())],
                )],
                context_unrolls: vec![],
                connections: vec![],
            },
            max_cycle_depth: Some(10),
            doc: None,
        },
    );

    let result = resolve_neuron_contracts(&mut program);
    assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

    // Verify the match was replaced and connections were spliced
    let multi_step = program.neurons.get("MultiStep").unwrap();
    if let NeuronBody::Graph { connections, .. } = &multi_step.body {
        // Original: in -> match(block)
        // After resolution with pipeline [blocks, out]:
        //   in -> blocks (match replaced with first endpoint)
        //   blocks -> out (spliced connection)
        assert_eq!(
            connections.len(),
            2,
            "Expected 2 connections after splicing, got {}",
            connections.len()
        );

        // First connection: in -> blocks
        match &connections[0].destination {
            Endpoint::Ref(port_ref) => {
                assert_eq!(port_ref.node, "blocks");
            }
            other => panic!("Expected Ref(blocks) destination, got {:?}", other),
        }

        // Second connection: blocks -> out
        match (&connections[1].source, &connections[1].destination) {
            (Endpoint::Ref(src), Endpoint::Ref(dst)) => {
                assert_eq!(src.node, "blocks");
                assert_eq!(dst.node, "out");
            }
            other => panic!(
                "Expected blocks -> out connection, got {:?}",
                other
            ),
        }
    } else {
        panic!("Expected Graph body");
    }
}

#[test]
fn test_multi_endpoint_pipeline_three_steps() {
    // Test a 3-step pipeline: blocks -> norm -> out
    let mut program = Program::new();
    program.neurons.insert(
        "Block3D".to_string(),
        NeuronDef {
            name: "Block3D".to_string(),
            params: vec![],
            inputs: vec![make_port(
                "default",
                vec![
                    Dim::Wildcard,
                    Dim::Named("seq".to_string()),
                    Dim::Named("dim".to_string()),
                ],
            )],
            outputs: vec![make_port(
                "default",
                vec![
                    Dim::Wildcard,
                    Dim::Named("seq".to_string()),
                    Dim::Named("dim".to_string()),
                ],
            )],
            body: NeuronBody::Primitive(ImplRef::Source {
                source: "test".to_string(),
                path: "test".to_string(),
            }),
            max_cycle_depth: None,
            doc: None,
        },
    );

    program.neurons.insert(
        "ThreeStep".to_string(),
        NeuronDef {
            name: "ThreeStep".to_string(),
            params: vec![Param {
                name: "block".to_string(),
                default: None,
                type_annotation: Some(ParamType::Neuron),
            }],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![],
                context_unrolls: vec![],
                connections: vec![Connection {
                    source: ref_endpoint("in"),
                    destination: Endpoint::Match(MatchExpr {
                        subject: MatchSubject::Named("block".to_string()),
                        arms: vec![make_neuron_contract_arm(
                            vec![
                                Dim::Wildcard,
                                Dim::Named("seq".to_string()),
                                Dim::Named("dim".to_string()),
                            ],
                            vec![
                                Dim::Wildcard,
                                Dim::Named("seq".to_string()),
                                Dim::Named("dim".to_string()),
                            ],
                            // 3-step pipeline
                            vec![
                                ref_endpoint("blocks"),
                                ref_endpoint("norm"),
                                ref_endpoint("out"),
                            ],
                        )],
                        id: 0,
                    }),
                }],
            },
            max_cycle_depth: Some(10),
            doc: None,
        },
    );

    program.neurons.insert(
        "Caller".to_string(),
        NeuronDef {
            name: "Caller".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![Binding::new(
                    "ts",
                    "ThreeStep",
                    vec![Value::Name("Block3D".to_string())],
                )],
                context_unrolls: vec![],
                connections: vec![],
            },
            max_cycle_depth: Some(10),
            doc: None,
        },
    );

    let result = resolve_neuron_contracts(&mut program);
    assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

    let three_step = program.neurons.get("ThreeStep").unwrap();
    if let NeuronBody::Graph { connections, .. } = &three_step.body {
        // in -> blocks, blocks -> norm, norm -> out
        assert_eq!(connections.len(), 3);
        // Verify chain
        match &connections[0].destination {
            Endpoint::Ref(r) => assert_eq!(r.node, "blocks"),
            o => panic!("Expected blocks, got {:?}", o),
        }
        match (&connections[1].source, &connections[1].destination) {
            (Endpoint::Ref(s), Endpoint::Ref(d)) => {
                assert_eq!(s.node, "blocks");
                assert_eq!(d.node, "norm");
            }
            o => panic!("Expected blocks->norm, got {:?}", o),
        }
        match (&connections[2].source, &connections[2].destination) {
            (Endpoint::Ref(s), Endpoint::Ref(d)) => {
                assert_eq!(s.node, "norm");
                assert_eq!(d.node, "out");
            }
            o => panic!("Expected norm->out, got {:?}", o),
        }
    }
}

#[test]
fn test_single_endpoint_resolution_succeeds() {
    // Create a concrete neuron with matching 2D ports
    let mut program = Program::new();
    program.neurons.insert(
        "Block2D".to_string(),
        NeuronDef {
            name: "Block2D".to_string(),
            params: vec![],
            inputs: vec![make_port(
                "default",
                vec![Dim::Wildcard, Dim::Named("dim".to_string())],
            )],
            outputs: vec![make_port(
                "default",
                vec![Dim::Wildcard, Dim::Named("dim".to_string())],
            )],
            body: NeuronBody::Primitive(ImplRef::Source {
                source: "test".to_string(),
                path: "test".to_string(),
            }),
            max_cycle_depth: None,
            doc: None,
        },
    );

    // Higher-order neuron with single-endpoint arm
    program.neurons.insert(
        "Wrapper".to_string(),
        NeuronDef {
            name: "Wrapper".to_string(),
            params: vec![Param {
                name: "block".to_string(),
                default: None,
                type_annotation: Some(ParamType::Neuron),
            }],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![],
                context_unrolls: vec![],
                connections: vec![Connection {
                    source: ref_endpoint("in"),
                    destination: Endpoint::Match(MatchExpr {
                        subject: MatchSubject::Named("block".to_string()),
                        arms: vec![make_neuron_contract_arm(
                            vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                            vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                            vec![ref_endpoint("blocks")],
                        )],
                        id: 0,
                    }),
                }],
            },
            max_cycle_depth: Some(10),
            doc: None,
        },
    );

    // Caller passes Block2D
    program.neurons.insert(
        "Caller".to_string(),
        NeuronDef {
            name: "Caller".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![Binding::new(
                    "w",
                    "Wrapper",
                    vec![Value::Name("Block2D".to_string())],
                )],
                context_unrolls: vec![],
                connections: vec![],
            },
            max_cycle_depth: Some(10),
            doc: None,
        },
    );

    let result = resolve_neuron_contracts(&mut program);
    assert!(result.is_ok());

    // Verify the match was replaced with the ref endpoint
    let wrapper = program.neurons.get("Wrapper").unwrap();
    if let NeuronBody::Graph { connections, .. } = &wrapper.body {
        assert_eq!(connections.len(), 1);
        match &connections[0].destination {
            Endpoint::Ref(port_ref) => {
                assert_eq!(port_ref.node, "blocks");
            }
            other => panic!("Expected Ref endpoint, got {:?}", other),
        }
    } else {
        panic!("Expected Graph body");
    }
}

#[test]
fn test_contract_match_nested_in_if_branch() {
    // Contract match inside an if-expression branch should still be resolved
    let mut program = Program::new();
    program.neurons.insert(
        "Block2D".to_string(),
        NeuronDef {
            name: "Block2D".to_string(),
            params: vec![],
            inputs: vec![make_port(
                "default",
                vec![Dim::Wildcard, Dim::Named("dim".to_string())],
            )],
            outputs: vec![make_port(
                "default",
                vec![Dim::Wildcard, Dim::Named("dim".to_string())],
            )],
            body: NeuronBody::Primitive(ImplRef::Source {
                source: "test".to_string(),
                path: "test".to_string(),
            }),
            max_cycle_depth: None,
            doc: None,
        },
    );

    // Higher-order neuron with contract match nested inside an if branch pipeline
    let contract_match = Endpoint::Match(MatchExpr {
        subject: MatchSubject::Named("block".to_string()),
        arms: vec![make_neuron_contract_arm(
            vec![Dim::Wildcard, Dim::Named("dim".to_string())],
            vec![Dim::Wildcard, Dim::Named("dim".to_string())],
            vec![ref_endpoint("blocks")],
        )],
        id: 0,
    });

    program.neurons.insert(
        "Conditional".to_string(),
        NeuronDef {
            name: "Conditional".to_string(),
            params: vec![
                Param {
                    name: "block".to_string(),
                    default: None,
                    type_annotation: Some(ParamType::Neuron),
                },
                Param {
                    name: "use_block".to_string(),
                    default: Some(Value::Int(1)),
                    type_annotation: None,
                },
            ],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![],
                context_unrolls: vec![],
                connections: vec![Connection {
                    source: ref_endpoint("in"),
                    destination: Endpoint::If(IfExpr {
                        branches: vec![IfBranch {
                            condition: Value::Name("use_block".to_string()),
                            pipeline: vec![contract_match],
                        }],
                        else_branch: Some(vec![ref_endpoint("out")]),
                        id: 0,
                    }),
                }],
            },
            max_cycle_depth: Some(10),
            doc: None,
        },
    );

    // Caller
    program.neurons.insert(
        "Caller".to_string(),
        NeuronDef {
            name: "Caller".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![Binding::new(
                    "c",
                    "Conditional",
                    vec![
                        Value::Name("Block2D".to_string()),
                        Value::Int(1),
                    ],
                )],
                context_unrolls: vec![],
                connections: vec![],
            },
            max_cycle_depth: Some(10),
            doc: None,
        },
    );

    let result = resolve_neuron_contracts(&mut program);
    assert!(result.is_ok());

    // Verify the nested contract match was resolved inside the if branch
    let conditional = program.neurons.get("Conditional").unwrap();
    if let NeuronBody::Graph { connections, .. } = &conditional.body {
        if let Endpoint::If(if_expr) = &connections[0].destination {
            assert_eq!(if_expr.branches.len(), 1);
            // The contract match should have been replaced with Ref("blocks")
            assert_eq!(if_expr.branches[0].pipeline.len(), 1);
            match &if_expr.branches[0].pipeline[0] {
                Endpoint::Ref(port_ref) => {
                    assert_eq!(port_ref.node, "blocks");
                }
                other => panic!(
                    "Expected Ref endpoint in if branch, got {:?}",
                    other
                ),
            }
        } else {
            panic!("Expected If endpoint");
        }
    } else {
        panic!("Expected Graph body");
    }
}

#[test]
fn test_unresolved_contract_detected_post_resolution() {
    // A higher-order neuron with a Named match but no call sites should
    // be flagged as unresolved in the post-resolution check
    let mut program = Program::new();

    program.neurons.insert(
        "Uncalled".to_string(),
        NeuronDef {
            name: "Uncalled".to_string(),
            params: vec![Param {
                name: "block".to_string(),
                default: None,
                type_annotation: Some(ParamType::Neuron),
            }],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![],
                context_unrolls: vec![],
                connections: vec![Connection {
                    source: ref_endpoint("in"),
                    destination: Endpoint::Match(MatchExpr {
                        subject: MatchSubject::Named("block".to_string()),
                        arms: vec![make_neuron_contract_arm(
                            vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                            vec![Dim::Wildcard, Dim::Named("dim".to_string())],
                            vec![ref_endpoint("blocks")],
                        )],
                        id: 0,
                    }),
                }],
            },
            max_cycle_depth: Some(10),
            doc: None,
        },
    );

    let result = resolve_neuron_contracts(&mut program);
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert_eq!(errors.len(), 1);
    let msg = format!("{}", errors[0]);
    assert!(
        msg.contains("Unresolved contract match"),
        "Expected 'Unresolved contract match' error, got: {}",
        msg
    );
}

#[test]
fn test_parameter_default_substitution_enables_matching() {
    // A concrete neuron with named dimensions and default values should
    // have those dimensions substituted before matching
    let mut program = Program::new();

    // TransformerBlock with d_model parameter defaulting to 512
    program.neurons.insert(
        "TransformerBlock".to_string(),
        NeuronDef {
            name: "TransformerBlock".to_string(),
            params: vec![Param {
                name: "d_model".to_string(),
                default: Some(Value::Int(512)),
                type_annotation: None,
            }],
            inputs: vec![make_port(
                "default",
                vec![
                    Dim::Wildcard,
                    Dim::Named("seq".to_string()),
                    Dim::Named("d_model".to_string()),
                ],
            )],
            outputs: vec![make_port(
                "default",
                vec![
                    Dim::Wildcard,
                    Dim::Named("seq".to_string()),
                    Dim::Named("d_model".to_string()),
                ],
            )],
            body: NeuronBody::Primitive(ImplRef::Source {
                source: "core".to_string(),
                path: "transformer/TransformerBlock".to_string(),
            }),
            max_cycle_depth: None,
            doc: None,
        },
    );

    // Higher-order neuron matching on 3D shapes
    program.neurons.insert(
        "Stack".to_string(),
        NeuronDef {
            name: "Stack".to_string(),
            params: vec![Param {
                name: "block".to_string(),
                default: None,
                type_annotation: Some(ParamType::Neuron),
            }],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![],
                context_unrolls: vec![],
                connections: vec![Connection {
                    source: ref_endpoint("in"),
                    destination: Endpoint::Match(MatchExpr {
                        subject: MatchSubject::Named("block".to_string()),
                        arms: vec![make_neuron_contract_arm(
                            vec![
                                Dim::Wildcard,
                                Dim::Named("seq".to_string()),
                                Dim::Named("d".to_string()),
                            ],
                            vec![
                                Dim::Wildcard,
                                Dim::Named("seq".to_string()),
                                Dim::Named("d".to_string()),
                            ],
                            vec![ref_endpoint("layers")],
                        )],
                        id: 0,
                    }),
                }],
            },
            max_cycle_depth: Some(10),
            doc: None,
        },
    );

    // Caller passes TransformerBlock
    program.neurons.insert(
        "Caller".to_string(),
        NeuronDef {
            name: "Caller".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph {
                context_bindings: vec![Binding::new(
                    "s",
                    "Stack",
                    vec![Value::Name("TransformerBlock".to_string())],
                )],
                context_unrolls: vec![],
                connections: vec![],
            },
            max_cycle_depth: Some(10),
            doc: None,
        },
    );

    let result = resolve_neuron_contracts(&mut program);
    assert!(
        result.is_ok(),
        "Expected Ok after default substitution, got: {:?}",
        result
    );

    // Verify the match was resolved
    let stack = program.neurons.get("Stack").unwrap();
    if let NeuronBody::Graph { connections, .. } = &stack.body {
        match &connections[0].destination {
            Endpoint::Ref(port_ref) => {
                assert_eq!(port_ref.node, "layers");
            }
            other => panic!("Expected Ref(layers), got {:?}", other),
        }
    }
}

#[test]
fn test_build_default_bindings() {
    use super::resolution::build_default_bindings;

    let params = vec![
        Param {
            name: "d_model".to_string(),
            default: Some(Value::Int(512)),
            type_annotation: None,
        },
        Param {
            name: "block".to_string(),
            default: None,
            type_annotation: Some(ParamType::Neuron),
        },
        Param {
            name: "num_heads".to_string(),
            default: Some(Value::Int(8)),
            type_annotation: None,
        },
    ];

    let bindings = build_default_bindings(&params);
    assert_eq!(bindings.len(), 2);
    assert_eq!(bindings["d_model"], 512);
    assert_eq!(bindings["num_heads"], 8);
    assert!(!bindings.contains_key("block"));
}
