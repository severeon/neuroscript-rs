//! NeuroScript Graph Validator
//!
//! Validates that NeuroScript programs are well-formed:
//! 1. All referenced neurons exist
//! 2. Connection endpoints match (tuple arity, port names, shapes)
//! 3. No cycles in the dependency graph

pub mod core;

// Re-export public API
pub use core::Validator;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interfaces::*;

    // Helper to create simple neuron
    fn simple_neuron(name: &str, in_shape: Shape, out_shape: Shape) -> NeuronDef {
        NeuronDef {
            name: name.to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: in_shape }],
            outputs: vec![Port { name: "default".to_string(), shape: out_shape }],
            body: NeuronBody::Primitive(ImplRef::Source {
                source: "test".to_string(),
                path: "test".to_string(),
            }),
        }
    }

    // Helper to create multi-port neuron
    fn multi_port_neuron(name: &str, inputs: Vec<Port>, outputs: Vec<Port>) -> NeuronDef {
        NeuronDef {
            name: name.to_string(),
            params: vec![],
            inputs,
            outputs,
            body: NeuronBody::Primitive(ImplRef::Source {
                source: "test".to_string(),
                path: "test".to_string(),
            }),
        }
    }

    // Helper shapes
    fn wildcard() -> Shape {
        Shape::new(vec![Dim::Wildcard])
    }

    fn shape_512() -> Shape {
        Shape::new(vec![Dim::Literal(512)])
    }

    fn shape_256() -> Shape {
        Shape::new(vec![Dim::Literal(256)])
    }

    fn shape_batch_512() -> Shape {
        Shape::new(vec![Dim::Wildcard, Dim::Literal(512)])
    }

    fn shape_batch_256() -> Shape {
        Shape::new(vec![Dim::Wildcard, Dim::Literal(256)])
    }

    // ========== MISSING NEURON TESTS ==========

    #[test]
    fn test_missing_neuron_in_call() {
        let mut program = Program::new();
        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Ref(PortRef::new("in")),
                destination: Endpoint::Call {
                    name: "MissingNeuron".to_string(),
                    args: vec![],
                    kwargs: vec![],
                    id: 0
                },
            }]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::MissingNeuron { name, .. } if name == "MissingNeuron"
        )));
    }

    #[test]
    fn test_missing_neuron_in_match() {
        let mut program = Program::new();
        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Ref(PortRef::new("in")),
                destination: Endpoint::Match(MatchExpr {
                    arms: vec![MatchArm {
                        pattern: wildcard(),
                        guard: None,
                        pipeline: vec![
                            Endpoint::Call {
                                name: "MissingInMatch".to_string(),
                                args: vec![],
                                kwargs: vec![],
                                id: 0
                            }
                        ],
                        is_reachable: true,
                    }],
                }),
            }]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::MissingNeuron { name, .. } if name == "MissingInMatch"
        )));
    }

    // ========== ARITY MISMATCH TESTS ==========

    #[test]
    fn test_arity_mismatch_call_to_call() {
        let mut program = Program::new();

        // TwoOut: 1 input -> 2 outputs
        program.neurons.insert("TwoOut".to_string(), multi_port_neuron(
            "TwoOut",
            vec![Port { name: "default".to_string(), shape: wildcard() }],
            vec![
                Port { name: "a".to_string(), shape: wildcard() },
                Port { name: "b".to_string(), shape: wildcard() },
            ],
        ));

        // OneIn: 1 input -> 1 output
        program.neurons.insert("OneIn".to_string(), simple_neuron("OneIn", wildcard(), wildcard()));

        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Call { name: "TwoOut".to_string(), args: vec![], kwargs: vec![], id: 0 },
                destination: Endpoint::Call { name: "OneIn".to_string(), args: vec![], kwargs: vec![], id: 0 },
            }]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::ArityMismatch { expected: 1, got: 2, .. }
        )));
    }

    #[test]
    fn test_arity_mismatch_tuple_unpacking() {
        let mut program = Program::new();

        // OneOut: 1 input -> 1 output
        program.neurons.insert("OneOut".to_string(), simple_neuron("OneOut", wildcard(), wildcard()));

        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Call { name: "OneOut".to_string(), args: vec![], kwargs: vec![], id: 0 },
                destination: Endpoint::Tuple(vec![
                    PortRef::new("a"),
                    PortRef::new("b"),
                ]),
            }]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::ArityMismatch { expected: 2, got: 1, .. }
        )));
    }

    #[test]
    fn test_arity_mismatch_tuple_to_call() {
        let mut program = Program::new();

        // TwoIn: 2 inputs -> 1 output
        program.neurons.insert("TwoIn".to_string(), multi_port_neuron(
            "TwoIn",
            vec![
                Port { name: "left".to_string(), shape: wildcard() },
                Port { name: "right".to_string(), shape: wildcard() },
            ],
            vec![Port { name: "default".to_string(), shape: wildcard() }],
        ));

        // Fork: 1 input -> 2 outputs
        program.neurons.insert("Fork".to_string(), multi_port_neuron(
            "Fork",
            vec![Port { name: "default".to_string(), shape: wildcard() }],
            vec![
                Port { name: "a".to_string(), shape: wildcard() },
                Port { name: "b".to_string(), shape: wildcard() },
            ],
        ));

        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![
                // Fork creates (a, b)
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Call { name: "Fork".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
                Connection {
                    source: Endpoint::Call { name: "Fork".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Tuple(vec![PortRef::new("a"), PortRef::new("b")]),
                },
                // Only send (a) to TwoIn - arity mismatch
                Connection {
                    source: Endpoint::Tuple(vec![PortRef::new("a")]),
                    destination: Endpoint::Call { name: "TwoIn".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
            ]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::ArityMismatch { expected: 2, got: 1, .. }
        )));
    }

    // ========== SHAPE MISMATCH TESTS ==========

    #[test]
    fn test_shape_mismatch_literal() {
        let mut program = Program::new();
        program.neurons.insert("Out512".to_string(), simple_neuron("Out512", wildcard(), shape_512()));
        program.neurons.insert("In256".to_string(), simple_neuron("In256", shape_256(), wildcard()));

        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Call { name: "Out512".to_string(), args: vec![], kwargs: vec![], id: 0 },
                destination: Endpoint::Call { name: "In256".to_string(), args: vec![], kwargs: vec![], id: 0 },
            }]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(e, ValidationError::PortMismatch { .. })));
    }

    #[test]
    fn test_shape_mismatch_multi_dim() {
        let mut program = Program::new();
        program.neurons.insert("Out512".to_string(), simple_neuron("Out512", wildcard(), shape_batch_512()));
        program.neurons.insert("In256".to_string(), simple_neuron("In256", shape_batch_256(), wildcard()));

        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Call { name: "Out512".to_string(), args: vec![], kwargs: vec![], id: 0 },
                destination: Endpoint::Call { name: "In256".to_string(), args: vec![], kwargs: vec![], id: 0 },
            }]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(e, ValidationError::PortMismatch { .. })));
    }

    #[test]
    fn test_shape_match_exact() {
        let mut program = Program::new();
        program.neurons.insert("Out512".to_string(), simple_neuron("Out512", wildcard(), shape_512()));
        program.neurons.insert("In512".to_string(), simple_neuron("In512", shape_512(), wildcard()));

        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Call { name: "Out512".to_string(), args: vec![], kwargs: vec![], id: 0 },
                destination: Endpoint::Call { name: "In512".to_string(), args: vec![], kwargs: vec![], id: 0 },
            }]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_ok());
    }

    // ========== CYCLE DETECTION TESTS ==========

    #[test]
    fn test_simple_cycle() {
        let mut program = Program::new();
        program.neurons.insert("A".to_string(), simple_neuron("A", wildcard(), wildcard()));
        program.neurons.insert("B".to_string(), simple_neuron("B", wildcard(), wildcard()));

        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![
                Connection {
                    source: Endpoint::Call { name: "A".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Call { name: "B".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
                Connection {
                    source: Endpoint::Call { name: "B".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Call { name: "A".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
            ]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(e, ValidationError::CycleDetected { .. })));
    }

    #[test]
    fn test_cycle_through_unpacked_ports() {
        let mut program = Program::new();
        program.neurons.insert("Fork".to_string(), multi_port_neuron(
            "Fork",
            vec![Port { name: "default".to_string(), shape: wildcard() }],
            vec![
                Port { name: "a".to_string(), shape: wildcard() },
                Port { name: "b".to_string(), shape: wildcard() },
            ],
        ));
        program.neurons.insert("A".to_string(), simple_neuron("A", wildcard(), wildcard()));

        let composite = NeuronDef {
            name: "Composite".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![
                Connection {
                    source: Endpoint::Call { name: "A".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Call { name: "Fork".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
                Connection {
                    source: Endpoint::Call { name: "Fork".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Tuple(vec![PortRef::new("main"), PortRef::new("skip")]),
                },
                // main -> A creates cycle: A -> Fork -> main -> A
                Connection {
                    source: Endpoint::Ref(PortRef::new("main")),
                    destination: Endpoint::Call { name: "A".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
            ]),
        };
        program.neurons.insert("Composite".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(e, ValidationError::CycleDetected { .. })));
    }

    #[test]
    fn test_no_cycle_valid_residual() {
        let mut program = Program::new();

        // Fork: 1 -> 2
        program.neurons.insert("Fork".to_string(), multi_port_neuron(
            "Fork",
            vec![Port { name: "default".to_string(), shape: wildcard() }],
            vec![
                Port { name: "a".to_string(), shape: wildcard() },
                Port { name: "b".to_string(), shape: wildcard() },
            ],
        ));

        // Add: 2 -> 1
        program.neurons.insert("Add".to_string(), multi_port_neuron(
            "Add",
            vec![
                Port { name: "left".to_string(), shape: wildcard() },
                Port { name: "right".to_string(), shape: wildcard() },
            ],
            vec![Port { name: "default".to_string(), shape: wildcard() }],
        ));

        program.neurons.insert("Process".to_string(), simple_neuron("Process", wildcard(), wildcard()));

        // Residual: in -> Fork -> (main, skip), main -> Process -> processed, (processed, skip) -> Add -> out
        let composite = NeuronDef {
            name: "Residual".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Call { name: "Fork".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
                Connection {
                    source: Endpoint::Call { name: "Fork".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Tuple(vec![PortRef::new("main"), PortRef::new("skip")]),
                },
                Connection {
                    source: Endpoint::Ref(PortRef::new("main")),
                    destination: Endpoint::Call { name: "Process".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
                Connection {
                    source: Endpoint::Call { name: "Process".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Ref(PortRef::new("processed")),
                },
                Connection {
                    source: Endpoint::Tuple(vec![PortRef::new("processed"), PortRef::new("skip")]),
                    destination: Endpoint::Call { name: "Add".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
                Connection {
                    source: Endpoint::Call { name: "Add".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Ref(PortRef::new("out")),
                },
            ]),
        };
        program.neurons.insert("Residual".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_ok(), "Valid residual pattern should not have cycles: {:?}", result);
    }

    // ========== VALID CASES ==========

    #[test]
    fn test_empty_graph() {
        let mut program = Program::new();
        let composite = NeuronDef {
            name: "Empty".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![]),
        };
        program.neurons.insert("Empty".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_ok());
    }

    #[test]
    fn test_simple_passthrough() {
        let mut program = Program::new();
        let composite = NeuronDef {
            name: "Passthrough".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Ref(PortRef::new("in")),
                destination: Endpoint::Ref(PortRef::new("out")),
            }]),
        };
        program.neurons.insert("Passthrough".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_pipeline() {
        let mut program = Program::new();
        program.neurons.insert("A".to_string(), simple_neuron("A", wildcard(), wildcard()));
        program.neurons.insert("B".to_string(), simple_neuron("B", wildcard(), wildcard()));

        let composite = NeuronDef {
            name: "Pipeline".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            outputs: vec![Port { name: "default".to_string(), shape: wildcard() }],
            body: NeuronBody::Graph(vec![
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Call { name: "A".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
                Connection {
                    source: Endpoint::Call { name: "A".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Call { name: "B".to_string(), args: vec![], kwargs: vec![], id: 0 },
                },
                Connection {
                    source: Endpoint::Call { name: "B".to_string(), args: vec![], kwargs: vec![], id: 0 },
                    destination: Endpoint::Ref(PortRef::new("out")),
                },
            ]),
        };
        program.neurons.insert("Pipeline".to_string(), composite);

        let result = Validator::validate(&program);
        assert!(result.is_ok());
    }

    #[test]
    fn test_match_exhaustiveness_with_catchall() {
        // Match with catch-all pattern should pass
        let mut program = Program::new();
        let neuron = NeuronDef {
            name: "TestMatch".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Wildcard]) }],
            outputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]) }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Ref(PortRef::new("in")),
                destination: Endpoint::Match(MatchExpr {
                    arms: vec![
                        MatchArm {
                            pattern: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]),
                            guard: None,
                            pipeline: vec![Endpoint::Ref(PortRef::new("out"))],
                            is_reachable: true,
                        },
                        MatchArm {
                            pattern: Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]),
                            guard: None,
                            pipeline: vec![Endpoint::Ref(PortRef::new("out"))],
                            is_reachable: true,
                        },
                    ],
                }),
            }]),
        };
        program.neurons.insert("TestMatch".to_string(), neuron);

        let result = Validator::validate(&program);
        assert!(result.is_ok(), "Match with catch-all pattern should be valid");
    }

    #[test]
    fn test_match_exhaustiveness_without_catchall() {
        // Match without catch-all pattern should fail
        let mut program = Program::new();
        let neuron = NeuronDef {
            name: "TestMatch".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Wildcard]) }],
            outputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]) }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Ref(PortRef::new("in")),
                destination: Endpoint::Match(MatchExpr {
                    arms: vec![
                        MatchArm {
                            pattern: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]),
                            guard: None,
                            pipeline: vec![Endpoint::Ref(PortRef::new("out"))],
                            is_reachable: true,
                        },
                        MatchArm {
                            pattern: Shape::new(vec![Dim::Wildcard, Dim::Literal(256)]),
                            guard: None,
                            pipeline: vec![Endpoint::Ref(PortRef::new("out"))],
                            is_reachable: true,
                        },
                    ],
                }),
            }]),
        };
        program.neurons.insert("TestMatch".to_string(), neuron);

        let result = Validator::validate(&program);
        assert!(result.is_err(), "Match without catch-all should fail");
        if let Err(errors) = result {
            assert!(errors.iter().any(|e| matches!(e, ValidationError::NonExhaustiveMatch { .. })));
        }
    }

    #[test]
    fn test_match_pattern_shadowing() {
        // Match with shadowed pattern should fail
        let mut program = Program::new();
        let neuron = NeuronDef {
            name: "TestMatch".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Wildcard]) }],
            outputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]) }],
            body: NeuronBody::Graph(vec![Connection {
                source: Endpoint::Ref(PortRef::new("in")),
                destination: Endpoint::Match(MatchExpr {
                    arms: vec![
                        MatchArm {
                            pattern: Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]),
                            guard: None,
                            pipeline: vec![Endpoint::Ref(PortRef::new("out"))],
                            is_reachable: true,
                        },
                        MatchArm {
                            pattern: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]),
                            guard: None,
                            pipeline: vec![Endpoint::Ref(PortRef::new("out"))],
                            is_reachable: true,
                        },
                    ],
                }),
            }]),
        };
        program.neurons.insert("TestMatch".to_string(), neuron);

        let result = Validator::validate(&program);
        assert!(result.is_err(), "Match with shadowed pattern should fail");
        if let Err(errors) = result {
            assert!(errors.iter().any(|e| matches!(e, ValidationError::UnreachableMatchArm { .. })));
        }
    }

    #[test]
    fn test_pattern_subsumption() {
        // Test pattern subsumption logic
        let general = Shape::new(vec![Dim::Wildcard, Dim::Wildcard]);
        let specific = Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]);
        assert!(Validator::pattern_subsumes(&general, &specific));

        let named = Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]);
        let literal = Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]);
        assert!(Validator::pattern_subsumes(&named, &literal));

        let lit1 = Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]);
        let lit2 = Shape::new(vec![Dim::Wildcard, Dim::Literal(256)]);
        assert!(!Validator::pattern_subsumes(&lit1, &lit2));
    }

    #[test]
    fn test_is_catch_all_pattern() {
        // All wildcards
        let pattern1 = Shape::new(vec![Dim::Wildcard, Dim::Wildcard]);
        assert!(Validator::is_catch_all_pattern(&pattern1));

        // Named dimensions
        let pattern2 = Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]);
        assert!(Validator::is_catch_all_pattern(&pattern2));

        // Variadic without literals
        let pattern3 = Shape::new(vec![Dim::Variadic("shape".to_string())]);
        assert!(Validator::is_catch_all_pattern(&pattern3));

        // Has literal - not catch-all
        let pattern4 = Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]);
        assert!(!Validator::is_catch_all_pattern(&pattern4));

        // Empty - not catch-all
        let pattern5 = Shape::new(vec![]);
        assert!(!Validator::is_catch_all_pattern(&pattern5));
    }
}
