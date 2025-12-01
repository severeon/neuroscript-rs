use crate::interfaces::*;

/// Optimize match expressions by removing unreachable arms.
///
/// This pass traverses the program graph and finds all `Endpoint::Match` nodes.
/// For each match expression, it removes arms where `is_reachable` is false.
/// It returns the total number of pruned arms.
pub fn optimize_matches(program: &mut Program, enable_dead_elim: bool) -> usize {
    if !enable_dead_elim {
        return 0;
    }
    let mut pruned_count = 0;
    for neuron in program.neurons.values_mut() {
        if let NeuronBody::Graph(connections) = &mut neuron.body {
            for connection in connections {
                pruned_count += optimize_endpoint(&mut connection.source);
                pruned_count += optimize_endpoint(&mut connection.destination);
            }
        }
    }
    pruned_count
}

fn optimize_endpoint(endpoint: &mut Endpoint) -> usize {
    let mut count = 0;
    match endpoint {
        Endpoint::Match(match_expr) => {
            // Prune arms
            let initial_len = match_expr.arms.len();
            match_expr.arms.retain(|arm| arm.is_reachable);
            count += initial_len - match_expr.arms.len();

            // Recurse into remaining arms
            for arm in &mut match_expr.arms {
                for pipe_endpoint in &mut arm.pipeline {
                    count += optimize_endpoint(pipe_endpoint);
                }
            }
        }
        _ => {}
    }
    count
}

/// Count the total number of match expressions in the program.
/// This is useful for logging optimizer statistics.
pub fn count_matches(program: &Program) -> usize {
    let mut count = 0;
    for neuron in program.neurons.values() {
        if let NeuronBody::Graph(connections) = &neuron.body {
            for connection in connections {
                count += count_matches_in_endpoint(&connection.source);
                count += count_matches_in_endpoint(&connection.destination);
            }
        }
    }
    count
}

fn count_matches_in_endpoint(endpoint: &Endpoint) -> usize {
    let mut count = 0;
    match endpoint {
        Endpoint::Match(match_expr) => {
            count += 1;
            // Recurse into arms
            for arm in &match_expr.arms {
                for pipe_endpoint in &arm.pipeline {
                    count += count_matches_in_endpoint(pipe_endpoint);
                }
            }
        }
        _ => {}
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_optimize_matches_basic() {
        // Construct a program with a match expression having an unreachable arm
        let mut program = Program {
            uses: vec![],
            neurons: HashMap::new(),
        };

        let match_expr = MatchExpr {
            arms: vec![
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Literal(1)],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: true,
                },
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Literal(1)],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: false, // This one should be pruned
                },
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Literal(2)],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: true,
                },
            ],
        };

        let connection = Connection {
            source: Endpoint::Ref(PortRef {
                node: "in".to_string(),
                port: "default".to_string(),
            }),
            destination: Endpoint::Match(match_expr),
        };

        let neuron = NeuronDef {
            name: "TestNeuron".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph(vec![connection]),
        };

        program.neurons.insert("TestNeuron".to_string(), neuron);

        let pruned = optimize_matches(&mut program, true);
        assert_eq!(pruned, 1);

        // Verify the arm was removed
        let neuron = program.neurons.get("TestNeuron").unwrap();
        if let NeuronBody::Graph(connections) = &neuron.body {
            if let Endpoint::Match(match_expr) = &connections[0].destination {
                assert_eq!(match_expr.arms.len(), 2);
                assert_eq!(match_expr.arms[0].is_reachable, true);
                assert_eq!(match_expr.arms[1].is_reachable, true);
            } else {
                panic!("Expected Match endpoint");
            }
        } else {
            panic!("Expected Graph body");
        }
    }

    #[test]
    fn test_optimize_matches_shadowing() {
        // Test case: [*, d] shadows [*, 512]
        let mut program = Program {
            uses: vec![],
            neurons: HashMap::new(),
        };

        let match_expr = MatchExpr {
            arms: vec![
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Wildcard, Dim::Named("d".to_string())],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: true,
                },
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Wildcard, Dim::Literal(512)],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: false, // Shadowed by first arm
                },
            ],
        };

        let connection = Connection {
            source: Endpoint::Ref(PortRef::new("in")),
            destination: Endpoint::Match(match_expr),
        };

        let neuron = NeuronDef {
            name: "ShadowTest".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph(vec![connection]),
        };

        program.neurons.insert("ShadowTest".to_string(), neuron);

        let pruned = optimize_matches(&mut program, true);
        assert_eq!(pruned, 1, "Should prune 1 shadowed arm");

        // Verify only the general pattern remains
        let neuron = program.neurons.get("ShadowTest").unwrap();
        if let NeuronBody::Graph(connections) = &neuron.body {
            if let Endpoint::Match(match_expr) = &connections[0].destination {
                assert_eq!(match_expr.arms.len(), 1, "Should have 1 arm after pruning");
                assert_eq!(
                    match_expr.arms[0].pattern.dims,
                    vec![Dim::Wildcard, Dim::Named("d".to_string())]
                );
            } else {
                panic!("Expected Match endpoint");
            }
        } else {
            panic!("Expected Graph body");
        }
    }

    #[test]
    fn test_optimize_matches_guards_prevent_pruning() {
        // Guards make arms reachable even if patterns overlap
        let mut program = Program {
            uses: vec![],
            neurons: HashMap::new(),
        };

        let match_expr = MatchExpr {
            arms: vec![
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Wildcard, Dim::Named("d".to_string())],
                    },
                    guard: Some(Value::BinOp {
                        op: BinOp::Gt,
                        left: Box::new(Value::Name("d".to_string())),
                        right: Box::new(Value::Int(512)),
                    }),
                    pipeline: vec![],
                    is_reachable: true,
                },
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Wildcard, Dim::Named("d".to_string())],
                    },
                    guard: None, // No guard - catch-all for same pattern
                    pipeline: vec![],
                    is_reachable: true, // Should remain reachable (guard makes it distinct)
                },
            ],
        };

        let connection = Connection {
            source: Endpoint::Ref(PortRef::new("in")),
            destination: Endpoint::Match(match_expr),
        };

        let neuron = NeuronDef {
            name: "GuardTest".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph(vec![connection]),
        };

        program.neurons.insert("GuardTest".to_string(), neuron);

        let pruned = optimize_matches(&mut program, true);
        assert_eq!(pruned, 0, "Guards prevent pruning - both arms reachable");

        // Verify both arms remain
        let neuron = program.neurons.get("GuardTest").unwrap();
        if let NeuronBody::Graph(connections) = &neuron.body {
            if let Endpoint::Match(match_expr) = &connections[0].destination {
                assert_eq!(match_expr.arms.len(), 2, "Both arms should remain");
                assert!(match_expr.arms[0].guard.is_some(), "First arm has guard");
                assert!(match_expr.arms[1].guard.is_none(), "Second arm no guard");
            } else {
                panic!("Expected Match endpoint");
            }
        } else {
            panic!("Expected Graph body");
        }
    }

    #[test]
    fn test_optimize_matches_multiple_unreachable() {
        // Multiple unreachable arms pruned at once
        let mut program = Program {
            uses: vec![],
            neurons: HashMap::new(),
        };

        let match_expr = MatchExpr {
            arms: vec![
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Wildcard],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: true,
                },
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Literal(512)],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: false, // Shadowed
                },
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Literal(256)],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: false, // Shadowed
                },
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Named("d".to_string())],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: false, // Shadowed
                },
            ],
        };

        let connection = Connection {
            source: Endpoint::Ref(PortRef::new("in")),
            destination: Endpoint::Match(match_expr),
        };

        let neuron = NeuronDef {
            name: "MultiPrune".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph(vec![connection]),
        };

        program.neurons.insert("MultiPrune".to_string(), neuron);

        let pruned = optimize_matches(&mut program, true);
        assert_eq!(pruned, 3, "Should prune 3 shadowed arms");

        // Verify only catch-all remains
        let neuron = program.neurons.get("MultiPrune").unwrap();
        if let NeuronBody::Graph(connections) = &neuron.body {
            if let Endpoint::Match(match_expr) = &connections[0].destination {
                assert_eq!(match_expr.arms.len(), 1);
                assert_eq!(match_expr.arms[0].pattern.dims, vec![Dim::Wildcard]);
            } else {
                panic!("Expected Match endpoint");
            }
        } else {
            panic!("Expected Graph body");
        }
    }

    #[test]
    fn test_optimize_matches_disabled() {
        // When optimization is disabled, nothing should be pruned
        let mut program = Program {
            uses: vec![],
            neurons: HashMap::new(),
        };

        let match_expr = MatchExpr {
            arms: vec![
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Wildcard],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: true,
                },
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Literal(512)],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: false, // Would be pruned if enabled
                },
            ],
        };

        let connection = Connection {
            source: Endpoint::Ref(PortRef::new("in")),
            destination: Endpoint::Match(match_expr),
        };

        let neuron = NeuronDef {
            name: "DisabledTest".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph(vec![connection]),
        };

        program.neurons.insert("DisabledTest".to_string(), neuron);

        let pruned = optimize_matches(&mut program, false); // Disabled
        assert_eq!(pruned, 0, "No pruning when optimization disabled");

        // Verify both arms remain
        let neuron = program.neurons.get("DisabledTest").unwrap();
        if let NeuronBody::Graph(connections) = &neuron.body {
            if let Endpoint::Match(match_expr) = &connections[0].destination {
                assert_eq!(match_expr.arms.len(), 2, "Both arms should remain");
            } else {
                panic!("Expected Match endpoint");
            }
        } else {
            panic!("Expected Graph body");
        }
    }

    #[test]
    fn test_optimize_matches_nested() {
        // Test nested match expressions (match inside match)
        let mut program = Program {
            uses: vec![],
            neurons: HashMap::new(),
        };

        let inner_match = MatchExpr {
            arms: vec![
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Literal(512)],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: true,
                },
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Literal(512)],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: false, // Unreachable inner arm
                },
            ],
        };

        let outer_match = MatchExpr {
            arms: vec![
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Wildcard],
                    },
                    guard: None,
                    pipeline: vec![Endpoint::Match(inner_match)],
                    is_reachable: true,
                },
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Literal(256)],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: false, // Unreachable outer arm
                },
            ],
        };

        let connection = Connection {
            source: Endpoint::Ref(PortRef::new("in")),
            destination: Endpoint::Match(outer_match),
        };

        let neuron = NeuronDef {
            name: "NestedTest".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph(vec![connection]),
        };

        program.neurons.insert("NestedTest".to_string(), neuron);

        let pruned = optimize_matches(&mut program, true);
        assert_eq!(
            pruned, 2,
            "Should prune 1 outer + 1 inner unreachable arm"
        );

        // Verify pruning at both levels
        let neuron = program.neurons.get("NestedTest").unwrap();
        if let NeuronBody::Graph(connections) = &neuron.body {
            if let Endpoint::Match(outer_match) = &connections[0].destination {
                assert_eq!(outer_match.arms.len(), 1, "Outer should have 1 arm");

                // Check inner match
                if let Endpoint::Match(inner_match) = &outer_match.arms[0].pipeline[0] {
                    assert_eq!(inner_match.arms.len(), 1, "Inner should have 1 arm");
                } else {
                    panic!("Expected nested Match endpoint");
                }
            } else {
                panic!("Expected Match endpoint");
            }
        } else {
            panic!("Expected Graph body");
        }
    }

    #[test]
    fn test_count_matches() {
        let mut program = Program {
            uses: vec![],
            neurons: HashMap::new(),
        };

        // Neuron with 2 match expressions
        let match1 = MatchExpr {
            arms: vec![MatchArm {
                pattern: Shape {
                    dims: vec![Dim::Wildcard],
                },
                guard: None,
                pipeline: vec![],
                is_reachable: true,
            }],
        };

        let match2 = MatchExpr {
            arms: vec![MatchArm {
                pattern: Shape {
                    dims: vec![Dim::Literal(512)],
                },
                guard: None,
                pipeline: vec![],
                is_reachable: true,
            }],
        };

        let neuron = NeuronDef {
            name: "CountTest".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph(vec![
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Match(match1),
                },
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Match(match2),
                },
            ]),
        };

        program.neurons.insert("CountTest".to_string(), neuron);

        let count = count_matches(&program);
        assert_eq!(count, 2, "Should count 2 match expressions");
    }
}
