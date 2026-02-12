use super::*;
use std::collections::HashMap;

#[test]
fn test_optimize_matches_basic() {
    // Construct a program with a match expression having an unreachable arm
    let mut program = Program {
        uses: vec![],
        globals: vec![],
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
        id: 0,
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
        max_cycle_depth: Some(10),
        doc: None,
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![connection],
        },
    };

    program.neurons.insert("TestNeuron".to_string(), neuron);

    let pruned = optimize_matches(&mut program, true);
    assert_eq!(pruned, 1);

    // Verify the arm was removed
    let neuron = program.neurons.get("TestNeuron").unwrap();
    if let NeuronBody::Graph { connections, .. } = &neuron.body {
        if let Endpoint::Match(match_expr) = &connections[0].destination {
            assert_eq!(match_expr.arms.len(), 2);
            assert!(match_expr.arms[0].is_reachable);
            assert!(match_expr.arms[1].is_reachable);
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
        globals: vec![],
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
        id: 0,
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
        max_cycle_depth: Some(10),
        doc: None,
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![connection],
        },
    };

    program.neurons.insert("ShadowTest".to_string(), neuron);

    let pruned = optimize_matches(&mut program, true);
    assert_eq!(pruned, 1, "Should prune 1 shadowed arm");

    // Verify only the general pattern remains
    let neuron = program.neurons.get("ShadowTest").unwrap();
    if let NeuronBody::Graph { connections, .. } = &neuron.body {
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
        globals: vec![],
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
        id: 0,
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
        max_cycle_depth: Some(10),
        doc: None,
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![connection],
        },
    };

    program.neurons.insert("GuardTest".to_string(), neuron);

    let pruned = optimize_matches(&mut program, true);
    assert_eq!(pruned, 0, "Guards prevent pruning - both arms reachable");

    // Verify both arms remain
    let neuron = program.neurons.get("GuardTest").unwrap();
    if let NeuronBody::Graph { connections, .. } = &neuron.body {
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
        globals: vec![],
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
        id: 0,
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
        max_cycle_depth: Some(10),
        doc: None,
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![connection],
        },
    };

    program.neurons.insert("MultiPrune".to_string(), neuron);

    let pruned = optimize_matches(&mut program, true);
    assert_eq!(pruned, 3, "Should prune 3 shadowed arms");

    // Verify only catch-all remains
    let neuron = program.neurons.get("MultiPrune").unwrap();
    if let NeuronBody::Graph { connections, .. } = &neuron.body {
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
        globals: vec![],
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
        id: 0,
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
        max_cycle_depth: Some(10),
        doc: None,
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![connection],
        },
    };

    program.neurons.insert("DisabledTest".to_string(), neuron);

    let pruned = optimize_matches(&mut program, false); // Disabled
    assert_eq!(pruned, 0, "No pruning when optimization disabled");

    // Verify both arms remain
    let neuron = program.neurons.get("DisabledTest").unwrap();
    if let NeuronBody::Graph { connections, .. } = &neuron.body {
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
        globals: vec![],
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
        id: 1,
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
        id: 0,
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
        max_cycle_depth: Some(10),
        doc: None,
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![connection],
        },
    };

    program.neurons.insert("NestedTest".to_string(), neuron);

    let pruned = optimize_matches(&mut program, true);
    assert_eq!(pruned, 2, "Should prune 1 outer + 1 inner unreachable arm");

    // Verify pruning at both levels
    let neuron = program.neurons.get("NestedTest").unwrap();
    if let NeuronBody::Graph { connections, .. } = &neuron.body {
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
        globals: vec![],
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
        id: 0,
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
        id: 1,
    };

    let neuron = NeuronDef {
        name: "CountTest".to_string(),
        params: vec![],
        inputs: vec![],
        outputs: vec![],
        max_cycle_depth: Some(10),
        doc: None,
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Match(match1),
                },
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Match(match2),
                },
            ],
        },
    };

    program.neurons.insert("CountTest".to_string(), neuron);

    let count = count_matches(&program);
    assert_eq!(count, 2, "Should count 2 match expressions");
}

#[test]
fn test_pattern_specificity() {
    // Test specificity scoring

    // Literal pattern: [512, 256] - very specific
    let literal_arm = MatchArm {
        pattern: Shape {
            dims: vec![Dim::Literal(512), Dim::Literal(256)],
        },
        guard: None,
        pipeline: vec![],
        is_reachable: true,
    };
    let (literal_score, _) = pattern_specificity(&literal_arm);
    assert_eq!(literal_score, 200, "Two literals = 200");

    // Named pattern: [*, d] - less specific
    let named_arm = MatchArm {
        pattern: Shape {
            dims: vec![Dim::Wildcard, Dim::Named("d".to_string())],
        },
        guard: None,
        pipeline: vec![],
        is_reachable: true,
    };
    let (named_score, _) = pattern_specificity(&named_arm);
    assert_eq!(named_score, 11, "Wildcard + named = 11");

    // Wildcard pattern: [*] - least specific
    let wildcard_arm = MatchArm {
        pattern: Shape {
            dims: vec![Dim::Wildcard],
        },
        guard: None,
        pipeline: vec![],
        is_reachable: true,
    };
    let (wildcard_score, _) = pattern_specificity(&wildcard_arm);
    assert_eq!(wildcard_score, 1, "Single wildcard = 1");

    assert!(literal_score > named_score);
    assert!(named_score > wildcard_score);
}

#[test]
fn test_reorder_match_arms() {
    // Create a match with arms in wrong order (general before specific)
    let mut program = Program {
        uses: vec![],
        globals: vec![],
        neurons: HashMap::new(),
    };

    let match_expr = MatchExpr {
        arms: vec![
            // General pattern first (wrong order)
            MatchArm {
                pattern: Shape {
                    dims: vec![Dim::Wildcard, Dim::Named("d".to_string())],
                },
                guard: None,
                pipeline: vec![],
                is_reachable: true,
            },
            // Specific pattern second (should be first)
            MatchArm {
                pattern: Shape {
                    dims: vec![Dim::Literal(2), Dim::Literal(512)],
                },
                guard: None,
                pipeline: vec![],
                is_reachable: true,
            },
        ],
        id: 0,
    };

    let connection = Connection {
        source: Endpoint::Ref(PortRef::new("in")),
        destination: Endpoint::Match(match_expr),
    };

    let neuron = NeuronDef {
        name: "ReorderTest".to_string(),
        params: vec![],
        inputs: vec![],
        outputs: vec![],
        max_cycle_depth: Some(10),
        doc: None,
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![connection],
        },
    };

    program.neurons.insert("ReorderTest".to_string(), neuron);

    let reordered = reorder_match_arms(&mut program);
    assert_eq!(reordered, 1, "Should reorder 1 match expression");

    // Verify order is now correct (specific first)
    let neuron = program.neurons.get("ReorderTest").unwrap();
    if let NeuronBody::Graph { connections, .. } = &neuron.body {
        if let Endpoint::Match(match_expr) = &connections[0].destination {
            // First arm should now be the specific pattern
            assert!(matches!(
                match_expr.arms[0].pattern.dims[0],
                Dim::Literal(2)
            ));
            assert!(matches!(
                match_expr.arms[0].pattern.dims[1],
                Dim::Literal(512)
            ));

            // Second arm should be the general pattern
            assert!(matches!(match_expr.arms[1].pattern.dims[0], Dim::Wildcard));
            assert!(matches!(match_expr.arms[1].pattern.dims[1], Dim::Named(_)));
        } else {
            panic!("Expected Match endpoint");
        }
    } else {
        panic!("Expected Graph body");
    }
}

#[test]
fn test_static_resolve_concrete_shape() {
    // Test compile-time resolution with fully concrete shape
    let mut ctx = InferenceContext::default();
    ctx.resolved_dims.insert("batch".to_string(), 32);
    ctx.resolved_dims.insert("dim".to_string(), 512);

    let match_expr = MatchExpr {
        arms: vec![
            MatchArm {
                pattern: Shape {
                    dims: vec![Dim::Wildcard, Dim::Literal(256)],
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
                is_reachable: true,
            },
            MatchArm {
                pattern: Shape {
                    dims: vec![Dim::Wildcard, Dim::Named("d".to_string())],
                },
                guard: None,
                pipeline: vec![],
                is_reachable: true,
            },
        ],
        id: 0,
    };

    // Input shape: [batch, dim] where both are resolved
    let input_shape = Shape {
        dims: vec![
            Dim::Named("batch".to_string()),
            Dim::Named("dim".to_string()),
        ],
    };

    let resolved_arm = try_static_resolve(&match_expr, &input_shape, &ctx);
    assert_eq!(resolved_arm, Some(1), "Should resolve to arm 1 ([*, 512])");
}

#[test]
fn test_static_resolve_with_guard() {
    // Test compile-time resolution with guard evaluation
    let mut ctx = InferenceContext::default();
    ctx.resolved_dims.insert("batch".to_string(), 32);
    ctx.resolved_dims.insert("d".to_string(), 1024);

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
                guard: None,
                pipeline: vec![],
                is_reachable: true,
            },
        ],
        id: 0,
    };

    // Input shape has concrete first dimension and resolved second dimension
    let input_shape = Shape {
        dims: vec![Dim::Named("batch".to_string()), Dim::Named("d".to_string())],
    };

    let resolved_arm = try_static_resolve(&match_expr, &input_shape, &ctx);
    assert_eq!(
        resolved_arm,
        Some(0),
        "Should resolve to arm 0 (guard true: 1024 > 512)"
    );
}

#[test]
fn test_static_resolve_runtime_needed() {
    // Test that runtime check is required when shape is not fully concrete
    let ctx = InferenceContext::default();

    let match_expr = MatchExpr {
        arms: vec![MatchArm {
            pattern: Shape {
                dims: vec![Dim::Wildcard, Dim::Literal(512)],
            },
            guard: None,
            pipeline: vec![],
            is_reachable: true,
        }],
        id: 0,
    };

    // Input shape has unresolved dimension
    let input_shape = Shape {
        dims: vec![Dim::Wildcard, Dim::Named("unknown".to_string())],
    };

    let resolved_arm = try_static_resolve(&match_expr, &input_shape, &ctx);
    assert_eq!(
        resolved_arm, None,
        "Should require runtime check (unknown dimension)"
    );
}

#[test]
fn test_pattern_matches_shape() {
    let mut ctx = InferenceContext::default();
    ctx.resolved_dims.insert("batch".to_string(), 32);

    // Test literal matching
    let pattern1 = Shape {
        dims: vec![Dim::Wildcard, Dim::Literal(512)],
    };
    let concrete1 = Shape {
        dims: vec![Dim::Named("batch".to_string()), Dim::Literal(512)],
    };
    assert!(pattern_matches_shape(&pattern1, &concrete1, &ctx));

    // Test named dimension capture
    let pattern2 = Shape {
        dims: vec![Dim::Wildcard, Dim::Named("d".to_string())],
    };
    let concrete2 = Shape {
        dims: vec![Dim::Literal(32), Dim::Literal(256)],
    };
    assert!(pattern_matches_shape(&pattern2, &concrete2, &ctx));

    // Test mismatch
    let pattern3 = Shape {
        dims: vec![Dim::Wildcard, Dim::Literal(512)],
    };
    let concrete3 = Shape {
        dims: vec![Dim::Literal(32), Dim::Literal(256)],
    };
    assert!(!pattern_matches_shape(&pattern3, &concrete3, &ctx));
}

#[test]
fn test_evaluate_guard() {
    let mut ctx = InferenceContext::default();
    ctx.resolved_dims.insert("d".to_string(), 1024);

    // Test greater than
    let guard1 = Value::BinOp {
        op: BinOp::Gt,
        left: Box::new(Value::Name("d".to_string())),
        right: Box::new(Value::Int(512)),
    };
    assert_eq!(try_evaluate_guard(&guard1, &ctx), Some(true));

    // Test less than
    let guard2 = Value::BinOp {
        op: BinOp::Lt,
        left: Box::new(Value::Name("d".to_string())),
        right: Box::new(Value::Int(2048)),
    };
    assert_eq!(try_evaluate_guard(&guard2, &ctx), Some(true));

    // Test equality
    let guard3 = Value::BinOp {
        op: BinOp::Eq,
        left: Box::new(Value::Name("d".to_string())),
        right: Box::new(Value::Int(1024)),
    };
    assert_eq!(try_evaluate_guard(&guard3, &ctx), Some(true));
}
