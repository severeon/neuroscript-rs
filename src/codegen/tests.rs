use super::*;
use crate::interfaces::*;
use crate::{parse, validate};

#[test]
fn test_codegen_match() {
    // Construct a simple program with a match expression
    let mut program = Program::new();
    let neuron = NeuronDef {
        name: "MatchTest".to_string(),
        params: vec![],
        inputs: vec![Port {
            name: "default".to_string(),
            shape: Shape::new(vec![Dim::Wildcard, Dim::Named("dim".to_string())]),
            variadic: false,
        }],
        outputs: vec![Port {
            name: "default".to_string(),
            shape: Shape::new(vec![Dim::Wildcard, Dim::Named("dim".to_string())]),
            variadic: false,
        }],
        max_cycle_depth: Some(10),
        doc: None,
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![Connection {
                source: Endpoint::Ref(PortRef::new("in")),
                destination: Endpoint::Match(MatchExpr {
                    subject: MatchSubject::Implicit,
                    arms: vec![
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![Dim::Wildcard, Dim::Literal(512)])),
                            guard: None,
                            pipeline: vec![
                                Endpoint::Call {
                                    name: "Identity".to_string(),
                                    args: vec![],
                                    kwargs: vec![],
                                    id: 0,
                                    frozen: false,
                                },
                                Endpoint::Ref(PortRef::new("out")),
                            ],
                            is_reachable: true,
                        },
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![Dim::Wildcard, Dim::Literal(256)])),
                            guard: None,
                            pipeline: vec![
                                Endpoint::Call {
                                    name: "Linear".to_string(),
                                    args: vec![Value::Int(256), Value::Int(512)],
                                    kwargs: vec![],
                                    id: 1,
                                    frozen: false,
                                },
                                Endpoint::Ref(PortRef::new("out")),
                            ],
                            is_reachable: true,
                        },
                    ],
                    id: 0,
                }),
            }],
        },
    };

    program.neurons.insert("MatchTest".to_string(), neuron);

    let code = generate_pytorch(&program, "MatchTest").unwrap();
    println!("{}", code);

    assert!(code.contains("if x.ndim == 2 and x.shape[1] == 512:"));
    assert!(code.contains("elif x.ndim == 2 and x.shape[1] == 256:"));
    // Note: IDs might vary depending on counter, but names should be consistent
    assert!(code.contains("self.identity_"));
    assert!(code.contains("self.linear_"));
}

#[test]
fn test_codegen_match_with_captured_dims() {
    // Test match expression with captured dimensions
    let mut program = Program::new();
    let neuron = NeuronDef {
        name: "DynamicMatch".to_string(),
        params: vec![],
        inputs: vec![Port {
            name: "default".to_string(),
            shape: Shape::new(vec![Dim::Wildcard, Dim::Wildcard]),
            variadic: false,
        }],
        outputs: vec![Port {
            name: "default".to_string(),
            shape: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]),
            variadic: false,
        }],
        max_cycle_depth: Some(10),
        doc: None,
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![Connection {
                source: Endpoint::Ref(PortRef::new("in")),
                destination: Endpoint::Match(MatchExpr {
                    subject: MatchSubject::Implicit,
                    arms: vec![
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())])),
                            guard: Some(Value::BinOp {
                                op: BinOp::Gt,
                                left: Box::new(Value::Name("d".to_string())),
                                right: Box::new(Value::Int(512)),
                            }),
                            pipeline: vec![
                                Endpoint::Call {
                                    name: "Linear".to_string(),
                                    args: vec![Value::Name("d".to_string()), Value::Int(512)],
                                    kwargs: vec![],
                                    id: 0,
                                    frozen: false,
                                },
                                Endpoint::Ref(PortRef::new("out")),
                            ],
                            is_reachable: true,
                        },
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())])),
                            guard: None,
                            pipeline: vec![
                                Endpoint::Call {
                                    name: "Linear".to_string(),
                                    args: vec![Value::Name("d".to_string()), Value::Int(256)],
                                    kwargs: vec![],
                                    id: 1,
                                    frozen: false,
                                },
                                Endpoint::Call {
                                    name: "Linear".to_string(),
                                    args: vec![Value::Int(256), Value::Int(512)],
                                    kwargs: vec![],
                                    id: 2,
                                    frozen: false,
                                },
                                Endpoint::Ref(PortRef::new("out")),
                            ],
                            is_reachable: true,
                        },
                    ],
                    id: 0,
                }),
            }],
        },
    };

    program.neurons.insert("DynamicMatch".to_string(), neuron);

    let code = generate_pytorch(&program, "DynamicMatch").unwrap();
    println!("{}", code);

    // Verify dimension binding is generated
    assert!(
        code.contains("d = x.shape[1]"),
        "Dimension binding should be generated"
    );

    // Verify guard condition includes the bound dimension (on separate line after binding)
    assert!(
        code.contains("if d > 512:"),
        "Guard should reference bound dimension"
    );

    // Verify lazy instantiation for modules with captured dimensions
    assert!(
        code.contains("self._linear_") && code.contains("= None"),
        "Should have lazy instantiation"
    );
    assert!(
        code.contains("if self._linear_") && code.contains("is None:"),
        "Should check for lazy instantiation"
    );
    assert!(
        code.contains("Linear(d,"),
        "Should instantiate Linear with captured dimension"
    );
}

#[test]
fn test_codegen_match_guards_with_bindings() {
    // Test guard expression that references captured dimension
    let mut program = Program::new();
    let neuron = NeuronDef {
        name: "GuardTest".to_string(),
        params: vec![],
        inputs: vec![Port {
            name: "default".to_string(),
            shape: Shape::new(vec![Dim::Wildcard, Dim::Wildcard]),
            variadic: false,
        }],
        outputs: vec![Port {
            name: "default".to_string(),
            shape: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]),
            variadic: false,
        }],
        max_cycle_depth: Some(10),
        doc: None,
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![Connection {
                source: Endpoint::Ref(PortRef::new("in")),
                destination: Endpoint::Match(MatchExpr {
                    subject: MatchSubject::Implicit,
                    arms: vec![MatchArm {
                        pattern: MatchPattern::Shape(Shape::new(vec![Dim::Wildcard, Dim::Named("dim".to_string())])),
                        guard: Some(Value::BinOp {
                            op: BinOp::Le,
                            left: Box::new(Value::Name("dim".to_string())),
                            right: Box::new(Value::Int(512)),
                        }),
                        pipeline: vec![
                            Endpoint::Call {
                                name: "Identity".to_string(),
                                args: vec![],
                                kwargs: vec![],
                                id: 0,
                                frozen: false,
                            },
                            Endpoint::Ref(PortRef::new("out")),
                        ],
                        is_reachable: true,
                    }],
                    id: 0,
                }),
            }],
        },
    };

    program.neurons.insert("GuardTest".to_string(), neuron);

    let code = generate_pytorch(&program, "GuardTest").unwrap();
    println!("{}", code);

    // Verify dimension is bound before being used in guard
    assert!(code.contains("dim = x.shape[1]"), "Dimension must be bound");
    assert!(
        code.contains("if dim <= 512:"),
        "Guard should use bound dimension"
    );
}

#[test]
fn test_codegen_optimized_match_fewer_branches() {
    // Test that optimizer removes unreachable arms, resulting in cleaner code
    use crate::optimizer::optimize_matches;

    // Create program where all arms start as REACHABLE (simulating before validator marks them)
    let create_reachable_program = || {
        let mut program = Program::new();
        let neuron = NeuronDef {
            name: "OptimizedMatch".to_string(),
            params: vec![],
            inputs: vec![Port {
                name: "default".to_string(),
                shape: Shape::new(vec![Dim::Wildcard, Dim::Wildcard]),
                variadic: false,
            }],
            outputs: vec![Port {
                name: "default".to_string(),
                shape: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]),
                variadic: false,
            }],
            max_cycle_depth: Some(10),
            doc: None,
            body: NeuronBody::Graph {
                context_bindings: vec![],
                context_unrolls: vec![],
                connections: vec![Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Match(MatchExpr {
                        subject: MatchSubject::Implicit,
                        arms: vec![
                            MatchArm {
                                pattern: MatchPattern::Shape(Shape::new(vec![
                                    Dim::Wildcard,
                                    Dim::Named("d".to_string()),
                                ])),
                                guard: None,
                                pipeline: vec![
                                    Endpoint::Call {
                                        name: "Linear".to_string(),
                                        args: vec![Value::Name("d".to_string()), Value::Int(512)],
                                        kwargs: vec![],
                                        id: 0,
                                        frozen: false,
                                    },
                                    Endpoint::Ref(PortRef::new("out")),
                                ],
                                is_reachable: true,
                            },
                            MatchArm {
                                pattern: MatchPattern::Shape(Shape::new(vec![Dim::Wildcard, Dim::Literal(512)])),
                                guard: None,
                                pipeline: vec![
                                    Endpoint::Call {
                                        name: "Identity".to_string(),
                                        args: vec![],
                                        kwargs: vec![],
                                        id: 1,
                                        frozen: false,
                                    },
                                    Endpoint::Ref(PortRef::new("out")),
                                ],
                                is_reachable: true, // All start reachable
                            },
                            MatchArm {
                                pattern: MatchPattern::Shape(Shape::new(vec![Dim::Wildcard, Dim::Literal(256)])),
                                guard: None,
                                pipeline: vec![
                                    Endpoint::Call {
                                        name: "Identity".to_string(),
                                        args: vec![],
                                        kwargs: vec![],
                                        id: 2,
                                        frozen: false,
                                    },
                                    Endpoint::Ref(PortRef::new("out")),
                                ],
                                is_reachable: true, // All start reachable
                            },
                        ],
                        id: 0,
                    }),
                }],
            },
        };
        program.neurons.insert("OptimizedMatch".to_string(), neuron);
        program
    };

    // Generate code with ALL arms reachable
    let all_reachable_program = create_reachable_program();
    let all_reachable_code = generate_pytorch(&all_reachable_program, "OptimizedMatch").unwrap();

    // Count branches when all arms are reachable
    let all_reachable_if_count = all_reachable_code.matches("if x.ndim").count();
    let all_reachable_elif_count = all_reachable_code.matches("elif x.ndim").count();

    // Now mark some as unreachable (simulating validator) and optimize
    let mut marked_program = create_reachable_program();
    // Mark the shadowed arms as unreachable (simulating validator output)
    if let NeuronBody::Graph { connections, .. } = &mut marked_program
        .neurons
        .get_mut("OptimizedMatch")
        .unwrap()
        .body
    {
        if let Endpoint::Match(match_expr) = &mut connections[0].destination {
            match_expr.arms[1].is_reachable = false; // [*, 512] shadowed by [*, d]
            match_expr.arms[2].is_reachable = false; // [*, 256] shadowed by [*, d]
        }
    }

    // Optimize (removes unreachable arms)
    let pruned = optimize_matches(&mut marked_program, true);
    assert_eq!(pruned, 2, "Should prune 2 shadowed arms");

    let optimized_code = generate_pytorch(&marked_program, "OptimizedMatch").unwrap();

    // Count branches after optimization
    let optimized_if_count = optimized_code.matches("if x.ndim").count();
    let optimized_elif_count = optimized_code.matches("elif x.ndim").count();

    println!("=== ALL ARMS REACHABLE (BEFORE VALIDATOR) ===");
    println!("{}", all_reachable_code);
    println!(
        "All reachable: {} if, {} elif = {} total",
        all_reachable_if_count,
        all_reachable_elif_count,
        all_reachable_if_count + all_reachable_elif_count
    );

    println!("\n=== AFTER OPTIMIZATION (PRUNED DEAD ARMS) ===");
    println!("{}", optimized_code);
    println!(
        "Optimized: {} if, {} elif = {} total",
        optimized_if_count,
        optimized_elif_count,
        optimized_if_count + optimized_elif_count
    );

    // Verify we had more branches before pruning (3 arms means multiple branches)
    assert!(
        all_reachable_if_count + all_reachable_elif_count >= 3,
        "Should have at least 3 branches when all arms are reachable (was {} if + {} elif = {})",
        all_reachable_if_count,
        all_reachable_elif_count,
        all_reachable_if_count + all_reachable_elif_count
    );

    // Verify we have only 1 branch after pruning
    assert_eq!(
        optimized_if_count, 1,
        "Should have only 1 if statement after pruning"
    );
    assert_eq!(
        optimized_elif_count, 0,
        "Should have no elif statements after pruning"
    );

    // Verify the Identity modules are NOT in optimized code (arms pruned)
    assert!(
        !optimized_code.contains("self.identity_"),
        "Pruned arms should not generate module instantiation"
    );
    assert!(
        !optimized_code.contains("Identity"),
        "Identity import should be removed"
    );

    // Verify the Linear module IS in optimized code
    assert!(
        optimized_code.contains("self._linear_"),
        "Reachable arm should generate module"
    );
    assert!(
        optimized_code.contains("from neuroscript_runtime.primitives.linear import Linear"),
        "Linear import should remain"
    );
}

#[test]
fn test_codegen_optimized_match_with_guards() {
    // Guards prevent pruning - verify both arms generate code
    use crate::optimizer::optimize_matches;

    let mut program = Program::new();
    let neuron = NeuronDef {
        name: "GuardedMatch".to_string(),
        params: vec![],
        inputs: vec![Port {
            name: "default".to_string(),
            shape: Shape::new(vec![Dim::Wildcard, Dim::Wildcard]),
            variadic: false,
        }],
        outputs: vec![Port {
            name: "default".to_string(),
            shape: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]),
            variadic: false,
        }],
        max_cycle_depth: Some(10),
        doc: None,
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![Connection {
                source: Endpoint::Ref(PortRef::new("in")),
                destination: Endpoint::Match(MatchExpr {
                    subject: MatchSubject::Implicit,
                    arms: vec![
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())])),
                            guard: Some(Value::BinOp {
                                op: BinOp::Gt,
                                left: Box::new(Value::Name("d".to_string())),
                                right: Box::new(Value::Int(512)),
                            }),
                            pipeline: vec![
                                Endpoint::Call {
                                    name: "Linear".to_string(),
                                    args: vec![Value::Name("d".to_string()), Value::Int(512)],
                                    kwargs: vec![],
                                    id: 0,
                                    frozen: false,
                                },
                                Endpoint::Ref(PortRef::new("out")),
                            ],
                            is_reachable: true,
                        },
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())])),
                            guard: None, // Catch-all for same pattern
                            pipeline: vec![
                                Endpoint::Call {
                                    name: "Identity".to_string(),
                                    args: vec![],
                                    kwargs: vec![],
                                    id: 1,
                                    frozen: false,
                                },
                                Endpoint::Ref(PortRef::new("out")),
                            ],
                            is_reachable: true, // Guard makes this reachable
                        },
                    ],
                    id: 0,
                }),
            }],
        },
    };

    program.neurons.insert("GuardedMatch".to_string(), neuron);

    // Optimize
    let pruned = optimize_matches(&mut program, true);
    assert_eq!(pruned, 0, "Guards prevent pruning");

    let code = generate_pytorch(&program, "GuardedMatch").unwrap();

    // Both branches should exist
    assert!(code.contains("if x.ndim == 2:"), "Should have if statement");
    assert!(code.contains("if d > 512:"), "Should have guard check");
    assert!(code.contains("else:"), "Should have else for fallback arm");

    // Both modules should be instantiated
    assert!(code.contains("self._linear_"), "First arm needs Linear");
    assert!(code.contains("self.identity_"), "Second arm needs Identity");

    println!("{}", code);
}

#[test]
fn test_codegen_roundtrip_parse_validate_optimize() {
    // Integration test: parse → validate → optimize → codegen
    use crate::{optimizer::optimize_matches, parse, validate};

    let source = r#"
neuron OptimizeDemo:
    in: [*, d]
    out: [*, 512]
    graph:
        in -> match: ->
            [*, d] where d > 512: Linear(d, 512) -> out
            [*, d]: Linear(d, 256) -> Linear(256, 512) -> out
            [*, 512]: Identity() -> out
"#;

    // Parse
    let mut program = parse(source).expect("Parse should succeed");

    // Validate (this marks unreachable arms)
    let validation_result = validate(&mut program);
    // Note: validator may error about shadowing, but for this test we continue
    if let Err(validation_errors) = validation_result {
        println!(
            "Validation errors (expected for shadowed arms): {} errors",
            validation_errors.len()
        );
        for err in &validation_errors {
            println!("  - {:?}", err);
        }
    }

    // Optimize
    let pruned = optimize_matches(&mut program, true);
    println!("Pruned {} arms", pruned);

    // Codegen
    let code = generate_pytorch(&program, "OptimizeDemo").expect("Codegen should succeed");

    println!("=== GENERATED CODE ===");
    println!("{}", code);

    // Verify the code is valid Python-like structure
    assert!(code.contains("class OptimizeDemo"));
    assert!(code.contains("def __init__"));
    assert!(code.contains("def forward"));

    // Should have shape checks
    assert!(code.contains("x.ndim == 2"));

    // Should have dimension binding
    assert!(code.contains("d = x.shape[1]"));
}

#[test]
fn test_codegen_if_else() {
    let mut program = Program::new();
    let neuron = NeuronDef {
        name: "IfTest".to_string(),
        params: vec![Param {
            name: "d".to_string(),
            default: Some(Value::Int(64)),
            type_annotation: None,
        }],
        inputs: vec![Port {
            name: "default".to_string(),
            shape: Shape::new(vec![Dim::Wildcard, Dim::Literal(64)]),
            variadic: false,
        }],
        outputs: vec![Port {
            name: "default".to_string(),
            shape: Shape::new(vec![Dim::Wildcard, Dim::Wildcard]), // Simplified output shape
            variadic: false,
        }],
        max_cycle_depth: Some(10),
        doc: None,
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![Connection {
                source: Endpoint::Ref(PortRef::new("in")),
                destination: Endpoint::If(IfExpr {
                    branches: vec![
                        IfBranch {
                            condition: Value::BinOp {
                                op: BinOp::Gt,
                                left: Box::new(Value::Name("d".to_string())),
                                right: Box::new(Value::Int(512)),
                            },
                            pipeline: vec![
                                Endpoint::Call {
                                    name: "Linear".to_string(),
                                    args: vec![Value::Name("d".to_string()), Value::Int(512)],
                                    kwargs: vec![],
                                    id: 0,
                                    frozen: false,
                                },
                                Endpoint::Ref(PortRef::new("out")),
                            ],
                        },
                        IfBranch {
                            condition: Value::BinOp {
                                op: BinOp::Eq,
                                left: Box::new(Value::Name("d".to_string())),
                                right: Box::new(Value::Int(256)),
                            },
                            pipeline: vec![
                                Endpoint::Call {
                                    name: "Linear".to_string(),
                                    args: vec![Value::Int(256), Value::Int(512)],
                                    kwargs: vec![],
                                    id: 1,
                                    frozen: false,
                                },
                                Endpoint::Ref(PortRef::new("out")),
                            ],
                        },
                    ],
                    else_branch: Some(vec![
                        Endpoint::Call {
                            name: "Linear".to_string(),
                            args: vec![Value::Name("d".to_string()), Value::Int(512)],
                            kwargs: vec![],
                            id: 2,
                            frozen: false,
                        },
                        Endpoint::Ref(PortRef::new("out")),
                    ]),
                    id: 0,
                }),
            }],
        },
    };

    program.neurons.insert("IfTest".to_string(), neuron);

    let code = generate_pytorch(&program, "IfTest").unwrap();
    println!("{}", code);

    assert!(code.contains("if self.d > 512:"));
    assert!(code.contains("elif self.d == 256:"));
    assert!(code.contains("else:"));

    // Check linear instantiations
    // Since 'd' is a parameter, it's statically resolvable in __init__, so we expect static instantiation:
    assert!(code.contains("self.linear_"));
}

// ============================================================================
// Unroll codegen integration tests (parse → expand → validate → codegen)
// ============================================================================

#[test]
fn test_codegen_unroll_threaded() {
    let source = include_str!("../../examples/unroll_threaded.ns");
    let mut program = parse(source).expect("Parse should succeed");
    if let Ok(stdlib) = crate::stdlib::load_stdlib() {
        // Use merge_programs so user neurons take priority over stdlib.
        // The example defines TransformerStack which also exists in stdlib;
        // extend() would clobber the example's definition.
        program = crate::stdlib::merge_programs(stdlib, program);
    }
    validate(&mut program).expect("Validation should succeed");
    let code = generate_pytorch(&program, "TransformerStack").expect("Codegen should succeed");

    // Should use nn.ModuleList for unrolled blocks
    assert!(
        code.contains("self.blocks = nn.ModuleList(["),
        "Should use nn.ModuleList for unrolled blocks"
    );
    assert!(
        code.contains("TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)"),
        "Should use range(num_layers) in comprehension"
    );

    // Forward should use a for loop
    assert!(
        code.contains("for block in self.blocks:"),
        "Should iterate over module list"
    );
    assert!(
        code.contains("x = block(x)"),
        "Should apply each block in-place"
    );
}

#[test]
fn test_codegen_unroll_context() {
    let source = include_str!("../../examples/unroll_context.ns");
    let mut program = parse(source).expect("Parse should succeed");
    if let Ok(stdlib) = crate::stdlib::load_stdlib() {
        program = crate::stdlib::merge_programs(stdlib, program);
    }
    validate(&mut program).expect("Validation should succeed");
    let code = generate_pytorch(&program, "NamedStack").expect("Codegen should succeed");

    // Should use nn.ModuleList for unrolled blocks
    assert!(
        code.contains("self.blocks = nn.ModuleList(["),
        "Should use nn.ModuleList for blocks"
    );
    assert!(
        code.contains("TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)"),
        "Should use range(num_layers) in comprehension"
    );

    // Forward should use a for loop
    assert!(
        code.contains("for block in self.blocks:"),
        "Should iterate over blocks"
    );
    assert!(
        code.contains("x = block(x)"),
        "Should apply each block in-place"
    );
}

#[test]
fn test_codegen_unroll_static() {
    let source = include_str!("../../examples/unroll_static.ns");
    let mut program = parse(source).expect("Parse should succeed");
    if let Ok(stdlib) = crate::stdlib::load_stdlib() {
        program = crate::stdlib::merge_programs(stdlib, program);
    }
    validate(&mut program).expect("Validation should succeed");
    let code = generate_pytorch(&program, "SharedLayers").expect("Codegen should succeed");

    // Should have exactly ONE class-level module (conditional instantiation)
    assert!(
        code.contains("if not hasattr(self.__class__, 'block'):"),
        "Should check for existing class-level block"
    );
    assert!(
        code.contains("self.__class__.block = TransformerBlock(d_model, num_heads, d_ff)"),
        "Should instantiate shared block at class level"
    );

    // Should have a for loop calling the shared instance N times
    assert!(
        code.contains("for _ in range(self.num_layers):"),
        "Should iterate num_layers times"
    );
    assert!(
        code.contains("x = self.__class__.block(x)"),
        "Should call shared class-level block in loop"
    );

    // Should NOT have suffixed instances or nn.ModuleList
    assert!(!code.contains("block_0"), "Static block should not be suffixed");
    assert!(!code.contains("block_1"), "Static block should not be suffixed");
    assert!(!code.contains("nn.ModuleList"), "Static unroll should not use ModuleList");
}

#[test]
fn test_codegen_unroll_gpt2() {
    let source = include_str!("../../examples/unroll_gpt2.ns");
    let mut program = parse(source).expect("Parse should succeed");

    // Load stdlib neurons since the example file references TransformerBlock etc.
    if let Ok(stdlib) = crate::stdlib::load_stdlib() {
        program = crate::stdlib::merge_programs(stdlib, program);
    }

    validate(&mut program).expect("Validation should succeed");
    let code = generate_pytorch(&program, "GPT2Small").expect("Codegen should succeed");

    // Should have class definition
    assert!(code.contains("class GPT2Small(nn.Module)"));

    // Should use nn.ModuleList for unrolled blocks
    assert!(
        code.contains("self.blocks = nn.ModuleList(["),
        "Should use nn.ModuleList for blocks"
    );
    assert!(
        code.contains("TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)"),
        "Should use range(num_layers) in comprehension"
    );

    // Non-unrolled bindings should each appear exactly once in __init__
    assert_eq!(
        code.matches("self.embed = Embedding(").count(), 1,
        "embed should appear once"
    );
    assert_eq!(
        code.matches("self.ln_f = LayerNorm(").count(), 1,
        "ln_f should appear once"
    );
    assert_eq!(
        code.matches("self.head = Linear(").count(), 1,
        "head should appear once"
    );

    // Forward should use semantic variable names and for loop
    assert!(code.contains("self.embed(x)"), "Should start with embed");
    assert!(code.contains("for block in self.blocks:"), "Should iterate over blocks");
    assert!(code.contains("embed = block(embed)"), "Should apply block to embed");
    assert!(code.contains("self.ln_f(embed)"), "Should include ln_f after blocks");
    assert!(code.contains("self.head(ln_f)"), "Should include head after ln_f");
}

// ============================================================================
// Fat arrow reshape codegen tests
// ============================================================================

#[test]
fn test_codegen_fat_arrow_reshape_basic() {
    let source = r#"
neuron ReshapeTest(dim, heads):
  in: [batch, seq, dim]
  out: [batch, heads, seq, dim]
  graph:
    in => [batch, seq, heads, dh] -> out
"#;
    let mut program = parse(source).unwrap();
    validate(&mut program).unwrap();
    let result = generate_pytorch(&program, "ReshapeTest");
    assert!(result.is_ok(), "codegen should succeed: {:?}", result);
    let code = result.unwrap();
    println!("=== RESHAPE BASIC ===\n{}", code);
    assert!(
        code.contains(".reshape("),
        "should contain reshape call"
    );
    assert!(
        code.contains("class ReshapeTest(nn.Module)"),
        "should have class definition"
    );
}

#[test]
fn test_codegen_fat_arrow_reshape_with_binding() {
    let source = r#"
neuron MultiHeadReshape(dim, heads):
  in: [batch, seq, dim]
  out: [batch, heads, seq, dim]
  graph:
    in => [batch, seq, heads, dh=dim/heads] -> out
"#;
    let mut program = parse(source).unwrap();
    validate(&mut program).unwrap();
    let result = generate_pytorch(&program, "MultiHeadReshape");
    assert!(result.is_ok(), "codegen should succeed: {:?}", result);
    let code = result.unwrap();
    println!("=== RESHAPE WITH BINDING ===\n{}", code);
    assert!(
        code.contains(".reshape("),
        "should contain reshape call"
    );
    assert!(
        code.contains("dh = self.dim // self.heads"),
        "should contain binding assignment with integer division and self-prefixed params"
    );
}

#[test]
fn test_codegen_fat_arrow_chained() {
    let source = r#"
neuron TransposeHeads(dim, heads):
  in: [batch, seq, dim]
  out: [batch, heads, seq, dh]
  graph:
    in => [batch, seq, heads, dh] => [batch, heads, seq, dh] -> out
"#;
    let mut program = parse(source).unwrap();
    validate(&mut program).unwrap();
    let result = generate_pytorch(&program, "TransposeHeads");
    assert!(result.is_ok(), "codegen should succeed: {:?}", result);
    let code = result.unwrap();
    println!("=== CHAINED RESHAPE ===\n{}", code);
    // Should have two reshape calls (one for each =>)
    let reshape_count = code.matches(".reshape(").count();
    assert!(
        reshape_count >= 2,
        "should have at least 2 reshape calls for chained fat arrows, got {}",
        reshape_count
    );
}

#[test]
fn test_codegen_fat_arrow_reduce() {
    let source = r#"
neuron GlobalAvgPool(dim):
  in: [batch, seq, dim]
  out: [batch, dim]
  graph:
    in => @reduce(mean) [batch, dim] -> out
"#;
    let mut program = parse(source).unwrap();
    validate(&mut program).unwrap();
    let result = generate_pytorch(&program, "GlobalAvgPool");
    assert!(result.is_ok(), "codegen should succeed: {:?}", result);
    let code = result.unwrap();
    println!("=== REDUCE MEAN ===\n{}", code);
    assert!(
        code.contains(".mean(dim="),
        "should contain .mean(dim= reduction call"
    );
}

#[test]
fn test_codegen_fat_arrow_reduce_sum() {
    let source = r#"
neuron SumPool(dim):
  in: [batch, seq, dim]
  out: [batch, dim]
  graph:
    in => @reduce(sum) [batch, dim] -> out
"#;
    let mut program = parse(source).unwrap();
    validate(&mut program).unwrap();
    let result = generate_pytorch(&program, "SumPool");
    assert!(result.is_ok(), "codegen should succeed: {:?}", result);
    let code = result.unwrap();
    println!("=== REDUCE SUM ===\n{}", code);
    assert!(
        code.contains(".sum(dim="),
        "should contain .sum(dim= reduction call"
    );
}

#[test]
fn test_codegen_fat_arrow_repeat_copy() {
    let source = r#"
neuron BroadcastTest(dim):
  in: [batch, dim]
  out: [batch, seq, dim]
  graph:
    in => @repeat(copy) [batch, seq, dim] -> out
"#;
    let mut program = parse(source).unwrap();
    validate(&mut program).unwrap();
    let result = generate_pytorch(&program, "BroadcastTest");
    assert!(result.is_ok(), "codegen should succeed: {:?}", result);
    let code = result.unwrap();
    println!("=== REPEAT COPY ===\n{}", code);
    assert!(
        code.contains(".expand("),
        "should contain expand call for copy repeat"
    );
}

#[test]
fn test_codegen_fat_arrow_reduce_min() {
    let source = r#"
neuron MinPool(dim):
  in: [batch, seq, dim]
  out: [batch, dim]
  graph:
    in => @reduce(min) [batch, dim] -> out
"#;
    let mut program = parse(source).unwrap();
    validate(&mut program).unwrap();
    let result = generate_pytorch(&program, "MinPool");
    assert!(result.is_ok(), "codegen should succeed: {:?}", result);
    let code = result.unwrap();
    println!("=== REDUCE MIN ===\n{}", code);
    assert!(
        code.contains(".amin("),
        "should contain .amin() call (not .min() which returns (values, indices))"
    );
}

#[test]
fn test_codegen_fat_arrow_reduce_max() {
    let source = r#"
neuron MaxPool(dim):
  in: [batch, seq, dim]
  out: [batch, dim]
  graph:
    in => @reduce(max) [batch, dim] -> out
"#;
    let mut program = parse(source).unwrap();
    validate(&mut program).unwrap();
    let result = generate_pytorch(&program, "MaxPool");
    assert!(result.is_ok(), "codegen should succeed: {:?}", result);
    let code = result.unwrap();
    println!("=== REDUCE MAX ===\n{}", code);
    assert!(
        code.contains(".amax("),
        "should contain .amax() call (not .max() which returns (values, indices))"
    );
}

#[test]
fn test_codegen_fat_arrow_reshape_in_match_arm() {
    // Construct IR directly to test reshape inside a match arm pipeline.
    // This exercises the arm_reshape_src resolution code path in codegen.
    let mut program = Program::new();
    let neuron = NeuronDef {
        name: "MatchReshape".to_string(),
        params: vec![
            Param { name: "dim".to_string(), default: None, type_annotation: None },
            Param { name: "heads".to_string(), default: None, type_annotation: None },
        ],
        inputs: vec![Port {
            name: "default".to_string(),
            shape: Shape::new(vec![Dim::Wildcard, Dim::Named("seq".to_string()), Dim::Named("dim".to_string())]),
            variadic: false,
        }],
        outputs: vec![Port {
            name: "default".to_string(),
            shape: Shape::new(vec![Dim::Wildcard, Dim::Named("heads".to_string()), Dim::Named("seq".to_string()), Dim::Named("dim".to_string())]),
            variadic: false,
        }],
        max_cycle_depth: Some(10),
        doc: None,
        body: NeuronBody::Graph {
            context_bindings: vec![],
            context_unrolls: vec![],
            connections: vec![Connection {
                source: Endpoint::Ref(PortRef::new("in")),
                destination: Endpoint::Match(MatchExpr {
                    subject: MatchSubject::Implicit,
                    arms: vec![
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![
                                Dim::Wildcard,
                                Dim::Named("seq".to_string()),
                                Dim::Named("d".to_string()),
                            ])),
                            guard: Some(Value::BinOp {
                                op: BinOp::Gt,
                                left: Box::new(Value::Name("d".to_string())),
                                right: Box::new(Value::Int(256)),
                            }),
                            pipeline: vec![
                                Endpoint::Reshape(ReshapeExpr {
                                    dims: vec![
                                        ReshapeDim::Named("batch".to_string()),
                                        ReshapeDim::Named("seq".to_string()),
                                        ReshapeDim::Named("heads".to_string()),
                                        ReshapeDim::Binding {
                                            name: "dh".to_string(),
                                            expr: Box::new(Value::BinOp {
                                                op: BinOp::Div,
                                                left: Box::new(Value::Name("dim".to_string())),
                                                right: Box::new(Value::Name("heads".to_string())),
                                            }),
                                        },
                                    ],
                                    annotation: None,
                                    id: 10,
                                }),
                                Endpoint::Ref(PortRef::new("out")),
                            ],
                            is_reachable: true,
                        },
                        MatchArm {
                            pattern: MatchPattern::Shape(Shape::new(vec![
                                Dim::Wildcard,
                                Dim::Named("seq".to_string()),
                                Dim::Named("d".to_string()),
                            ])),
                            guard: None,
                            pipeline: vec![
                                Endpoint::Call {
                                    name: "Linear".to_string(),
                                    args: vec![Value::Name("d".to_string()), Value::Name("dim".to_string())],
                                    kwargs: vec![],
                                    id: 11,
                                    frozen: false,
                                },
                                Endpoint::Ref(PortRef::new("out")),
                            ],
                            is_reachable: true,
                        },
                    ],
                    id: 0,
                }),
            }],
        },
    };

    program.neurons.insert("MatchReshape".to_string(), neuron);

    let result = generate_pytorch(&program, "MatchReshape");
    assert!(result.is_ok(), "codegen should succeed: {:?}", result);
    let code = result.unwrap();
    println!("=== RESHAPE IN MATCH ARM ===\n{}", code);
    assert!(
        code.contains(".reshape("),
        "should contain reshape call inside match arm"
    );
    assert!(
        code.contains("class MatchReshape(nn.Module)"),
        "should have class definition"
    );
}

#[test]
fn test_higher_order_neuron_passthrough() {
    let source = r#"
neuron Wrapper(layer: Neuron, dim):
    in: [*, dim]
    out: [*, dim]
    context:
        inner = layer
    graph:
        in -> inner -> out
"#;
    let mut program = parse(source).unwrap();
    validate(&mut program).unwrap();
    let code = generate_pytorch(&program, "Wrapper").unwrap();
    assert!(
        code.contains("self.inner = layer"),
        "Expected pass-through assignment, got:\n{}",
        code
    );
    assert!(
        !code.contains("from neuroscript_runtime"),
        "Should not generate import for neuron param:\n{}",
        code
    );
}

#[test]
fn test_higher_order_neuron_construct() {
    let source = r#"
neuron Wrapper(layer: Neuron, dim):
    in: [*, dim]
    out: [*, dim]
    context:
        inner = layer(dim)
    graph:
        in -> inner -> out
"#;
    let mut program = parse(source).unwrap();
    validate(&mut program).unwrap();
    let code = generate_pytorch(&program, "Wrapper").unwrap();
    assert!(
        code.contains("self.inner = layer(dim)"),
        "Expected construct-from-type, got:\n{}",
        code
    );
    assert!(
        !code.contains("from neuroscript_runtime"),
        "Should not generate import for neuron param:\n{}",
        code
    );
}

#[test]
fn test_wrap_ref_codegen() {
    let source = r#"
neuron SimpleWrapper(layer: Neuron, dim):
    in: [*, dim]
    out: [*, dim]
    graph:
        in -> layer -> out

neuron Test(dim):
    in: [*, dim]
    out: [*, dim]
    context:
        attn = MultiHeadSelfAttention(dim, 8)
    graph:
        in -> @wrap(SimpleWrapper, dim): attn -> out
"#;
    let mut program = parse(source).unwrap();
    validate(&mut program).unwrap();
    let code = generate_pytorch(&program, "Test").unwrap();
    // After desugaring, @wrap(SimpleWrapper, dim): attn
    // becomes SimpleWrapper(attn, dim), which is a Call endpoint
    assert!(
        code.contains("SimpleWrapper"),
        "Expected SimpleWrapper wrapper call, got:\n{}",
        code
    );
}

#[test]
fn test_wrap_pipeline_codegen() {
    let source = r#"
neuron SimpleWrapper(layer: Neuron, dim):
    in: [*, dim]
    out: [*, dim]
    graph:
        in -> layer -> out

neuron Test(dim):
    in: [*, dim]
    out: [*, dim]
    graph:
        in ->
            @wrap(SimpleWrapper, dim): ->
                LayerNorm(dim)
                Linear(dim, dim)
            out
"#;
    let mut program = parse(source).unwrap();
    validate(&mut program).unwrap();
    let code = generate_pytorch(&program, "Test").unwrap();
    // After desugaring, the pipeline form should create nn.Sequential
    assert!(
        code.contains("nn.Sequential"),
        "Expected nn.Sequential for pipeline form, got:\n{}",
        code
    );
    assert!(
        code.contains("SimpleWrapper"),
        "Expected SimpleWrapper wrapper call, got:\n{}",
        code
    );
}
