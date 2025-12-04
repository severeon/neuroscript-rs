//! PyTorch code generation from NeuroScript IR
//!
//! This module generates Python code that implements NeuroScript neurons
//! as PyTorch nn.Module classes.
//!
//! # Example
//!
//! ```ignore
//! use neuroscript::parse;
//! use neuroscript::codegen::generate_pytorch;
//!
//! let program = parse("neuron MLP(dim): ...")?;
//! let code = generate_pytorch(&program, "MLP")?;
//! println!("{}", code);
//! ```

// Module organization
pub mod generator;
pub mod instantiation;
pub mod forward;
pub mod utils;

// Re-exports for public API
pub use crate::interfaces::CodegenError;
pub use generator::generate_pytorch;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interfaces::*;

    #[test]
    fn test_codegen_match() {
        // Construct a simple program with a match expression
        let mut program = Program::new();
        let neuron = NeuronDef {
            name: "MatchTest".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Named("dim".to_string())]) }],
            outputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Named("dim".to_string())]) }],
            body: NeuronBody::Graph {
                let_bindings: vec![],
                set_bindings: vec![],
                connections: vec![
                    Connection {
                        source: Endpoint::Ref(PortRef::new("in")),
                        destination: Endpoint::Match(MatchExpr {
                            arms: vec![
                                MatchArm {
                                    pattern: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]),
                                    guard: None,
                                    pipeline: vec![
                                        Endpoint::Call { name: "Identity".to_string(), args: vec![], kwargs: vec![], id: 0 },
                                        Endpoint::Ref(PortRef::new("out"))
                                    ],
                                    is_reachable: true,
                                },
                                MatchArm {
                                    pattern: Shape::new(vec![Dim::Wildcard, Dim::Literal(256)]),
                                    guard: None,
                                    pipeline: vec![
                                        Endpoint::Call { name: "Linear".to_string(), args: vec![Value::Int(256), Value::Int(512)], kwargs: vec![], id: 1 },
                                        Endpoint::Ref(PortRef::new("out"))
                                    ],
                                    is_reachable: true,
                                }
                            ]
                        })
                    }
                ]
            }
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
            inputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Wildcard]) }],
            outputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]) }],
            body: NeuronBody::Graph {
                let_bindings: vec![],
                set_bindings: vec![],
                connections: vec![
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Match(MatchExpr {
                        arms: vec![
                            MatchArm {
                                pattern: Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]),
                                guard: Some(Value::BinOp {
                                    op: BinOp::Gt,
                                    left: Box::new(Value::Name("d".to_string())),
                                    right: Box::new(Value::Int(512))
                                }),
                                pipeline: vec![
                                    Endpoint::Call {
                                        name: "Linear".to_string(),
                                        args: vec![Value::Name("d".to_string()), Value::Int(512)],
                                        kwargs: vec![],
                                        id: 0
                                    },
                                    Endpoint::Ref(PortRef::new("out"))
                                ],
                                is_reachable: true,
                            },
                            MatchArm {
                                pattern: Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]),
                                guard: None,
                                pipeline: vec![
                                    Endpoint::Call {
                                        name: "Linear".to_string(),
                                        args: vec![Value::Name("d".to_string()), Value::Int(256)],
                                        kwargs: vec![],
                                        id: 1
                                    },
                                    Endpoint::Call {
                                        name: "Linear".to_string(),
                                        args: vec![Value::Int(256), Value::Int(512)],
                                        kwargs: vec![],
                                        id: 2
                                    },
                                    Endpoint::Ref(PortRef::new("out"))
                                ],
                                is_reachable: true,
                            }
                        ]
                    })
                }
            ]
            }
        };

        program.neurons.insert("DynamicMatch".to_string(), neuron);

        let code = generate_pytorch(&program, "DynamicMatch").unwrap();
        println!("{}", code);

        // Verify dimension binding is generated
        assert!(code.contains("d = x.shape[1]"), "Dimension binding should be generated");

        // Verify guard condition includes the bound dimension (on separate line after binding)
        assert!(code.contains("if d > 512:"), "Guard should reference bound dimension");

        // Verify lazy instantiation for modules with captured dimensions
        assert!(code.contains("self._linear_") && code.contains("= None"), "Should have lazy instantiation");
        assert!(code.contains("if self._linear_") && code.contains("is None:"), "Should check for lazy instantiation");
        assert!(code.contains("Linear(d,"), "Should instantiate Linear with captured dimension");
    }

    #[test]
    fn test_codegen_match_guards_with_bindings() {
        // Test guard expression that references captured dimension
        let mut program = Program::new();
        let neuron = NeuronDef {
            name: "GuardTest".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Wildcard]) }],
            outputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]) }],
            body: NeuronBody::Graph {
                let_bindings: vec![],
                set_bindings: vec![],
                connections: vec![
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Match(MatchExpr {
                        arms: vec![
                            MatchArm {
                                pattern: Shape::new(vec![Dim::Wildcard, Dim::Named("dim".to_string())]),
                                guard: Some(Value::BinOp {
                                    op: BinOp::Le,
                                    left: Box::new(Value::Name("dim".to_string())),
                                    right: Box::new(Value::Int(512))
                                }),
                                pipeline: vec![
                                    Endpoint::Call { name: "Identity".to_string(), args: vec![], kwargs: vec![], id: 0 },
                                    Endpoint::Ref(PortRef::new("out"))
                                ],
                                is_reachable: true,
                            }
                        ]
                    })
                }
            ]}
        };

        program.neurons.insert("GuardTest".to_string(), neuron);

        let code = generate_pytorch(&program, "GuardTest").unwrap();
        println!("{}", code);

        // Verify dimension is bound before being used in guard
        assert!(code.contains("dim = x.shape[1]"), "Dimension must be bound");
        assert!(code.contains("if dim <= 512:"), "Guard should use bound dimension");
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
                inputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Wildcard]) }],
                outputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]) }],
                body: NeuronBody::Graph {
                let_bindings: vec![],
                set_bindings: vec![],
                connections: vec![
                    Connection {
                        source: Endpoint::Ref(PortRef::new("in")),
                        destination: Endpoint::Match(MatchExpr {
                            arms: vec![
                                MatchArm {
                                    pattern: Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]),
                                    guard: None,
                                    pipeline: vec![
                                        Endpoint::Call {
                                            name: "Linear".to_string(),
                                            args: vec![Value::Name("d".to_string()), Value::Int(512)],
                                            kwargs: vec![],
                                            id: 0
                                        },
                                        Endpoint::Ref(PortRef::new("out"))
                                    ],
                                    is_reachable: true,
                                },
                                MatchArm {
                                    pattern: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]),
                                    guard: None,
                                    pipeline: vec![
                                        Endpoint::Call { name: "Identity".to_string(), args: vec![], kwargs: vec![], id: 1 },
                                        Endpoint::Ref(PortRef::new("out"))
                                    ],
                                    is_reachable: true, // All start reachable
                                },
                                MatchArm {
                                    pattern: Shape::new(vec![Dim::Wildcard, Dim::Literal(256)]),
                                    guard: None,
                                    pipeline: vec![
                                        Endpoint::Call { name: "Identity".to_string(), args: vec![], kwargs: vec![], id: 2 },
                                        Endpoint::Ref(PortRef::new("out"))
                                    ],
                                    is_reachable: true, // All start reachable
                                }
                            ]
                        })
                    }
                ]}
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
        if let NeuronBody::Graph { connections, .. } = &mut marked_program.neurons.get_mut("OptimizedMatch").unwrap().body {
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
        println!("All reachable: {} if, {} elif = {} total", all_reachable_if_count, all_reachable_elif_count, all_reachable_if_count + all_reachable_elif_count);

        println!("\n=== AFTER OPTIMIZATION (PRUNED DEAD ARMS) ===");
        println!("{}", optimized_code);
        println!("Optimized: {} if, {} elif = {} total", optimized_if_count, optimized_elif_count, optimized_if_count + optimized_elif_count);

        // Verify we had more branches before pruning (3 arms means multiple branches)
        assert!(
            all_reachable_if_count + all_reachable_elif_count >= 3,
            "Should have at least 3 branches when all arms are reachable (was {} if + {} elif = {})",
            all_reachable_if_count, all_reachable_elif_count, all_reachable_if_count + all_reachable_elif_count
        );

        // Verify we have only 1 branch after pruning
        assert_eq!(optimized_if_count, 1, "Should have only 1 if statement after pruning");
        assert_eq!(optimized_elif_count, 0, "Should have no elif statements after pruning");

        // Verify the Identity modules are NOT in optimized code (arms pruned)
        assert!(!optimized_code.contains("self.identity_"), "Pruned arms should not generate module instantiation");
        assert!(!optimized_code.contains("Identity"), "Identity import should be removed");

        // Verify the Linear module IS in optimized code
        assert!(optimized_code.contains("self._linear_"), "Reachable arm should generate module");
        assert!(optimized_code.contains("from neuroscript_runtime.primitives.linear import Linear"), "Linear import should remain");
    }

    #[test]
    fn test_codegen_optimized_match_with_guards() {
        // Guards prevent pruning - verify both arms generate code
        use crate::optimizer::optimize_matches;

        let mut program = Program::new();
        let neuron = NeuronDef {
            name: "GuardedMatch".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Wildcard]) }],
            outputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]) }],
            body: NeuronBody::Graph {
                let_bindings: vec![],
                set_bindings: vec![],
                connections: vec![
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Match(MatchExpr {
                        arms: vec![
                            MatchArm {
                                pattern: Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]),
                                guard: Some(Value::BinOp {
                                    op: BinOp::Gt,
                                    left: Box::new(Value::Name("d".to_string())),
                                    right: Box::new(Value::Int(512))
                                }),
                                pipeline: vec![
                                    Endpoint::Call {
                                        name: "Linear".to_string(),
                                        args: vec![Value::Name("d".to_string()), Value::Int(512)],
                                        kwargs: vec![],
                                        id: 0
                                    },
                                    Endpoint::Ref(PortRef::new("out"))
                                ],
                                is_reachable: true,
                            },
                            MatchArm {
                                pattern: Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]),
                                guard: None, // Catch-all for same pattern
                                pipeline: vec![
                                    Endpoint::Call { name: "Identity".to_string(), args: vec![], kwargs: vec![], id: 1 },
                                    Endpoint::Ref(PortRef::new("out"))
                                ],
                                is_reachable: true, // Guard makes this reachable
                            }
                        ]
                    })
                }
            ]}
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
        use crate::{parse, validate, optimizer::optimize_matches};

        let source = r#"
neuron OptimizeDemo:
  in: [*, d]
  out: [*, 512]
  graph:
    in -> match:
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
            println!("Validation errors (expected for shadowed arms): {} errors", validation_errors.len());
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
}
