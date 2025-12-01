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
    use crate::ir::*;

    #[test]
    fn test_codegen_match() {
        // Construct a simple program with a match expression
        let mut program = Program::new();
        let neuron = NeuronDef {
            name: "MatchTest".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Named("dim".to_string())]) }],
            outputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Named("dim".to_string())]) }],
            body: NeuronBody::Graph(vec![
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
            ])
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
            body: NeuronBody::Graph(vec![
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
            ])
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
            body: NeuronBody::Graph(vec![
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
            ])
        };

        program.neurons.insert("GuardTest".to_string(), neuron);

        let code = generate_pytorch(&program, "GuardTest").unwrap();
        println!("{}", code);

        // Verify dimension is bound before being used in guard
        assert!(code.contains("dim = x.shape[1]"), "Dimension must be bound");
        assert!(code.contains("if dim <= 512:"), "Guard should use bound dimension");
    }
}
