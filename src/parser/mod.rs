//! NeuroScript Parser
//!
//! Recursive descent parser with good error messages.

// Module organization
pub mod core;

// Re-exports for public API
pub use crate::interfaces::Parser;

#[cfg(test)]
mod tests {
    use crate::interfaces::*;

    // ===== Use Statement Tests =====

    #[test]
    fn test_parse_use_simple() {
        let source = "use core,nn/*\n";
        let program = Parser::parse(source).unwrap();
        assert_eq!(program.uses.len(), 1);
        assert_eq!(program.uses[0].source, "core");
        assert_eq!(program.uses[0].path, vec!["nn", "*"]);
    }

    #[test]
    fn test_parse_use_nested_path() {
        let source = "use stdlib,blocks/attention/MultiHead\n";
        let program = Parser::parse(source).unwrap();
        assert_eq!(program.uses.len(), 1);
        assert_eq!(program.uses[0].source, "stdlib");
        assert_eq!(program.uses[0].path, vec!["blocks", "attention", "MultiHead"]);
    }

    #[test]
    fn test_parse_multiple_uses() {
        let source = r#"
use core,nn/*
use stdlib,blocks/*
"#;
        let program = Parser::parse(source).unwrap();
        assert_eq!(program.uses.len(), 2);
    }

    // ===== Neuron Definition Tests =====

    #[test]
    fn test_parse_simple_neuron() {
        let source = r#"
neuron Linear(in_dim, out_dim):
  in: [*, in_dim]
  out: [*, out_dim]
  impl: core,nn/Linear
"#;
        let program = Parser::parse(source).unwrap();
        assert_eq!(program.neurons.len(), 1);
        assert!(program.neurons.contains_key("Linear"));
        
        let linear = &program.neurons["Linear"];
        assert_eq!(linear.name, "Linear");
        assert_eq!(linear.params.len(), 2);
        assert_eq!(linear.inputs.len(), 1);
        assert_eq!(linear.outputs.len(), 1);
    }

    #[test]
    fn test_parse_neuron_no_params() {
        let source = r#"
neuron Identity:
  in: [*shape]
  out: [*shape]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let identity = &program.neurons["Identity"];
        assert_eq!(identity.params.len(), 0);
    }

    #[test]
    fn test_parse_neuron_with_defaults() {
        let source = r#"
neuron Conv(in_ch, out_ch, kernel=3, stride=1):
  in: [*, in_ch, *, *]
  out: [*, out_ch, *, *]
  impl: core,nn/Conv2d
"#;
        let program = Parser::parse(source).unwrap();
        let conv = &program.neurons["Conv"];
        assert_eq!(conv.params.len(), 4);
        assert!(conv.params[0].default.is_none());
        assert!(conv.params[1].default.is_none());
        assert!(conv.params[2].default.is_some());
        assert!(conv.params[3].default.is_some());
    }

    #[test]
    fn test_duplicate_neuron_error() {
        let source = r#"
neuron Foo:
  in: [*x]
  out: [*x]
  impl: core,builtin/Identity

neuron Foo:
  in: [*y]
  out: [*y]
  impl: core,builtin/Identity
"#;
        let result = Parser::parse(source);
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::DuplicateNeuron { name, .. } => assert_eq!(name, "Foo"),
            _ => panic!("Expected DuplicateNeuron error"),
        }
    }

    // ===== Port Tests =====

    #[test]
    fn test_parse_named_ports() {
        let source = r#"
neuron MultiPort:
  in query: [*, dim]
  in key: [*, dim]
  out attention: [*, dim]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["MultiPort"];
        assert_eq!(neuron.inputs.len(), 2);
        assert_eq!(neuron.inputs[0].name, "query");
        assert_eq!(neuron.inputs[1].name, "key");
        assert_eq!(neuron.outputs[0].name, "attention");
    }

    #[test]
    fn test_parse_default_port_name() {
        let source = r#"
neuron Simple:
  in: [*, dim]
  out: [*, dim]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Simple"];
        assert_eq!(neuron.inputs[0].name, "default");
        assert_eq!(neuron.outputs[0].name, "default");
    }

    // ===== Shape and Dimension Tests =====

    #[test]
    fn test_parse_shape_wildcard() {
        let source = r#"
neuron Test:
  in: [*, *, *]
  out: [*, *, *]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert_eq!(neuron.inputs[0].shape.dims.len(), 3);
        assert!(matches!(neuron.inputs[0].shape.dims[0], Dim::Wildcard));
    }

    #[test]
    fn test_parse_shape_literal() {
        let source = r#"
neuron Test:
  in: [32, 224, 224, 3]
  out: [32, 1000]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert_eq!(neuron.inputs[0].shape.dims.len(), 4);
        assert!(matches!(neuron.inputs[0].shape.dims[0], Dim::Literal(32)));
        assert!(matches!(neuron.inputs[0].shape.dims[1], Dim::Literal(224)));
    }

    #[test]
    fn test_parse_shape_named() {
        let source = r#"
neuron Test:
  in: [batch, height, width, channels]
  out: [batch, classes]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert!(matches!(&neuron.inputs[0].shape.dims[0], Dim::Named(s) if s == "batch"));
        assert!(matches!(&neuron.inputs[0].shape.dims[1], Dim::Named(s) if s == "height"));
    }

    #[test]
    fn test_parse_shape_variadic() {
        let source = r#"
neuron Test:
  in: [*shape]
  out: [*shape]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert!(matches!(&neuron.inputs[0].shape.dims[0], Dim::Variadic(s) if s == "shape"));
    }

    #[test]
    fn test_parse_shape_expr() {
        let source = r#"
neuron Test:
  in: [batch, dim]
  out: [batch, dim*2]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert!(matches!(&neuron.outputs[0].shape.dims[1], Dim::Expr(_)));
    }

    #[test]
    fn test_parse_negative_dimension() {
        let source = r#"
neuron Test:
  in: [batch, -1]
  out: [batch, dim]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        // Negative numbers are parsed as expressions
        assert!(matches!(&neuron.inputs[0].shape.dims[1], Dim::Expr(_)));
    }

    #[test]
    fn test_parse_empty_shape() {
        let source = r#"
neuron Scalar:
  in: []
  out: []
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Scalar"];
        assert_eq!(neuron.inputs[0].shape.dims.len(), 0);
    }

    // ===== Impl Body Tests =====

    #[test]
    fn test_parse_impl_source() {
        let source = r#"
neuron Test:
  in: [*x]
  out: [*x]
  impl: core,nn/Linear
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Primitive(ImplRef::Source { source, path }) => {
                assert_eq!(source, "core");
                assert_eq!(path, "nn/Linear");
            }
            _ => panic!("Expected primitive impl"),
        }
    }

    #[test]
    fn test_parse_impl_external() {
        let source = r#"
neuron Test:
  in: [*x]
  out: [*x]
  impl: external(module=`torch.nn`, class=`Linear`)
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Primitive(ImplRef::External { kwargs }) => {
                assert_eq!(kwargs.len(), 2);
            }
            _ => panic!("Expected external impl"),
        }
    }

    // ===== Graph Body Tests =====

    #[test]
    fn test_parse_simple_graph() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert!(neuron.is_composite());
        match &neuron.body {
            NeuronBody::Graph { connections, .. } => {
                assert_eq!(connections.len(), 2);
            }
            _ => panic!("Expected graph body"),
        }
    }

    #[test]
    fn test_parse_indented_pipeline() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in ->
      Linear(dim, dim*2)
      ReLU()
      Linear(dim*2, dim)
      out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Graph { connections, .. } => {
                assert_eq!(connections.len(), 4);
            }
            _ => panic!("Expected graph body"),
        }
    }

    #[test]
    fn test_parse_fork_and_tuple() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Fork() -> (a, b)
    (a, b) -> Add() -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Graph { connections, .. } => {
                assert_eq!(connections.len(), 4);
                // Check tuple endpoint
                assert!(matches!(&connections[1].destination, Endpoint::Tuple(_)));
                assert!(matches!(&connections[2].source, Endpoint::Tuple(_)));
            }
            _ => panic!("Expected graph body"),
        }
    }

    #[test]
    fn test_parse_port_access() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> node.port -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Graph { connections, .. } => {
                match &connections[0].destination {
                    Endpoint::Ref(port_ref) => {
                        assert_eq!(port_ref.node, "node");
                        assert_eq!(port_ref.port, "port");
                    }
                    _ => panic!("Expected port ref"),
                }
            }
            _ => panic!("Expected graph body"),
        }
    }

    #[test]
    fn test_parse_in_out_keywords_as_refs() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Graph { connections, .. } => {
                assert!(matches!(&connections[0].source, Endpoint::Ref(r) if r.node == "in"));
                assert!(matches!(&connections[1].destination, Endpoint::Ref(r) if r.node == "out"));
            }
            _ => panic!("Expected graph body"),
        }
    }

    // ===== Match Expression Tests =====

    #[test]
    fn test_parse_match_simple() {
        let source = r#"
neuron Identity:
  in: [*shape]
  out: [*shape]
  impl: core,builtin/Identity

neuron Linear(in_dim, out_dim):
  in: [*, in_dim]
  out: [*, out_dim]
  impl: core,nn/Linear

neuron SimpleMatch:
  in: [*, dim]
  out: [*, 512]
  graph:
    in -> match:
      [*, 512]: Identity() -> out
      [*, 256]: Linear(256, 512) -> out
      [*, 1024]: Linear(1024, 512) -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["SimpleMatch"];
        match &neuron.body {
            NeuronBody::Graph { connections, .. } => {
                match &connections[0].destination {
                    Endpoint::Match(match_expr) => {
                        assert_eq!(match_expr.arms.len(), 3);
                        // Check first arm
                        assert_eq!(match_expr.arms[0].pattern.dims.len(), 2);
                        assert_eq!(match_expr.arms[0].pipeline.len(), 2);
                    }
                    _ => panic!("Expected match endpoint"),
                }
            }
            _ => panic!("Expected graph body"),
        }
    }

    #[test]
    fn test_parse_match_with_wildcard() {
        let source = r#"
neuron Identity:
  in: [*shape]
  out: [*shape]
  impl: core,builtin/Identity

neuron Linear(in_dim, out_dim):
  in: [*, in_dim]
  out: [*, out_dim]
  impl: core,nn/Linear

neuron MatchWithWildcard:
  in: [*, dim]
  out: [*, 512]
  graph:
    in -> match:
      [*, 512]: Identity() -> out
      [*, d]: Linear(d, 512) -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["MatchWithWildcard"];
        match &neuron.body {
            NeuronBody::Graph { connections, .. } => {
                match &connections[0].destination {
                    Endpoint::Match(match_expr) => {
                        assert_eq!(match_expr.arms.len(), 2);
                        // Second arm should have named dimension
                        assert!(matches!(&match_expr.arms[1].pattern.dims[1], Dim::Named(s) if s == "d"));
                    }
                    _ => panic!("Expected match endpoint"),
                }
            }
            _ => panic!("Expected graph body"),
        }
    }

    #[test]
    fn test_parse_match_with_guard() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> match:
      [*, d] where d > 512: Linear(d, 512) -> out
      [*, d]: Identity() -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Graph { connections, .. } => {
                match &connections[0].destination {
                    Endpoint::Match(match_expr) => {
                        assert!(match_expr.arms[0].guard.is_some());
                        assert!(match_expr.arms[1].guard.is_none());
                    }
                    _ => panic!("Expected match endpoint"),
                }
            }
            _ => panic!("Expected graph body"),
        }
    }

    // ===== Expression Tests =====

    #[test]
    fn test_parse_expr_literals() {
        let source = r#"
neuron Test(a=42, b=3.14, c=`hello`, d=true, e=false):
  in: [*x]
  out: [*x]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert!(matches!(neuron.params[0].default, Some(Value::Int(42))));
        assert!(matches!(neuron.params[1].default, Some(Value::Float(_))));
        assert!(matches!(neuron.params[2].default, Some(Value::String(_))));
        assert!(matches!(neuron.params[3].default, Some(Value::Bool(true))));
        assert!(matches!(neuron.params[4].default, Some(Value::Bool(false))));
    }

    #[test]
    fn test_parse_expr_arithmetic() {
        let source = r#"
neuron Test(a=1+2, b=3*4, c=10-5, d=20/4):
  in: [*x]
  out: [*x]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert!(matches!(neuron.params[0].default, Some(Value::BinOp { .. })));
    }

    #[test]
    fn test_parse_expr_comparison() {
        let source = r#"
neuron Test(a=x>5, b=y<10, c=z==3, d=w!=0, e=a>=1, f=b<=2):
  in: [*x]
  out: [*x]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        for param in &neuron.params {
            assert!(matches!(param.default, Some(Value::BinOp { .. })));
        }
    }

    #[test]
    fn test_parse_expr_negative() {
        let source = r#"
neuron Test(a=-42, b=-3.14):
  in: [*x]
  out: [*x]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        // Negative numbers are parsed as binary operations (0 - value)
        assert!(matches!(neuron.params[0].default, Some(Value::BinOp { .. })));
    }

    #[test]
    fn test_parse_expr_function_call() {
        let source = r#"
neuron Test(a=sqrt(16), b=max(1, 2, 3)):
  in: [*x]
  out: [*x]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert!(matches!(neuron.params[0].default, Some(Value::Call { .. })));
        assert!(matches!(neuron.params[1].default, Some(Value::Call { .. })));
    }

    #[test]
    fn test_parse_expr_parentheses() {
        let source = r#"
neuron Test(a=(1+2)*3):
  in: [*x]
  out: [*x]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert!(matches!(neuron.params[0].default, Some(Value::BinOp { .. })));
    }

    // ===== Call Arguments Tests =====

    #[test]
    fn test_parse_call_positional_args() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(256, 512) -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Graph { connections, .. } => {
                match &connections[0].destination {
                    Endpoint::Call { args, kwargs, .. } => {
                        assert_eq!(args.len(), 2);
                        assert_eq!(kwargs.len(), 0);
                    }
                    _ => panic!("Expected call endpoint"),
                }
            }
            _ => panic!("Expected graph body"),
        }
    }

    #[test]
    fn test_parse_call_keyword_args() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Conv(in_ch=3, out_ch=64, kernel=3) -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Graph { connections, .. } => {
                match &connections[0].destination {
                    Endpoint::Call { args, kwargs, .. } => {
                        assert_eq!(args.len(), 0);
                        assert_eq!(kwargs.len(), 3);
                        assert_eq!(kwargs[0].0, "in_ch");
                        assert_eq!(kwargs[1].0, "out_ch");
                        assert_eq!(kwargs[2].0, "kernel");
                    }
                    _ => panic!("Expected call endpoint"),
                }
            }
            _ => panic!("Expected graph body"),
        }
    }

    #[test]
    fn test_parse_call_mixed_args() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Conv(3, 64, kernel=3, stride=1) -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Graph { connections, .. } => {
                match &connections[0].destination {
                    Endpoint::Call { args, kwargs, .. } => {
                        assert_eq!(args.len(), 2);
                        assert_eq!(kwargs.len(), 2);
                    }
                    _ => panic!("Expected call endpoint"),
                }
            }
            _ => panic!("Expected graph body"),
        }
    }

    // ===== Integration Tests =====

    #[test]
    fn test_parse_residual_example() {
        let source = std::fs::read_to_string("examples/residual.ns").unwrap();
        let program = Parser::parse(&source).unwrap();

        assert_eq!(program.neurons.len(), 6);
        assert_eq!(program.uses.len(), 1);

        // Get the Residual and MLP neurons
        let residual = &program.neurons["Residual"];
        let mlp = &program.neurons["MLP"];

        // Check that Residual and MLP are composite (have graphs)
        assert!(residual.is_composite());
        assert!(mlp.is_composite());

        // Extract connections from the graphs
        let NeuronBody::Graph { connections: residual_connections, .. } = &residual.body else {
            panic!("Expected graph body for Residual");
        };
        let NeuronBody::Graph { connections: mlp_connections, .. } = &mlp.body else {
            panic!("Expected graph body for MLP");
        };

        // Assert Residual has 6 connections
        assert_eq!(residual_connections.len(), 6);

        // Assert MLP has 4 connections
        assert_eq!(mlp_connections.len(), 4);

        // Assert the following 6 connections for Residual:
        // 1. Ref(PortRef { node: "in", port: "default" }) -> Call { name: "Fork", args: [], kwargs: [] }
        assert_eq!(residual_connections[0].source, Endpoint::Ref(PortRef::new("in")));
        assert_eq!(residual_connections[0].destination, Endpoint::Call {
            name: "Fork".to_string(),
            args: vec![],
            kwargs: vec![],
            id: 3
        });

        // 2. Call { name: "Fork", args: [], kwargs: [] } -> Tuple([PortRef { node: "main", port: "default" }, PortRef { node: "skip", port: "default" }])
        assert_eq!(residual_connections[1].source, Endpoint::Call {
            name: "Fork".to_string(),
            args: vec![],
            kwargs: vec![],
            id: 3
        });
        assert_eq!(residual_connections[1].destination, Endpoint::Tuple(vec![
            PortRef::new("main"),
            PortRef::new("skip")
        ]));

        // 3. Ref(PortRef { node: "main", port: "default" }) -> Call { name: "MLP", args: [Name("dim")], kwargs: [] }
        assert_eq!(residual_connections[2].source, Endpoint::Ref(PortRef::new("main")));
        assert_eq!(residual_connections[2].destination, Endpoint::Call {
            name: "MLP".to_string(),
            args: vec![Value::Name("dim".to_string())],
            kwargs: vec![],
            id: 4
        });

        // 4. Call { name: "MLP", args: [Name("dim")], kwargs: [] } -> Ref(PortRef { node: "processed", port: "default" })
        assert_eq!(residual_connections[3].source, Endpoint::Call {
            name: "MLP".to_string(),
            args: vec![Value::Name("dim".to_string())],
            kwargs: vec![],
            id: 4
        });
        assert_eq!(residual_connections[3].destination, Endpoint::Ref(PortRef::new("processed")));

        // 5. Tuple([PortRef { node: "processed", port: "default" }, PortRef { node: "skip", port: "default" }]) -> Call { name: "Add", args: [], kwargs: [] }
        assert_eq!(residual_connections[4].source, Endpoint::Tuple(vec![
            PortRef::new("processed"),
            PortRef::new("skip")
        ]));
        assert_eq!(residual_connections[4].destination, Endpoint::Call {
            name: "Add".to_string(),
            args: vec![],
            kwargs: vec![],
            id: 5
        });

        // 6. Call { name: "Add", args: [], kwargs: [] } -> Ref(PortRef { node: "out", port: "default" })
        assert_eq!(residual_connections[5].source, Endpoint::Call {
            name: "Add".to_string(),
            args: vec![],
            kwargs: vec![],
            id: 5
        });
        assert_eq!(residual_connections[5].destination, Endpoint::Ref(PortRef::new("out")));
    }

    // ===== Error Handling Tests =====

    #[test]
    fn test_error_unexpected_token() {
        let source = "invalid token sequence\n";
        let result = Parser::parse(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_missing_colon() {
        let source = r#"
neuron Test
  in: [*x]
  out: [*x]
  impl: core,builtin/Identity
"#;
        let result = Parser::parse(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_missing_impl() {
        let source = r#"
neuron Test:
  in: [*x]
  out: [*x]
"#;
        let result = Parser::parse(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unclosed_bracket() {
        let source = r#"
neuron Test:
  in: [*, dim
  out: [*, dim]
  impl: core,builtin/Identity
"#;
        let result = Parser::parse(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unclosed_paren() {
        let source = r#"
neuron Test(dim:
  in: [*, dim]
  out: [*, dim]
  impl: core,builtin/Identity
"#;
        let result = Parser::parse(source);
        assert!(result.is_err());
    }
}
