//! AST desugaring passes
//!
//! Rewrites syntactic sugar (like @wrap) into standard IR before validation.
//! This pass runs after parsing but before validation, so the validator
//! and codegen only ever see standard Endpoint variants.

use crate::interfaces::*;

/// Desugar all @wrap annotations in a program.
/// Must be called after parsing but before validation.
pub fn desugar_wraps(program: &mut Program) {
    let neuron_names: Vec<String> = program.neurons.keys().cloned().collect();
    for name in &neuron_names {
        if let Some(neuron) = program.neurons.get_mut(name) {
            desugar_neuron_wraps(neuron);
        }
    }
}

fn desugar_neuron_wraps(neuron: &mut NeuronDef) {
    if let NeuronBody::Graph {
        ref mut context_bindings,
        ref mut connections,
        ..
    } = neuron.body
    {
        let mut new_bindings = Vec::new();
        let mut wrap_counter = 0;

        for conn in connections.iter_mut() {
            desugar_endpoint_wraps(
                &mut conn.source,
                context_bindings,
                &mut new_bindings,
                &mut wrap_counter,
            );
            desugar_endpoint_wraps(
                &mut conn.destination,
                context_bindings,
                &mut new_bindings,
                &mut wrap_counter,
            );
        }

        // Prepend synthesized bindings to context
        if !new_bindings.is_empty() {
            context_bindings.splice(0..0, new_bindings);
        }
    }
}

fn desugar_endpoint_wraps(
    endpoint: &mut Endpoint,
    _existing_bindings: &[Binding],
    new_bindings: &mut Vec<Binding>,
    counter: &mut usize,
) {
    match endpoint {
        Endpoint::Wrap(wrap_expr) => {
            let wrap_id = *counter;
            *counter += 1;

            match &wrap_expr.content {
                WrapContent::Ref(binding_name) => {
                    // Reference form: @wrap(Wrapper, a, b): existing
                    // Desugars to: Wrapper(existing, a, b)
                    // The wrapper's first Neuron-typed param receives the existing binding.
                    let mut call_args = vec![Value::Name(binding_name.clone())];
                    call_args.extend(wrap_expr.wrapper_args.clone());

                    *endpoint = Endpoint::Call {
                        name: wrap_expr.wrapper_name.clone(),
                        args: call_args,
                        kwargs: wrap_expr.wrapper_kwargs.clone(),
                        id: wrap_expr.id,
                        frozen: false,
                    };
                }
                WrapContent::Pipeline(pipeline_endpoints) => {
                    // Pipeline form: @wrap(Wrapper, a, b): -> X -> Y
                    // Desugars to:
                    //   context: _wrap_N = __sequential__(X, Y)
                    //   Wrapper(_wrap_N, a, b)
                    let anon_name = format!("_wrap_{}", wrap_id);

                    // Create synthetic binding for the sequential pipeline.
                    // Each endpoint in the pipeline becomes a Value::Call or Value::Name arg
                    // for the __sequential__ pseudo-neuron.
                    // Filter out port references (in/out) that the parser may have
                    // captured from the surrounding pipeline context.
                    let seq_args: Vec<Value> = pipeline_endpoints
                        .iter()
                        .filter_map(|ep| match ep {
                            Endpoint::Call {
                                name,
                                args,
                                kwargs,
                                ..
                            } => Some(Value::Call {
                                name: name.clone(),
                                args: args.clone(),
                                kwargs: kwargs.clone(),
                            }),
                            Endpoint::Ref(port_ref) => {
                                // Skip port references (in/out) -- they are
                                // pipeline routing, not sequential members
                                if port_ref.node == "in" || port_ref.node == "out" {
                                    None
                                } else {
                                    Some(Value::Name(port_ref.node.clone()))
                                }
                            }
                            _ => None,
                        })
                        .collect();

                    new_bindings.push(Binding {
                        name: anon_name.clone(),
                        call_name: crate::interfaces::SEQUENTIAL_PSEUDO_NEURON.to_string(),
                        args: seq_args,
                        kwargs: vec![],
                        scope: Scope::Instance { lazy: false },
                        frozen: false,
                        unroll_group: None,
                    });

                    // Rewrite the endpoint to a Call to the wrapper
                    let mut call_args = vec![Value::Name(anon_name)];
                    call_args.extend(wrap_expr.wrapper_args.clone());

                    *endpoint = Endpoint::Call {
                        name: wrap_expr.wrapper_name.clone(),
                        args: call_args,
                        kwargs: wrap_expr.wrapper_kwargs.clone(),
                        id: wrap_expr.id,
                        frozen: false,
                    };
                }
            }
        }
        Endpoint::Match(match_expr) => {
            for arm in &mut match_expr.arms {
                for ep in &mut arm.pipeline {
                    desugar_endpoint_wraps(ep, _existing_bindings, new_bindings, counter);
                }
            }
        }
        Endpoint::If(if_expr) => {
            for branch in &mut if_expr.branches {
                for ep in &mut branch.pipeline {
                    desugar_endpoint_wraps(ep, _existing_bindings, new_bindings, counter);
                }
            }
            if let Some(else_branch) = &mut if_expr.else_branch {
                for ep in else_branch {
                    desugar_endpoint_wraps(ep, _existing_bindings, new_bindings, counter);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_desugar_wrap_ref() {
        let mut program = Program::new();
        let neuron = NeuronDef {
            name: "Test".to_string(),
            params: vec![Param {
                name: "dim".to_string(),
                default: None,
                type_annotation: None,
            }],
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
            max_cycle_depth: None,
            doc: None,
            body: NeuronBody::Graph {
                context_bindings: vec![Binding {
                    name: "attn".to_string(),
                    call_name: "MultiHeadSelfAttention".to_string(),
                    args: vec![Value::Name("dim".to_string()), Value::Int(8)],
                    kwargs: vec![],
                    scope: Scope::Instance { lazy: false },
                    frozen: false,
                    unroll_group: None,
                }],
                context_unrolls: vec![],
                connections: vec![Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Wrap(WrapExpr {
                        wrapper_name: "HyperConnect".to_string(),
                        wrapper_args: vec![
                            Value::Int(4),
                            Value::Name("dim".to_string()),
                            Value::Int(0),
                        ],
                        wrapper_kwargs: vec![],
                        content: WrapContent::Ref("attn".to_string()),
                        id: 99,
                    }),
                }],
            },
        };
        program.neurons.insert("Test".to_string(), neuron);

        desugar_wraps(&mut program);

        let test_neuron = program.neurons.get("Test").unwrap();
        if let NeuronBody::Graph { connections, .. } = &test_neuron.body {
            // After desugaring, the Wrap should be rewritten to a Call
            match &connections[0].destination {
                Endpoint::Call {
                    name, args, id, ..
                } => {
                    assert_eq!(name, "HyperConnect");
                    // First arg should be the binding name "attn"
                    assert_eq!(args[0], Value::Name("attn".to_string()));
                    // Remaining args: 4, dim, 0
                    assert_eq!(args[1], Value::Int(4));
                    assert_eq!(args[2], Value::Name("dim".to_string()));
                    assert_eq!(args[3], Value::Int(0));
                    assert_eq!(*id, 99);
                }
                other => panic!("Expected Call endpoint, got {:?}", other),
            }
        } else {
            panic!("Expected Graph body");
        }
    }

    #[test]
    fn test_desugar_wrap_pipeline() {
        let mut program = Program::new();
        let neuron = NeuronDef {
            name: "Test".to_string(),
            params: vec![Param {
                name: "dim".to_string(),
                default: None,
                type_annotation: None,
            }],
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
            max_cycle_depth: None,
            doc: None,
            body: NeuronBody::Graph {
                context_bindings: vec![],
                context_unrolls: vec![],
                connections: vec![Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Wrap(WrapExpr {
                        wrapper_name: "HyperConnect".to_string(),
                        wrapper_args: vec![
                            Value::Int(4),
                            Value::Name("dim".to_string()),
                            Value::Int(0),
                        ],
                        wrapper_kwargs: vec![],
                        content: WrapContent::Pipeline(vec![
                            Endpoint::Call {
                                name: "LayerNorm".to_string(),
                                args: vec![Value::Name("dim".to_string())],
                                kwargs: vec![],
                                id: 10,
                                frozen: false,
                            },
                            Endpoint::Call {
                                name: "Linear".to_string(),
                                args: vec![
                                    Value::Name("dim".to_string()),
                                    Value::Name("dim".to_string()),
                                ],
                                kwargs: vec![],
                                id: 11,
                                frozen: false,
                            },
                        ]),
                        id: 99,
                    }),
                }],
            },
        };
        program.neurons.insert("Test".to_string(), neuron);

        desugar_wraps(&mut program);

        let test_neuron = program.neurons.get("Test").unwrap();
        if let NeuronBody::Graph {
            context_bindings,
            connections,
            ..
        } = &test_neuron.body
        {
            // Should have synthesized a __sequential__ binding
            assert!(
                context_bindings
                    .iter()
                    .any(|b| b.call_name == crate::interfaces::SEQUENTIAL_PSEUDO_NEURON && b.name == "_wrap_0"),
                "Should have __sequential__ binding named _wrap_0"
            );

            let seq_binding = context_bindings
                .iter()
                .find(|b| b.call_name == crate::interfaces::SEQUENTIAL_PSEUDO_NEURON)
                .unwrap();
            assert_eq!(seq_binding.args.len(), 2); // LayerNorm and Linear

            // The connection destination should now be a Call to HyperConnect
            match &connections[0].destination {
                Endpoint::Call { name, args, .. } => {
                    assert_eq!(name, "HyperConnect");
                    // First arg is the synthesized binding name
                    assert_eq!(args[0], Value::Name("_wrap_0".to_string()));
                    assert_eq!(args[1], Value::Int(4));
                }
                other => panic!("Expected Call endpoint, got {:?}", other),
            }
        } else {
            panic!("Expected Graph body");
        }
    }
}
