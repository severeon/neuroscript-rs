//! Compile-time context-unroll expansion pass.
//!
//! Runs before validation and codegen, lowering `ContextUnroll` blocks into
//! ordinary bindings (N copies with suffixed names). Graph-level unroll
//! expansion is not handled here; connections are copied through as-is.

use crate::interfaces::*;

/// Maximum allowed unroll count per individual unroll block.
const MAX_UNROLL_COUNT: usize = 1024;

/// Maximum total expanded bindings across all unroll groups in a single neuron.
/// Prevents combinatorial explosion (e.g., unroll(100) with 100 bindings = 10,000).
const MAX_TOTAL_EXPANDED_BINDINGS: usize = 10_000;

/// Expand all context-unroll constructs in a program.
///
/// This modifies the program in place, replacing `ContextUnroll` blocks
/// with N suffixed bindings (e.g., `block_0`, `block_1`, ...).
///
/// Must be called before validation.
pub fn expand_unrolls(program: &mut Program) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    // Collect neuron names first to avoid borrow issues
    let neuron_names: Vec<String> = program.neurons.keys().cloned().collect();

    for name in &neuron_names {
        // Safety: name was collected from program.neurons.keys() above
        let neuron = program.neurons.get(name)
            .expect("neuron disappeared during unroll expansion");
        let params = neuron.params.clone();

        if let NeuronBody::Graph {
            context_bindings,
            context_unrolls,
            connections,
        } = &neuron.body
        {
            // Skip if no context unrolls present
            if context_unrolls.is_empty() {
                continue;
            }

            // --- Expand context unrolls ---
            // Pre-allocate with estimated capacity: existing bindings + unrolled ones
            let estimated_new = context_unrolls.iter().map(|u| u.bindings.len()).sum::<usize>()
                * 4; // rough overestimate per binding
            let mut new_bindings = Vec::with_capacity(context_bindings.len() + estimated_new);
            new_bindings.extend_from_slice(context_bindings);

            // Track total expanded bindings to prevent combinatorial explosion
            let mut total_expanded: usize = context_bindings.len();

            for unroll in context_unrolls {
                match resolve_count(&unroll.count, &params) {
                    Some(count) => {
                        // Pre-check: will this unroll group exceed the total limit?
                        let bindings_to_add: usize = unroll.bindings.iter()
                            .map(|b| if matches!(b.scope, Scope::Static) { 1 } else { count })
                            .sum();
                        total_expanded = total_expanded.saturating_add(bindings_to_add);
                        if total_expanded > MAX_TOTAL_EXPANDED_BINDINGS {
                            errors.push(ValidationError::InvalidUnrollCount {
                                neuron: name.clone(),
                                reason: format!(
                                    "Total expanded bindings ({}) exceeds maximum allowed ({})",
                                    total_expanded, MAX_TOTAL_EXPANDED_BINDINGS
                                ),
                            });
                            break;
                        }

                        for binding in &unroll.bindings {
                            if matches!(binding.scope, Scope::Static) {
                                // @static: single shared instance, keep unroll_group
                                // for aggregate name tracking
                                let mut b = binding.clone();
                                b.unroll_group = Some(UnrollGroupInfo {
                                    base_name: binding.name.clone(),
                                    count: unroll.count.clone(),
                                    index: 0,
                                    aggregate_name: unroll.aggregate_name.clone(),
                                });
                                new_bindings.push(b);
                            } else {
                                for i in 0..count {
                                    let suffixed_name = format!("{}_{}", binding.name, i);
                                    new_bindings.push(Binding {
                                        name: suffixed_name,
                                        call_name: binding.call_name.clone(),
                                        args: binding.args.clone(),
                                        kwargs: binding.kwargs.clone(),
                                        scope: binding.scope.clone(),
                                        frozen: binding.frozen,
                                        unroll_group: Some(UnrollGroupInfo {
                                            base_name: binding.name.clone(),
                                            count: unroll.count.clone(),
                                            index: i,
                                            aggregate_name: unroll.aggregate_name.clone(),
                                        }),
                                    });
                                }
                            }
                        }
                    }
                    None => {
                        errors.push(ValidationError::InvalidUnrollCount {
                            neuron: name.clone(),
                            reason: format!(
                                "Could not resolve unroll count '{}'",
                                unroll.count
                            ),
                        });
                    }
                }
            }

            // Copy connections through unchanged
            let new_connections = connections.clone();

            // Replace neuron body
            // Safety: name was collected from program.neurons.keys() above
            let neuron = program.neurons.get_mut(name)
                .expect("neuron disappeared during unroll expansion");
            neuron.body = NeuronBody::Graph {
                context_bindings: new_bindings,
                context_unrolls: vec![],
                connections: new_connections,
            };
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Resolve an unroll count to a concrete usize.
///
/// - `Value::Int(n)` where n > 0 -> Some(n)
/// - `Value::Name(name)` -> look up param default
/// - Everything else -> None (caller should push error)
fn resolve_count(count: &Value, params: &[Param]) -> Option<usize> {
    match count {
        Value::Int(n) if *n > 0 && (*n as usize) <= MAX_UNROLL_COUNT => Some(*n as usize),
        Value::Int(_) => None, // zero, negative, or exceeds limit
        Value::Name(name) => {
            // Look up parameter default value
            for param in params {
                if &param.name == name {
                    if let Some(Value::Int(n)) = &param.default {
                        if *n > 0 && (*n as usize) <= MAX_UNROLL_COUNT {
                            return Some(*n as usize);
                        }
                    }
                    return None;
                }
            }
            None
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_param(name: &str, default: Option<i64>) -> Param {
        Param {
            name: name.to_string(),
            default: default.map(Value::Int),
            type_annotation: None,
        }
    }

    #[test]
    fn test_resolve_count_literal() {
        let params = vec![];
        assert_eq!(resolve_count(&Value::Int(3), &params), Some(3));
        assert_eq!(resolve_count(&Value::Int(1), &params), Some(1));
        assert_eq!(resolve_count(&Value::Int(0), &params), None);
        assert_eq!(resolve_count(&Value::Int(-1), &params), None);
    }

    #[test]
    fn test_resolve_count_param_ref() {
        let params = vec![make_param("num_layers", Some(6))];
        assert_eq!(
            resolve_count(&Value::Name("num_layers".to_string()), &params),
            Some(6)
        );
        assert_eq!(
            resolve_count(&Value::Name("unknown".to_string()), &params),
            None
        );
    }

    #[test]
    fn test_expand_context_unrolls() {
        let mut program = Program::new();
        program.neurons.insert(
            "Stack".to_string(),
            NeuronDef {
                name: "Stack".to_string(),
                params: vec![make_param("d_model", Some(512)), make_param("num_layers", Some(3))],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![ContextUnroll {
                        aggregate_name: "blocks".to_string(),
                        count: Value::Name("num_layers".to_string()),
                        bindings: vec![Binding {
                            name: "block".to_string(),
                            call_name: "TransformerBlock".to_string(),
                            args: vec![Value::Name("d_model".to_string())],
                            kwargs: vec![],
                            scope: Scope::Instance { lazy: false },
                            frozen: false,
                            unroll_group: None,
                        }],
                    }],
                    connections: vec![],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = expand_unrolls(&mut program);
        assert!(result.is_ok());

        let neuron = program.neurons.get("Stack").unwrap();
        if let NeuronBody::Graph {
            context_bindings,
            context_unrolls,
            ..
        } = &neuron.body
        {
            assert!(context_unrolls.is_empty(), "unrolls should be cleared");
            assert_eq!(context_bindings.len(), 3);
            assert_eq!(context_bindings[0].name, "block_0");
            assert_eq!(context_bindings[1].name, "block_1");
            assert_eq!(context_bindings[2].name, "block_2");
            // Verify unroll_group metadata is set
            assert!(context_bindings[0].unroll_group.is_some());
            assert_eq!(context_bindings[0].unroll_group.as_ref().unwrap().base_name, "block");
            assert_eq!(context_bindings[0].unroll_group.as_ref().unwrap().index, 0);
            assert_eq!(context_bindings[2].unroll_group.as_ref().unwrap().index, 2);
            assert_eq!(context_bindings[0].unroll_group.as_ref().unwrap().aggregate_name, "blocks");
        } else {
            panic!("Expected Graph body, got Primitive");
        }
    }

    #[test]
    fn test_expand_static_context_unroll() {
        let mut program = Program::new();
        program.neurons.insert(
            "Shared".to_string(),
            NeuronDef {
                name: "Shared".to_string(),
                params: vec![make_param("num_layers", Some(3))],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![ContextUnroll {
                        aggregate_name: "blocks".to_string(),
                        count: Value::Name("num_layers".to_string()),
                        bindings: vec![Binding {
                            name: "block".to_string(),
                            call_name: "TransformerBlock".to_string(),
                            args: vec![],
                            kwargs: vec![],
                            scope: Scope::Static,
                            frozen: false,
                            unroll_group: None,
                        }],
                    }],
                    connections: vec![],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = expand_unrolls(&mut program);
        assert!(result.is_ok());

        let neuron = program.neurons.get("Shared").unwrap();
        if let NeuronBody::Graph {
            context_bindings, ..
        } = &neuron.body
        {
            // @static: single binding, no suffix, but with unroll_group for aggregate tracking
            assert_eq!(context_bindings.len(), 1);
            assert_eq!(context_bindings[0].name, "block");
            assert!(context_bindings[0].unroll_group.is_some(), "Should have unroll_group for aggregate tracking");
            let group = context_bindings[0].unroll_group.as_ref().unwrap();
            assert_eq!(group.base_name, "block");
            assert_eq!(group.aggregate_name, "blocks");
            assert_eq!(group.index, 0);
        } else {
            panic!("Expected Graph body, got Primitive");
        }
    }

    #[test]
    fn test_invalid_unroll_count_zero() {
        let mut program = Program::new();
        program.neurons.insert(
            "Bad".to_string(),
            NeuronDef {
                name: "Bad".to_string(),
                params: vec![],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![ContextUnroll {
                        aggregate_name: "items".to_string(),
                        count: Value::Int(0),
                        bindings: vec![],
                    }],
                    connections: vec![],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = expand_unrolls(&mut program);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_context_unrolls() {
        let mut program = Program::new();
        program.neurons.insert(
            "Multi".to_string(),
            NeuronDef {
                name: "Multi".to_string(),
                params: vec![],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![
                        ContextUnroll {
                            aggregate_name: "attns".to_string(),
                            count: Value::Int(2),
                            bindings: vec![Binding {
                                name: "attn".to_string(),
                                call_name: "Attention".to_string(),
                                args: vec![],
                                kwargs: vec![],
                                scope: Scope::Instance { lazy: false },
                                frozen: false,
                                unroll_group: None,
                            }],
                        },
                        ContextUnroll {
                            aggregate_name: "ffns".to_string(),
                            count: Value::Int(3),
                            bindings: vec![Binding {
                                name: "ffn".to_string(),
                                call_name: "FFN".to_string(),
                                args: vec![],
                                kwargs: vec![],
                                scope: Scope::Instance { lazy: false },
                                frozen: false,
                                unroll_group: None,
                            }],
                        },
                    ],
                    connections: vec![],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = expand_unrolls(&mut program);
        assert!(result.is_ok());

        let neuron = program.neurons.get("Multi").unwrap();
        if let NeuronBody::Graph {
            context_bindings, ..
        } = &neuron.body
        {
            // 2 attn + 3 ffn = 5 bindings
            assert_eq!(context_bindings.len(), 5);
            assert_eq!(context_bindings[0].name, "attn_0");
            assert_eq!(context_bindings[1].name, "attn_1");
            assert_eq!(context_bindings[2].name, "ffn_0");
            assert_eq!(context_bindings[3].name, "ffn_1");
            assert_eq!(context_bindings[4].name, "ffn_2");
        } else {
            panic!("Expected Graph body, got Primitive");
        }
    }

    /// Mixed regular bindings and unrolls in the same context section.
    /// Verifies ordering: layer1 (regular), block_0..2 (unrolled), layer2 (regular).
    #[test]
    fn test_mixed_bindings_and_unrolls() {
        let mut program = Program::new();
        program.neurons.insert(
            "Mixed".to_string(),
            NeuronDef {
                name: "Mixed".to_string(),
                params: vec![make_param("dim", Some(256))],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![
                        Binding {
                            name: "layer1".to_string(),
                            call_name: "Linear".to_string(),
                            args: vec![
                                Value::Name("dim".to_string()),
                                Value::Name("dim".to_string()),
                            ],
                            kwargs: vec![],
                            scope: Scope::Instance { lazy: false },
                            frozen: false,
                            unroll_group: None,
                        },
                        // layer2 comes after the unroll in the source, but the AST
                        // builder moves overflow bindings back to context_bindings
                        Binding {
                            name: "layer2".to_string(),
                            call_name: "Linear".to_string(),
                            args: vec![
                                Value::Name("dim".to_string()),
                                Value::Name("dim".to_string()),
                            ],
                            kwargs: vec![],
                            scope: Scope::Instance { lazy: false },
                            frozen: false,
                            unroll_group: None,
                        },
                    ],
                    context_unrolls: vec![ContextUnroll {
                        aggregate_name: "blocks".to_string(),
                        count: Value::Int(3),
                        bindings: vec![Binding {
                            name: "block".to_string(),
                            call_name: "TransformerBlock".to_string(),
                            args: vec![Value::Name("dim".to_string())],
                            kwargs: vec![],
                            scope: Scope::Instance { lazy: false },
                            frozen: false,
                            unroll_group: None,
                        }],
                    }],
                    connections: vec![],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = expand_unrolls(&mut program);
        assert!(result.is_ok(), "expand_unrolls failed: {:?}", result);

        let neuron = program.neurons.get("Mixed").unwrap();
        if let NeuronBody::Graph {
            context_bindings,
            context_unrolls,
            ..
        } = &neuron.body
        {
            assert!(context_unrolls.is_empty());
            // Original bindings (layer1, layer2) + 3 expanded (block_0..2)
            assert_eq!(context_bindings.len(), 5);
            assert_eq!(context_bindings[0].name, "layer1");
            assert_eq!(context_bindings[1].name, "layer2");
            assert_eq!(context_bindings[2].name, "block_0");
            assert_eq!(context_bindings[3].name, "block_1");
            assert_eq!(context_bindings[4].name, "block_2");
        } else {
            panic!("Expected Graph body, got Primitive");
        }
    }

    /// @lazy bindings inside unroll should expand correctly, preserving the
    /// lazy scope on each suffixed copy.
    #[test]
    fn test_unroll_with_lazy_binding() {
        let mut program = Program::new();
        program.neurons.insert(
            "LazyUnroll".to_string(),
            NeuronDef {
                name: "LazyUnroll".to_string(),
                params: vec![],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![ContextUnroll {
                        aggregate_name: "blocks".to_string(),
                        count: Value::Int(2),
                        bindings: vec![Binding {
                            name: "block".to_string(),
                            call_name: "TransformerBlock".to_string(),
                            args: vec![],
                            kwargs: vec![],
                            scope: Scope::Instance { lazy: true },
                            frozen: false,
                            unroll_group: None,
                        }],
                    }],
                    connections: vec![],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = expand_unrolls(&mut program);
        assert!(result.is_ok(), "expand_unrolls failed: {:?}", result);

        let neuron = program.neurons.get("LazyUnroll").unwrap();
        if let NeuronBody::Graph {
            context_bindings, ..
        } = &neuron.body
        {
            assert_eq!(context_bindings.len(), 2);
            assert_eq!(context_bindings[0].name, "block_0");
            assert_eq!(context_bindings[1].name, "block_1");
            // Each expanded binding preserves the @lazy scope
            assert_eq!(
                context_bindings[0].scope,
                Scope::Instance { lazy: true }
            );
            assert_eq!(
                context_bindings[1].scope,
                Scope::Instance { lazy: true }
            );
        } else {
            panic!("Expected Graph body, got Primitive");
        }
    }

    /// Using a name that isn't a neuron parameter should produce an error.
    /// This covers the case where someone writes unroll(x) with a forward
    /// pass variable rather than a neuron parameter.
    #[test]
    fn test_unroll_with_non_param_variable() {
        let mut program = Program::new();
        program.neurons.insert(
            "DynCount".to_string(),
            NeuronDef {
                name: "DynCount".to_string(),
                params: vec![make_param("dim", Some(256))],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![ContextUnroll {
                        aggregate_name: "blocks".to_string(),
                        count: Value::Name("num_layers".to_string()), // not a param!
                        bindings: vec![Binding {
                            name: "block".to_string(),
                            call_name: "Block".to_string(),
                            args: vec![],
                            kwargs: vec![],
                            scope: Scope::Instance { lazy: false },
                            frozen: false,
                            unroll_group: None,
                        }],
                    }],
                    connections: vec![],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = expand_unrolls(&mut program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        match &errors[0] {
            ValidationError::InvalidUnrollCount { neuron, reason } => {
                assert_eq!(neuron, "DynCount");
                assert!(
                    reason.contains("num_layers"),
                    "Error should mention the unresolved variable, got: {}",
                    reason
                );
            }
            other => panic!("Expected InvalidUnrollCount, got: {}", other),
        }
    }

    #[test]
    fn test_total_expanded_bindings_limit() {
        // Two unroll groups: unroll(100) with 51 bindings each = 10,200 total
        // This should exceed the MAX_TOTAL_EXPANDED_BINDINGS limit of 10,000
        let mut program = Program::new();
        let mut bindings_a: Vec<Binding> = Vec::new();
        for i in 0..51 {
            bindings_a.push(Binding {
                name: format!("a{}", i),
                call_name: "Block".to_string(),
                args: vec![],
                kwargs: vec![],
                scope: Scope::Instance { lazy: false },
                frozen: false,
                unroll_group: None,
            });
        }
        let mut bindings_b: Vec<Binding> = Vec::new();
        for i in 0..51 {
            bindings_b.push(Binding {
                name: format!("b{}", i),
                call_name: "Block".to_string(),
                args: vec![],
                kwargs: vec![],
                scope: Scope::Instance { lazy: false },
                frozen: false,
                unroll_group: None,
            });
        }
        program.neurons.insert(
            "Explosive".to_string(),
            NeuronDef {
                name: "Explosive".to_string(),
                params: vec![],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![
                        ContextUnroll {
                            aggregate_name: "group_a".to_string(),
                            count: Value::Int(100),
                            bindings: bindings_a,
                        },
                        ContextUnroll {
                            aggregate_name: "group_b".to_string(),
                            count: Value::Int(100),
                            bindings: bindings_b,
                        },
                    ],
                    connections: vec![],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = expand_unrolls(&mut program);
        assert!(result.is_err(), "Should fail due to total binding limit");
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(e,
            ValidationError::InvalidUnrollCount { reason, .. }
            if reason.contains("Total expanded bindings")
        )), "Error should mention total expanded bindings limit, got: {:?}", errors);
    }

    #[test]
    fn test_total_expanded_bindings_just_under_limit() {
        // Single unroll(1000) with 10 bindings = 10,000 total (exactly at limit)
        let mut program = Program::new();
        let mut bindings: Vec<Binding> = Vec::new();
        for i in 0..10 {
            bindings.push(Binding {
                name: format!("block{}", i),
                call_name: "Block".to_string(),
                args: vec![],
                kwargs: vec![],
                scope: Scope::Instance { lazy: false },
                frozen: false,
                unroll_group: None,
            });
        }
        program.neurons.insert(
            "JustOk".to_string(),
            NeuronDef {
                name: "JustOk".to_string(),
                params: vec![],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![ContextUnroll {
                        aggregate_name: "blocks".to_string(),
                        count: Value::Int(1000),
                        bindings,
                    }],
                    connections: vec![],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = expand_unrolls(&mut program);
        assert!(result.is_ok(), "Should succeed at exactly the limit: {:?}", result);
    }

    #[test]
    fn test_resolve_count_param_no_default() {
        // Parameter exists but has no default value -- can't resolve at compile time
        let params = vec![Param {
            name: "num_layers".to_string(),
            default: None,
            type_annotation: None,
        }];
        assert_eq!(
            resolve_count(&Value::Name("num_layers".to_string()), &params),
            None
        );
    }

    #[test]
    fn test_resolve_count_exceeds_max() {
        let params = vec![];
        assert_eq!(resolve_count(&Value::Int(1024), &params), Some(1024));
        assert_eq!(resolve_count(&Value::Int(1025), &params), None);
        assert_eq!(resolve_count(&Value::Int(999999), &params), None);
    }
}
