//! Compile-time unroll expansion pass.
//!
//! Runs before validation and codegen, lowering `Endpoint::Unroll` and
//! `ContextUnroll` into ordinary IR (N copies of connections/bindings).

use crate::interfaces::*;

/// Expand all unroll constructs in a program.
///
/// This modifies the program in place, replacing:
/// - `ContextUnroll` blocks with N suffixed bindings
/// - `Endpoint::Unroll` in connections with N chained connections
///
/// Must be called before validation.
pub fn expand_unrolls(program: &mut Program) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    // Collect neuron names first to avoid borrow issues
    let neuron_names: Vec<String> = program.neurons.keys().cloned().collect();

    for name in &neuron_names {
        let neuron = program.neurons.get(name).unwrap();
        let params = neuron.params.clone();

        if let NeuronBody::Graph {
            context_bindings,
            context_unrolls,
            connections,
        } = &neuron.body
        {
            // Skip if no unrolls present
            let has_graph_unrolls = connections.iter().any(|c| {
                has_unroll_endpoint(&c.source) || has_unroll_endpoint(&c.destination)
            });

            if context_unrolls.is_empty() && !has_graph_unrolls {
                continue;
            }

            // --- Expand context unrolls ---
            let mut new_bindings = context_bindings.clone();
            let mut expanded_context_names: Vec<(String, Vec<String>)> = Vec::new();

            for unroll in context_unrolls {
                match resolve_count(&unroll.count, &params, name) {
                    Some(count) => {
                        for binding in &unroll.bindings {
                            if matches!(binding.scope, Scope::Static) {
                                // @static: single shared instance, no suffix
                                new_bindings.push(binding.clone());
                                expanded_context_names.push((
                                    binding.name.clone(),
                                    vec![binding.name.clone()],
                                ));
                            } else {
                                let mut suffixed_names = Vec::new();
                                for i in 0..count {
                                    let suffixed_name = format!("{}_{}", binding.name, i);
                                    suffixed_names.push(suffixed_name.clone());
                                    new_bindings.push(Binding {
                                        name: suffixed_name,
                                        call_name: binding.call_name.clone(),
                                        args: binding.args.clone(),
                                        kwargs: binding.kwargs.clone(),
                                        scope: binding.scope.clone(),
                                        frozen: binding.frozen,
                                    });
                                }
                                expanded_context_names
                                    .push((binding.name.clone(), suffixed_names));
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

            // --- Expand graph-level unrolls in connections ---
            let mut new_connections = Vec::new();
            let mut next_id = find_max_endpoint_id(connections) + 1;

            for conn in connections {
                if has_unroll_endpoint(&conn.destination) {
                    match expand_connection_unroll(
                        conn,
                        &params,
                        name,
                        &expanded_context_names,
                        &mut next_id,
                        &mut errors,
                    ) {
                        Some(expanded) => new_connections.extend(expanded),
                        None => new_connections.push(conn.clone()),
                    }
                } else {
                    new_connections.push(conn.clone());
                }
            }

            // Replace neuron body
            let neuron = program.neurons.get_mut(name).unwrap();
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

/// Check if an endpoint contains an Unroll
fn has_unroll_endpoint(endpoint: &Endpoint) -> bool {
    matches!(endpoint, Endpoint::Unroll(_))
}

/// Resolve an unroll count to a concrete usize.
///
/// - `Value::Int(n)` where n > 0 -> Some(n)
/// - `Value::Name(name)` -> look up param default
/// - Everything else -> None (caller should push error)
fn resolve_count(count: &Value, params: &[Param], _neuron_name: &str) -> Option<usize> {
    match count {
        Value::Int(n) if *n > 0 => Some(*n as usize),
        Value::Int(_) => None, // zero or negative
        Value::Name(name) => {
            // Look up parameter default value
            for param in params {
                if &param.name == name {
                    if let Some(Value::Int(n)) = &param.default {
                        if *n > 0 {
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

/// Find the highest endpoint ID used in a set of connections.
fn find_max_endpoint_id(connections: &[Connection]) -> usize {
    let mut max_id = 0;
    for conn in connections {
        max_id = max_id.max(max_endpoint_id(&conn.source));
        max_id = max_id.max(max_endpoint_id(&conn.destination));
    }
    max_id
}

fn max_endpoint_id(endpoint: &Endpoint) -> usize {
    match endpoint {
        Endpoint::Call { id, .. } => *id,
        Endpoint::Match(m) => {
            let mut max = m.id;
            for arm in &m.arms {
                for ep in &arm.pipeline {
                    max = max.max(max_endpoint_id(ep));
                }
            }
            max
        }
        Endpoint::If(expr) => {
            let mut max = expr.id;
            for branch in &expr.branches {
                for ep in &branch.pipeline {
                    max = max.max(max_endpoint_id(ep));
                }
            }
            if let Some(else_branch) = &expr.else_branch {
                for ep in else_branch {
                    max = max.max(max_endpoint_id(ep));
                }
            }
            max
        }
        Endpoint::Unroll(u) => {
            let mut max = u.id;
            for ep in &u.pipeline {
                max = max.max(max_endpoint_id(ep));
            }
            max
        }
        _ => 0,
    }
}

/// Expand a single connection whose destination is an Unroll endpoint.
///
/// Given: `source -> unroll(N): -> pipeline`
/// Produces N chained connections:
///   source -> pipeline[0]_0 -> pipeline[1]_0 -> ... -> pipeline_0_out
///   pipeline_0_out -> pipeline[0]_1 -> ... -> pipeline_1_out
///   ...
///   pipeline_(N-2)_out -> pipeline[0]_(N-1) -> ... -> next_endpoint
fn expand_connection_unroll(
    conn: &Connection,
    params: &[Param],
    neuron_name: &str,
    expanded_context_names: &[(String, Vec<String>)],
    next_id: &mut usize,
    errors: &mut Vec<ValidationError>,
) -> Option<Vec<Connection>> {
    let unroll = match &conn.destination {
        Endpoint::Unroll(u) => u,
        _ => return None,
    };

    let count = match resolve_count(&unroll.count, params, neuron_name) {
        Some(c) => c,
        None => {
            errors.push(ValidationError::InvalidUnrollCount {
                neuron: neuron_name.to_string(),
                reason: format!("Could not resolve unroll count '{}'", unroll.count),
            });
            return None;
        }
    };

    if count == 0 {
        errors.push(ValidationError::InvalidUnrollCount {
            neuron: neuron_name.to_string(),
            reason: "Unroll count must be > 0".to_string(),
        });
        return None;
    }

    let mut result = Vec::new();

    // Separate the pipeline into "body" (repeated each iteration) and
    // "tail" (only emitted on the final iteration). A trailing Ref("out")
    // or Ref("in") in the pipeline is the tail — it's a terminus, not
    // something that should be chained through intermediate iterations.
    let (body_pipeline, tail_endpoint) = split_pipeline_tail(&unroll.pipeline);

    if body_pipeline.is_empty() && tail_endpoint.is_none() {
        return Some(vec![]);
    }

    let mut prev_source = conn.source.clone();

    for i in 0..count {
        let expanded_body: Vec<Endpoint> = body_pipeline
            .iter()
            .map(|ep| rewrite_endpoint_for_iteration(ep, i, expanded_context_names, next_id))
            .collect();

        let is_last_iteration = i == count - 1;

        // Chain: prev_source -> body[0] -> body[1] -> ... -> body[last]
        let mut current_source = prev_source.clone();
        for ep in &expanded_body {
            result.push(Connection {
                source: current_source.clone(),
                destination: ep.clone(),
            });
            current_source = ep.clone();
        }

        if !is_last_iteration {
            // Create intermediate temp ref for next iteration
            let temp_name = format!("_unroll_{}_{}", unroll.id, i);
            result.push(Connection {
                source: current_source,
                destination: Endpoint::Ref(PortRef::new(&temp_name)),
            });
            prev_source = Endpoint::Ref(PortRef::new(&temp_name));
        } else if let Some(tail) = &tail_endpoint {
            // Final iteration: connect to the tail endpoint (e.g., `out`)
            let rewritten_tail =
                rewrite_endpoint_for_iteration(tail, i, expanded_context_names, next_id);
            result.push(Connection {
                source: current_source,
                destination: rewritten_tail,
            });
        }
    }

    Some(result)
}

/// Split a pipeline into (body, optional_tail).
///
/// If the last element is a terminal Ref (like `out` or `in`), it's separated
/// as the tail so it's only emitted on the final iteration.
fn split_pipeline_tail(pipeline: &[Endpoint]) -> (&[Endpoint], Option<&Endpoint>) {
    if let Some(last) = pipeline.last() {
        if let Endpoint::Ref(port_ref) = last {
            if port_ref.node == "out" || port_ref.node == "in" {
                let body = &pipeline[..pipeline.len() - 1];
                return (body, Some(last));
            }
        }
    }
    (pipeline, None)
}

/// Rewrite an endpoint for a specific unroll iteration:
/// - Call endpoints get new unique IDs
/// - Ref endpoints matching expanded context names get suffixed
fn rewrite_endpoint_for_iteration(
    endpoint: &Endpoint,
    iteration: usize,
    expanded_context_names: &[(String, Vec<String>)],
    next_id: &mut usize,
) -> Endpoint {
    match endpoint {
        Endpoint::Call {
            name,
            args,
            kwargs,
            frozen,
            ..
        } => {
            let id = *next_id;
            *next_id += 1;
            Endpoint::Call {
                name: name.clone(),
                args: args.clone(),
                kwargs: kwargs.clone(),
                id,
                frozen: *frozen,
            }
        }
        Endpoint::Ref(port_ref) => {
            // Check if this ref matches an expanded context-unroll base name
            for (base_name, suffixed_names) in expanded_context_names {
                if port_ref.node == *base_name {
                    if suffixed_names.len() > 1 {
                        // N suffixed bindings: rewrite to iteration-specific name
                        if iteration < suffixed_names.len() {
                            return Endpoint::Ref(PortRef {
                                node: suffixed_names[iteration].clone(),
                                port: port_ref.port.clone(),
                            });
                        }
                    }
                    // Single name (@static): keep the same ref name but return as-is.
                    // The codegen will handle calling the same module multiple times
                    // because each connection through the chain creates a fresh call.
                    return endpoint.clone();
                }
            }
            endpoint.clone()
        }
        // Pass through other endpoint types unchanged
        _ => endpoint.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_param(name: &str, default: Option<i64>) -> Param {
        Param {
            name: name.to_string(),
            default: default.map(Value::Int),
        }
    }

    #[test]
    fn test_resolve_count_literal() {
        let params = vec![];
        assert_eq!(resolve_count(&Value::Int(3), &params, "Test"), Some(3));
        assert_eq!(resolve_count(&Value::Int(1), &params, "Test"), Some(1));
        assert_eq!(resolve_count(&Value::Int(0), &params, "Test"), None);
        assert_eq!(resolve_count(&Value::Int(-1), &params, "Test"), None);
    }

    #[test]
    fn test_resolve_count_param_ref() {
        let params = vec![make_param("num_layers", Some(6))];
        assert_eq!(
            resolve_count(&Value::Name("num_layers".to_string()), &params, "Test"),
            Some(6)
        );
        assert_eq!(
            resolve_count(&Value::Name("unknown".to_string()), &params, "Test"),
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
                        count: Value::Name("num_layers".to_string()),
                        bindings: vec![Binding {
                            name: "block".to_string(),
                            call_name: "TransformerBlock".to_string(),
                            args: vec![Value::Name("d_model".to_string())],
                            kwargs: vec![],
                            scope: Scope::Instance { lazy: false },
                            frozen: false,
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
        } else {
            panic!("Expected Graph body");
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
                        count: Value::Name("num_layers".to_string()),
                        bindings: vec![Binding {
                            name: "block".to_string(),
                            call_name: "TransformerBlock".to_string(),
                            args: vec![],
                            kwargs: vec![],
                            scope: Scope::Static,
                            frozen: false,
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
            // @static: single binding, no suffix
            assert_eq!(context_bindings.len(), 1);
            assert_eq!(context_bindings[0].name, "block");
        } else {
            panic!("Expected Graph body");
        }
    }

    #[test]
    fn test_expand_threaded_unroll() {
        let mut program = Program::new();
        program.neurons.insert(
            "Stack".to_string(),
            NeuronDef {
                name: "Stack".to_string(),
                params: vec![make_param("d_model", Some(512))],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![],
                    connections: vec![Connection {
                        source: Endpoint::Ref(PortRef::new("in")),
                        destination: Endpoint::Unroll(UnrollExpr {
                            count: Value::Int(3),
                            index_var: None,
                            pipeline: vec![
                                Endpoint::Call {
                                    name: "TransformerBlock".to_string(),
                                    args: vec![Value::Name("d_model".to_string())],
                                    kwargs: vec![],
                                    id: 0,
                                    frozen: false,
                                },
                                Endpoint::Ref(PortRef::new("out")),
                            ],
                            id: 100,
                        }),
                    }],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = expand_unrolls(&mut program);
        assert!(result.is_ok());

        let neuron = program.neurons.get("Stack").unwrap();
        if let NeuronBody::Graph { connections, .. } = &neuron.body {
            // Should have expanded into multiple connections
            // 3 iterations, each with 2 pipeline items (Call + out)
            // Iteration 0: in -> Call_0, Call_0 -> _unroll_100_0
            // Iteration 1: _unroll_100_0 -> Call_1, Call_1 -> _unroll_100_1
            // Iteration 2: _unroll_100_1 -> Call_2, Call_2 -> out
            assert!(connections.len() > 1, "Expected expanded connections, got {}", connections.len());

            // Verify no Unroll endpoints remain
            for conn in connections {
                assert!(
                    !has_unroll_endpoint(&conn.source),
                    "Source should not contain Unroll"
                );
                assert!(
                    !has_unroll_endpoint(&conn.destination),
                    "Destination should not contain Unroll"
                );
            }
        } else {
            panic!("Expected Graph body");
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
    fn test_count_one_degenerates() {
        let mut program = Program::new();
        program.neurons.insert(
            "Single".to_string(),
            NeuronDef {
                name: "Single".to_string(),
                params: vec![],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![],
                    connections: vec![Connection {
                        source: Endpoint::Ref(PortRef::new("in")),
                        destination: Endpoint::Unroll(UnrollExpr {
                            count: Value::Int(1),
                            index_var: None,
                            pipeline: vec![
                                Endpoint::Call {
                                    name: "Block".to_string(),
                                    args: vec![],
                                    kwargs: vec![],
                                    id: 0,
                                    frozen: false,
                                },
                                Endpoint::Ref(PortRef::new("out")),
                            ],
                            id: 200,
                        }),
                    }],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        let result = expand_unrolls(&mut program);
        assert!(result.is_ok());

        let neuron = program.neurons.get("Single").unwrap();
        if let NeuronBody::Graph { connections, .. } = &neuron.body {
            // count=1: single pass, no intermediate refs needed
            assert!(connections.len() >= 1);
            for conn in connections {
                assert!(!has_unroll_endpoint(&conn.destination));
            }
        } else {
            panic!("Expected Graph body");
        }
    }
}
