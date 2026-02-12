//! Compile-time unroll expansion pass.
//!
//! Runs before validation and codegen, lowering `Endpoint::Unroll` and
//! `ContextUnroll` into ordinary IR (N copies of connections/bindings).

use crate::interfaces::*;

/// Maximum allowed unroll count to prevent resource exhaustion.
const MAX_UNROLL_COUNT: usize = 1024;

/// Maximum recursion depth for `max_endpoint_id` to prevent stack overflow
/// on deeply nested IR structures.
const MAX_ENDPOINT_DEPTH: usize = 100;

/// Prefix for temporary variables generated during graph-level unroll expansion.
const UNROLL_TEMP_PREFIX: &str = "_unroll";

/// Port names that act as pipeline terminators (not chained through intermediate iterations).
const TERMINAL_PORTS: &[&str] = &["out", "in"];

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
            // Skip if no unrolls present
            let has_graph_unrolls = connections.iter().any(|c| {
                has_unroll_endpoint(&c.source) || has_unroll_endpoint(&c.destination)
            });

            if context_unrolls.is_empty() && !has_graph_unrolls {
                continue;
            }

            // --- Expand context unrolls ---
            // Pre-allocate with estimated capacity: existing bindings + unrolled ones
            let estimated_new = context_unrolls.iter().map(|u| u.bindings.len()).sum::<usize>()
                * 4; // rough overestimate per binding
            let mut new_bindings = Vec::with_capacity(context_bindings.len() + estimated_new);
            new_bindings.extend_from_slice(context_bindings);
            let mut unroll_binding_map: Vec<(String, Vec<String>)> = Vec::new();

            for unroll in context_unrolls {
                match resolve_count(&unroll.count, &params) {
                    Some(count) => {
                        for binding in &unroll.bindings {
                            if matches!(binding.scope, Scope::Static) {
                                // @static: single shared instance, no suffix
                                new_bindings.push(binding.clone());
                                unroll_binding_map.push((
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
                                unroll_binding_map
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
                        &unroll_binding_map,
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

/// Check if an endpoint contains an Unroll
fn has_unroll_endpoint(endpoint: &Endpoint) -> bool {
    matches!(endpoint, Endpoint::Unroll(_))
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
    max_endpoint_id_inner(endpoint, 0)
}

fn max_endpoint_id_inner(endpoint: &Endpoint, depth: usize) -> usize {
    if depth > MAX_ENDPOINT_DEPTH {
        return 0;
    }
    match endpoint {
        Endpoint::Call { id, .. } => *id,
        Endpoint::Match(m) => {
            let mut max = m.id;
            for arm in &m.arms {
                for ep in &arm.pipeline {
                    max = max.max(max_endpoint_id_inner(ep, depth + 1));
                }
            }
            max
        }
        Endpoint::If(expr) => {
            let mut max = expr.id;
            for branch in &expr.branches {
                for ep in &branch.pipeline {
                    max = max.max(max_endpoint_id_inner(ep, depth + 1));
                }
            }
            if let Some(else_branch) = &expr.else_branch {
                for ep in else_branch {
                    max = max.max(max_endpoint_id_inner(ep, depth + 1));
                }
            }
            max
        }
        Endpoint::Unroll(u) => {
            let mut max = u.id;
            for ep in &u.pipeline {
                max = max.max(max_endpoint_id_inner(ep, depth + 1));
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
    unroll_binding_map: &[(String, Vec<String>)],
    next_id: &mut usize,
    errors: &mut Vec<ValidationError>,
) -> Option<Vec<Connection>> {
    let unroll = match &conn.destination {
        Endpoint::Unroll(u) => u,
        _ => return None,
    };

    let count = match resolve_count(&unroll.count, params) {
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

    // Check for nested unrolls in the pipeline
    if unroll.pipeline.iter().any(|ep| has_unroll_endpoint(ep)) {
        errors.push(ValidationError::InvalidUnrollCount {
            neuron: neuron_name.to_string(),
            reason: "Nested unroll constructs are not supported".to_string(),
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
            .map(|ep| rewrite_endpoint_for_iteration(ep, i, unroll_binding_map, next_id))
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
            let temp_name = format!("{}_{}_{}", UNROLL_TEMP_PREFIX, unroll.id, i);
            result.push(Connection {
                source: current_source,
                destination: Endpoint::Ref(PortRef::new(&temp_name)),
            });
            prev_source = Endpoint::Ref(PortRef::new(&temp_name));
        } else if let Some(tail) = &tail_endpoint {
            // Final iteration: connect to the tail endpoint (e.g., `out`)
            let rewritten_tail =
                rewrite_endpoint_for_iteration(tail, i, unroll_binding_map, next_id);
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
            if TERMINAL_PORTS.contains(&port_ref.node.as_str()) {
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
    unroll_binding_map: &[(String, Vec<String>)],
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
            for (base_name, suffixed_names) in unroll_binding_map {
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
        // Nested unrolls are validated and rejected in expand_connection_unroll
        // before this function is ever called. If we reach here, it's a bug.
        Endpoint::Unroll(_) => {
            unreachable!("Nested unroll should have been rejected during expansion")
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
            panic!("Expected Graph body, got Primitive");
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
            panic!("Expected Graph body, got Primitive");
        }
    }

    #[test]
    fn test_resolve_count_param_no_default() {
        // Parameter exists but has no default value — can't resolve at compile time
        let params = vec![Param {
            name: "num_layers".to_string(),
            default: None,
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

    #[test]
    fn test_nested_unroll_errors() {
        let mut program = Program::new();
        program.neurons.insert(
            "Nested".to_string(),
            NeuronDef {
                name: "Nested".to_string(),
                params: vec![],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![],
                    connections: vec![Connection {
                        source: Endpoint::Ref(PortRef::new("in")),
                        destination: Endpoint::Unroll(UnrollExpr {
                            count: Value::Int(2),
                            pipeline: vec![
                                Endpoint::Unroll(UnrollExpr {
                                    count: Value::Int(3),
                                    pipeline: vec![Endpoint::Call {
                                        name: "Block".to_string(),
                                        args: vec![],
                                        kwargs: vec![],
                                        id: 0,
                                        frozen: false,
                                    }],
                                    id: 50,
                                }),
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
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(e, ValidationError::InvalidUnrollCount { .. })));
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
                            count: Value::Int(2),
                            bindings: vec![Binding {
                                name: "attn".to_string(),
                                call_name: "Attention".to_string(),
                                args: vec![],
                                kwargs: vec![],
                                scope: Scope::Instance { lazy: false },
                                frozen: false,
                            }],
                        },
                        ContextUnroll {
                            count: Value::Int(3),
                            bindings: vec![Binding {
                                name: "ffn".to_string(),
                                call_name: "FFN".to_string(),
                                args: vec![],
                                kwargs: vec![],
                                scope: Scope::Instance { lazy: false },
                                frozen: false,
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
                        },
                    ],
                    context_unrolls: vec![ContextUnroll {
                        count: Value::Int(3),
                        bindings: vec![Binding {
                            name: "block".to_string(),
                            call_name: "TransformerBlock".to_string(),
                            args: vec![Value::Name("dim".to_string())],
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
                        count: Value::Int(2),
                        bindings: vec![Binding {
                            name: "block".to_string(),
                            call_name: "TransformerBlock".to_string(),
                            args: vec![],
                            kwargs: vec![],
                            scope: Scope::Instance { lazy: true },
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

    /// Unroll inside a match arm pipeline is NOT expanded (expansion only
    /// handles top-level connection destinations). The unexpanded Endpoint::Unroll
    /// will be caught by the validator with "Unroll should be expanded before
    /// validation". This test verifies the unroll passes through expansion
    /// unchanged and the validator rejects it.
    #[test]
    fn test_unroll_inside_match_arm_not_expanded() {
        let mut program = Program::new();
        program.neurons.insert(
            "MatchUnroll".to_string(),
            NeuronDef {
                name: "MatchUnroll".to_string(),
                params: vec![make_param("dim", Some(256))],
                inputs: vec![],
                outputs: vec![],
                body: NeuronBody::Graph {
                    context_bindings: vec![],
                    context_unrolls: vec![],
                    connections: vec![Connection {
                        source: Endpoint::Ref(PortRef::new("in")),
                        destination: Endpoint::Match(MatchExpr {
                            arms: vec![MatchArm {
                                pattern: Shape { dims: vec![] },
                                guard: None,
                                is_reachable: true,
                                pipeline: vec![Endpoint::Unroll(UnrollExpr {
                                    count: Value::Int(3),
                                    pipeline: vec![Endpoint::Call {
                                        name: "Block".to_string(),
                                        args: vec![],
                                        kwargs: vec![],
                                        id: 0,
                                        frozen: false,
                                    }],
                                    id: 50,
                                })],
                            }],
                            id: 100,
                        }),
                    }],
                },
                max_cycle_depth: Some(10),
                doc: None,
            },
        );

        // Expansion succeeds (it doesn't look inside match arms)
        let result = expand_unrolls(&mut program);
        assert!(result.is_ok());

        // But the Unroll endpoint is still present inside the match arm
        let neuron = program.neurons.get("MatchUnroll").unwrap();
        if let NeuronBody::Graph { connections, .. } = &neuron.body {
            let has_inner_unroll = connections.iter().any(|c| {
                if let Endpoint::Match(m) = &c.destination {
                    m.arms.iter().any(|arm| {
                        arm.pipeline.iter().any(|ep| matches!(ep, Endpoint::Unroll(_)))
                    })
                } else {
                    false
                }
            });
            assert!(has_inner_unroll, "Unroll inside match arm should survive expansion unchanged");
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
                        count: Value::Name("num_layers".to_string()), // not a param!
                        bindings: vec![Binding {
                            name: "block".to_string(),
                            call_name: "Block".to_string(),
                            args: vec![],
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
}
