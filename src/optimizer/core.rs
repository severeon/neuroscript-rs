use crate::interfaces::*;
use std::cmp::Ordering;

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

/// Calculate the specificity of a pattern for reordering.
/// More specific patterns should be checked first.
/// Returns a tuple (specificity_score, has_guard) for comparison.
/// Higher scores = more specific.
fn pattern_specificity(arm: &MatchArm) -> (usize, bool) {
    let mut score = 0;

    // Count literal dimensions (most specific)
    for dim in &arm.pattern.dims {
        match dim {
            Dim::Literal(_) => score += 100,
            Dim::Named(_) => score += 10,  // Named captures less specific than literals
            Dim::Expr(_) => score += 50,   // Expressions moderately specific
            Dim::Wildcard => score += 1,   // Wildcards least specific
            Dim::Variadic(_) => score += 0, // Variadics are catch-all
        }
    }

    // Guards add specificity
    let has_guard = arm.guard.is_some();

    (score, has_guard)
}

/// Reorder match arms to check more specific patterns first.
/// This improves performance by putting common specific cases before general wildcards.
/// Returns the number of match expressions that were reordered.
///
/// Reordering rules:
/// 1. More literal dimensions = higher priority
/// 2. Expressions > Named captures > Wildcards
/// 3. Guards add slight priority (checked after pattern specificity)
/// 4. Maintain relative order of equally-specific patterns (stable sort)
pub fn reorder_match_arms(program: &mut Program) -> usize {
    let mut reordered_count = 0;

    for neuron in program.neurons.values_mut() {
        if let NeuronBody::Graph(connections) = &mut neuron.body {
            for connection in connections {
                reordered_count += reorder_endpoint(&mut connection.source);
                reordered_count += reorder_endpoint(&mut connection.destination);
            }
        }
    }

    reordered_count
}

fn reorder_endpoint(endpoint: &mut Endpoint) -> usize {
    let mut count = 0;

    match endpoint {
        Endpoint::Match(match_expr) => {
            // Check if reordering would change anything
            let original_order: Vec<_> = match_expr.arms.iter()
                .map(pattern_specificity)
                .collect();

            // Stable sort by specificity (descending - most specific first)
            match_expr.arms.sort_by(|a, b| {
                let spec_a = pattern_specificity(a);
                let spec_b = pattern_specificity(b);

                // Compare specificity scores first (higher = more specific)
                match spec_b.0.cmp(&spec_a.0) {
                    Ordering::Equal => {
                        // If equal specificity, guards come first
                        spec_b.1.cmp(&spec_a.1)
                    }
                    other => other
                }
            });

            // Check if order actually changed
            let new_order: Vec<_> = match_expr.arms.iter()
                .map(pattern_specificity)
                .collect();

            if original_order != new_order {
                count += 1;
            }

            // Recurse into arms
            for arm in &mut match_expr.arms {
                for pipe_endpoint in &mut arm.pipeline {
                    count += reorder_endpoint(pipe_endpoint);
                }
            }
        }
        _ => {}
    }

    count
}

/// Attempt to resolve a match expression at compile-time using static shape information.
/// If all dimensions in the input shape are known (literals), we can select the matching arm
/// at compile time and replace the entire match with a direct pipeline.
///
/// This returns:
/// - Some(arm_index) if the match can be resolved to a specific arm
/// - None if runtime checking is required
pub fn try_static_resolve(
    match_expr: &MatchExpr,
    input_shape: &Shape,
    ctx: &InferenceContext,
) -> Option<usize> {
    // Check if input shape is fully concrete (all dimensions resolved)
    let all_concrete = input_shape.dims.iter().all(|dim| {
        match dim {
            Dim::Literal(_) => true,
            Dim::Named(name) => ctx.resolved_dims.contains_key(name),
            Dim::Expr(expr) => ctx.evaluate_expr(expr).is_some(),
            _ => false,
        }
    });

    if !all_concrete {
        return None; // Need runtime check
    }

    // Try each arm in order
    for (i, arm) in match_expr.arms.iter().enumerate() {
        if !arm.is_reachable {
            continue;
        }

        // Check if pattern matches the concrete shape
        if pattern_matches_shape(&arm.pattern, input_shape, ctx) {
            // If there's a guard, we need to evaluate it
            if let Some(guard) = &arm.guard {
                if try_evaluate_guard(guard, input_shape, ctx) == Some(true) {
                    return Some(i);
                }
                // Guard failed, try next arm
            } else {
                // No guard, this arm matches
                return Some(i);
            }
        }
    }

    None // No arm matched (shouldn't happen if validation passed)
}

/// Check if a pattern matches a concrete shape
fn pattern_matches_shape(pattern: &Shape, concrete: &Shape, ctx: &InferenceContext) -> bool {
    // Handle variadic patterns
    let has_variadic = pattern.dims.iter().any(|d| matches!(d, Dim::Variadic(_)));

    if !has_variadic && pattern.dims.len() != concrete.dims.len() {
        return false;
    }

    for (pat_dim, concrete_dim) in pattern.dims.iter().zip(&concrete.dims) {
        match pat_dim {
            Dim::Wildcard => continue, // Always matches
            Dim::Variadic(_) => return true, // Matches rest of shape
            Dim::Literal(pat_val) => {
                let concrete_val = match concrete_dim {
                    Dim::Literal(v) => Some(*v as usize),
                    Dim::Named(n) => ctx.resolved_dims.get(n).copied(),
                    Dim::Expr(e) => ctx.evaluate_expr(e),
                    _ => None,
                };

                if concrete_val != Some(*pat_val as usize) {
                    return false;
                }
            }
            Dim::Named(_) => continue, // Capture matches anything
            Dim::Expr(pat_expr) => {
                let pat_val = ctx.evaluate_expr(pat_expr);
                let concrete_val = match concrete_dim {
                    Dim::Literal(v) => Some(*v as usize),
                    Dim::Named(n) => ctx.resolved_dims.get(n).copied(),
                    Dim::Expr(e) => ctx.evaluate_expr(e),
                    _ => None,
                };

                if pat_val.is_none() || concrete_val.is_none() || pat_val != concrete_val {
                    return false;
                }
            }
        }
    }

    true
}

/// Try to evaluate a guard condition at compile-time
fn try_evaluate_guard(guard: &Value, shape: &Shape, ctx: &InferenceContext) -> Option<bool> {
    match guard {
        Value::BinOp { op, left, right } => {
            let left_val = evaluate_value(left, shape, ctx)?;
            let right_val = evaluate_value(right, shape, ctx)?;

            Some(match op {
                BinOp::Eq => left_val == right_val,
                BinOp::Ne => left_val != right_val,
                BinOp::Lt => left_val < right_val,
                BinOp::Gt => left_val > right_val,
                BinOp::Le => left_val <= right_val,
                BinOp::Ge => left_val >= right_val,
                _ => return None, // Not a comparison
            })
        }
        _ => None,
    }
}

/// Evaluate a value expression at compile-time
fn evaluate_value(val: &Value, shape: &Shape, ctx: &InferenceContext) -> Option<i64> {
    match val {
        Value::Int(n) => Some(*n),
        Value::Name(name) => {
            // Check if it's a resolved dimension
            ctx.resolved_dims.get(name).map(|v| *v as i64)
        }
        Value::BinOp { op, left, right } => {
            let left_val = evaluate_value(left, shape, ctx)?;
            let right_val = evaluate_value(right, shape, ctx)?;

            Some(match op {
                BinOp::Add => left_val + right_val,
                BinOp::Sub => left_val - right_val,
                BinOp::Mul => left_val * right_val,
                BinOp::Div => left_val / right_val,
                _ => return None,
            })
        }
        _ => None,
    }
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
            body: NeuronBody::Graph(vec![connection]),
        };

        program.neurons.insert("ReorderTest".to_string(), neuron);

        let reordered = reorder_match_arms(&mut program);
        assert_eq!(reordered, 1, "Should reorder 1 match expression");

        // Verify order is now correct (specific first)
        let neuron = program.neurons.get("ReorderTest").unwrap();
        if let NeuronBody::Graph(connections) = &neuron.body {
            if let Endpoint::Match(match_expr) = &connections[0].destination {
                // First arm should now be the specific pattern
                assert!(matches!(match_expr.arms[0].pattern.dims[0], Dim::Literal(2)));
                assert!(matches!(match_expr.arms[0].pattern.dims[1], Dim::Literal(512)));

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
        };

        // Input shape: [batch, dim] where both are resolved
        let input_shape = Shape {
            dims: vec![Dim::Named("batch".to_string()), Dim::Named("dim".to_string())],
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
        };

        // Input shape has concrete first dimension and resolved second dimension
        let input_shape = Shape {
            dims: vec![Dim::Named("batch".to_string()), Dim::Named("d".to_string())],
        };

        let resolved_arm = try_static_resolve(&match_expr, &input_shape, &ctx);
        assert_eq!(resolved_arm, Some(0), "Should resolve to arm 0 (guard true: 1024 > 512)");
    }

    #[test]
    fn test_static_resolve_runtime_needed() {
        // Test that runtime check is required when shape is not fully concrete
        let ctx = InferenceContext::default();

        let match_expr = MatchExpr {
            arms: vec![
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Wildcard, Dim::Literal(512)],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: true,
                },
            ],
        };

        // Input shape has unresolved dimension
        let input_shape = Shape {
            dims: vec![Dim::Wildcard, Dim::Named("unknown".to_string())],
        };

        let resolved_arm = try_static_resolve(&match_expr, &input_shape, &ctx);
        assert_eq!(resolved_arm, None, "Should require runtime check (unknown dimension)");
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

        let shape = Shape {
            dims: vec![Dim::Wildcard, Dim::Named("d".to_string())],
        };

        // Test greater than
        let guard1 = Value::BinOp {
            op: BinOp::Gt,
            left: Box::new(Value::Name("d".to_string())),
            right: Box::new(Value::Int(512)),
        };
        assert_eq!(try_evaluate_guard(&guard1, &shape, &ctx), Some(true));

        // Test less than
        let guard2 = Value::BinOp {
            op: BinOp::Lt,
            left: Box::new(Value::Name("d".to_string())),
            right: Box::new(Value::Int(2048)),
        };
        assert_eq!(try_evaluate_guard(&guard2, &shape, &ctx), Some(true));

        // Test equality
        let guard3 = Value::BinOp {
            op: BinOp::Eq,
            left: Box::new(Value::Name("d".to_string())),
            right: Box::new(Value::Int(1024)),
        };
        assert_eq!(try_evaluate_guard(&guard3, &shape, &ctx), Some(true));
    }
}
