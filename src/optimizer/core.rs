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
        if let NeuronBody::Graph { connections, .. } = &mut neuron.body {
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
        Endpoint::If(if_expr) => {
            // Recurse into if/elif branches
            for branch in &mut if_expr.branches {
                for pipe_endpoint in &mut branch.pipeline {
                    count += optimize_endpoint(pipe_endpoint);
                }
            }
            // Recurse into else branch
            if let Some(else_branch) = &mut if_expr.else_branch {
                for pipe_endpoint in else_branch {
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
        if let NeuronBody::Graph { connections, .. } = &neuron.body {
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
        Endpoint::If(if_expr) => {
            // Recurse into if/elif branches
            for branch in &if_expr.branches {
                for pipe_endpoint in &branch.pipeline {
                    count += count_matches_in_endpoint(pipe_endpoint);
                }
            }
            // Recurse into else branch
            if let Some(else_branch) = &if_expr.else_branch {
                for pipe_endpoint in else_branch {
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
    if let Some(shape) = arm.pattern.as_shape() {
        for dim in &shape.dims {
            match dim {
                Dim::Literal(_) => score += 100,
                Dim::Named(_) => score += 10, // Named captures less specific than literals
                Dim::Expr(_) => score += 50,  // Expressions moderately specific
                Dim::Global(_) => score += 80, // Globals quite specific
                Dim::Wildcard => score += 1,  // Wildcards least specific
                Dim::Inferred => score += 1,  // Inferred (-1) least specific, like wildcard
                Dim::Variadic(_) => score += 0, // Variadics are catch-all
            }
        }
    }
    // NeuronContract patterns don't participate in specificity ordering

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
        if let NeuronBody::Graph { connections, .. } = &mut neuron.body {
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

    if let Endpoint::Match(match_expr) = endpoint {
        // Check if reordering would change anything
        let original_order: Vec<_> = match_expr.arms.iter().map(pattern_specificity).collect();

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
                other => other,
            }
        });

        // Check if order actually changed
        let new_order: Vec<_> = match_expr.arms.iter().map(pattern_specificity).collect();

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
    let all_concrete = input_shape.dims.iter().all(|dim| match dim {
        Dim::Literal(_) => true,
        Dim::Named(name) => ctx.resolved_dims.contains_key(name),
        Dim::Expr(expr) => ctx.evaluate_expr(expr).is_some(),
        _ => false,
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
        let arm_shape = match &arm.pattern {
            MatchPattern::Shape(s) => s,
            MatchPattern::NeuronContract(_) => continue,
        };
        if pattern_matches_shape(arm_shape, input_shape, ctx) {
            // If there's a guard, we need to evaluate it
            if let Some(guard) = &arm.guard {
                if try_evaluate_guard(guard, ctx) == Some(true) {
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
            Dim::Wildcard => continue,       // Always matches
            Dim::Inferred => continue,       // Inferred (-1) matches anything
            Dim::Variadic(_) => return true, // Matches rest of shape
            Dim::Global(pat_name) => {
                let concrete_name = match concrete_dim {
                    Dim::Global(n) => Some(n),
                    _ => None,
                };
                if concrete_name != Some(pat_name) {
                    return false;
                }
            }
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
fn try_evaluate_guard(guard: &Value, ctx: &InferenceContext) -> Option<bool> {
    match guard {
        Value::BinOp { op, left, right } => {
            let left_val = evaluate_value(left, ctx)?;
            let right_val = evaluate_value(right, ctx)?;

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
fn evaluate_value(val: &Value, ctx: &InferenceContext) -> Option<i64> {
    match val {
        Value::Int(n) => Some(*n),
        Value::Name(name) => {
            // Check if it's a resolved dimension
            ctx.resolved_dims.get(name).map(|v| *v as i64)
        }
        Value::BinOp { op, left, right } => {
            let left_val = evaluate_value(left, ctx)?;
            let right_val = evaluate_value(right, ctx)?;

            Some(match op {
                BinOp::Add => left_val + right_val,
                BinOp::Sub => left_val - right_val,
                BinOp::Mul => left_val * right_val,
                BinOp::Div => left_val / right_val,
                _ => return None,
            })
        }
        Value::Global(_) => None, // TODO: look up global
        _ => None,
    }
}

#[cfg(test)]
mod tests;
