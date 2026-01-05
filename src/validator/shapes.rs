use crate::interfaces::*;
use std::collections::HashMap;

/// Check if two shapes are compatible
pub(super) fn shapes_compatible(source: &Shape, dest: &Shape) -> bool {
    // Find variadic dimensions in both shapes
    let source_variadic_pos = source
        .dims
        .iter()
        .position(|d| matches!(d, Dim::Variadic(_)));
    let dest_variadic_pos = dest.dims.iter().position(|d| matches!(d, Dim::Variadic(_)));

    match (source_variadic_pos, dest_variadic_pos) {
        // Both have variadics - complex case, for now assume compatible
        (Some(_), Some(_)) => {
            // Would need sophisticated matching, for now allow it
            true
        }
        // Source has variadic, dest does not
        (Some(var_pos), None) => match_variadic_shape(&source.dims, var_pos, &dest.dims),
        // Dest has variadic, source does not
        (None, Some(var_pos)) => match_variadic_shape(&dest.dims, var_pos, &source.dims),
        // Neither has variadic - must match exactly
        (None, None) => {
            if source.dims.len() != dest.dims.len() {
                return false;
            }
            // Check each dimension pair
            for (src_dim, dst_dim) in source.dims.iter().zip(dest.dims.iter()) {
                if !dims_compatible(src_dim, dst_dim) {
                    return false;
                }
            }
            true
        }
    }
}

/// Check if two dimensions are compatible
pub(super) fn dims_compatible(source: &Dim, dest: &Dim) -> bool {
    match (source, dest) {
        // Wildcards match anything
        (Dim::Wildcard, _) | (_, Dim::Wildcard) => true,
        // Variadics match anything (shouldn't reach here due to check above, but safe)
        (Dim::Variadic(_), _) | (_, Dim::Variadic(_)) => true,
        // Exact matches
        (Dim::Literal(a), Dim::Literal(b)) => a == b,
        // Named dimensions: assume compatible (parameter binding handles unification)
        // Full shape inference would need parameter context
        (Dim::Named(_), Dim::Named(_)) => true,
        // Expressions: would need evaluation context, for now assume compatible
        (Dim::Expr(_), _) | (_, Dim::Expr(_)) => true,
        // Mixed named/literal: incompatible (can't unify 512 with a variable)
        (Dim::Named(_), Dim::Literal(_)) | (Dim::Literal(_), Dim::Named(_)) => false,
        // Global dimensions: compatible if same name (conservative)
        (Dim::Global(n1), Dim::Global(n2)) => n1 == n2,
        (Dim::Global(_), _) | (_, Dim::Global(_)) => true,
    }
}

/// Match a shape with a variadic against a concrete shape
/// pattern_dims: the dims with a variadic at var_pos
/// concrete_dims: the dims without variadic
pub(super) fn match_variadic_shape(
    pattern_dims: &[Dim],
    var_pos: usize,
    concrete_dims: &[Dim],
) -> bool {
    // Pattern: [prefix..., *variadic, suffix...]
    // Concrete: [concrete_dims...]

    // Count non-variadic dimensions in pattern
    let pattern_fixed_count = pattern_dims.len() - 1; // Subtract 1 for the variadic

    // The variadic must match at least 0 dimensions
    // So concrete must have at least as many dims as pattern's fixed dims
    if concrete_dims.len() < pattern_fixed_count {
        return false;
    }

    // Match prefix (before variadic)
    for i in 0..var_pos {
        if !dims_compatible(&pattern_dims[i], &concrete_dims[i]) {
            return false;
        }
    }

    // Match suffix (after variadic)
    let suffix_count = pattern_dims.len() - var_pos - 1;
    let concrete_suffix_start = concrete_dims.len() - suffix_count;

    for i in 0..suffix_count {
        let pattern_idx = var_pos + 1 + i;
        let concrete_idx = concrete_suffix_start + i;
        if !dims_compatible(&pattern_dims[pattern_idx], &concrete_dims[concrete_idx]) {
            return false;
        }
    }

    // Everything matches - the variadic captures the middle portion
    true
}

/// Substitute parameter values in port shapes
/// For example: Linear(512, 256) binds in_dim=512, out_dim=256
/// Then [*, in_dim] becomes [*, 512]
pub(super) fn substitute_params(ports: &[Port], params: &[Param], args: &[Value]) -> Vec<Port> {
    // Build parameter binding map
    let mut bindings: HashMap<String, i64> = HashMap::new();
    for (param, arg) in params.iter().zip(args.iter()) {
        if let Value::Int(val) = arg {
            bindings.insert(param.name.clone(), *val);
        } else if let Value::Name(_name) = arg {
            // Named arguments remain as named dimensions
            // We could handle this better with a more sophisticated type system
            // For now, we don't substitute named parameters
            continue;
        }
    }

    // Substitute in each port
    ports
        .iter()
        .map(|port| Port {
            name: port.name.clone(),
            shape: substitute_shape(&port.shape, &bindings),
        })
        .collect()
}

/// Substitute parameter values in a shape
pub(super) fn substitute_shape(shape: &Shape, bindings: &HashMap<String, i64>) -> Shape {
    Shape {
        dims: shape
            .dims
            .iter()
            .map(|dim| substitute_dim(dim, bindings))
            .collect(),
    }
}

/// Substitute parameter values in a dimension
pub(super) fn substitute_dim(dim: &Dim, bindings: &HashMap<String, i64>) -> Dim {
    match dim {
        Dim::Named(name) => {
            if let Some(val) = bindings.get(name) {
                Dim::Literal(*val)
            } else {
                dim.clone()
            }
        }
        Dim::Expr(expr) => {
            // Recursively substitute in expressions
            let left = substitute_dim(&expr.left, bindings);
            let right = substitute_dim(&expr.right, bindings);

            // Try to evaluate if both sides are now literals
            if let (Dim::Literal(l), Dim::Literal(r)) = (&left, &right) {
                let result = match expr.op {
                    BinOp::Add => l + r,
                    BinOp::Sub => l - r,
                    BinOp::Mul => l * r,
                    BinOp::Div => l / r,
                    _ => {
                        // Non-arithmetic operations, keep as expression
                        return Dim::Expr(Box::new(DimExpr {
                            op: expr.op,
                            left,
                            right,
                        }));
                    }
                };
                Dim::Literal(result)
            } else {
                Dim::Expr(Box::new(DimExpr {
                    op: expr.op,
                    left,
                    right,
                }))
            }
        }
        Dim::Global(_) => dim.clone(),
        _ => dim.clone(),
    }
}

/// Check if pattern is catch-all (matches any input)
/// This is used for detecting unreachable match arms
pub fn is_catch_all_pattern(pattern: &Shape) -> bool {
    // A pattern is catch-all if:
    // 1. All dimensions are wildcards: [*, *, ...]
    // 2. It's a single variadic: [*shape]
    // 3. It has only wildcards and/or named dimensions (no literals)

    if pattern.dims.is_empty() {
        return false;
    }

    // Check for variadic - a pattern with a variadic can match any rank
    let has_variadic = pattern.dims.iter().any(|d| matches!(d, Dim::Variadic(_)));
    if has_variadic {
        // Variadic patterns are catch-all if they have no literals
        return !pattern.dims.iter().any(|d| matches!(d, Dim::Literal(_)));
    }

    // Non-variadic patterns are catch-all if all dims are wildcards or named (no literals/globals)
    pattern
        .dims
        .iter()
        .all(|d| matches!(d, Dim::Wildcard | Dim::Named(_)))
}

/// Check if pattern `general` subsumes (is more general than) pattern `specific`
/// If `general` subsumes `specific`, then `specific` is unreachable when placed after `general`
pub fn pattern_subsumes(general: &Shape, specific: &Shape) -> bool {
    // Variadic patterns subsume based on their prefix/suffix constraints
    let general_has_variadic = general.dims.iter().any(|d| matches!(d, Dim::Variadic(_)));
    let specific_has_variadic = specific.dims.iter().any(|d| matches!(d, Dim::Variadic(_)));

    match (general_has_variadic, specific_has_variadic) {
        (true, _) => {
            // General has variadic - it subsumes specific if prefix/suffix match or are more general
            // For MVP, conservatively say variadic patterns don't subsume non-variadic
            // (to avoid false positives)
            specific_has_variadic && variadic_patterns_compatible(general, specific)
        }
        (false, true) => {
            // Specific has variadic but general doesn't - no subsumption
            false
        }
        (false, false) => {
            // Neither has variadic - check rank and dimension-wise subsumption
            non_variadic_subsumes(general, specific)
        }
    }
}

/// Check if two variadic patterns are compatible (conservative check)
pub(super) fn variadic_patterns_compatible(p1: &Shape, p2: &Shape) -> bool {
    // For MVP, just check if they have the same structure
    // A full implementation would check prefix/suffix compatibility
    p1.dims.len() == p2.dims.len()
}

/// Check if non-variadic pattern `general` subsumes `specific`
pub(super) fn non_variadic_subsumes(general: &Shape, specific: &Shape) -> bool {
    // Different ranks - no subsumption
    if general.dims.len() != specific.dims.len() {
        return false;
    }

    // Check dimension by dimension
    for (g_dim, s_dim) in general.dims.iter().zip(specific.dims.iter()) {
        match (g_dim, s_dim) {
            // Wildcard matches anything
            (Dim::Wildcard, _) => continue,
            // Named dimensions match anything (they capture)
            (Dim::Named(_), _) => continue,
            // Literal must match exactly
            (Dim::Literal(g_lit), Dim::Literal(s_lit)) => {
                if g_lit != s_lit {
                    return false; // Different literals - not subsumed
                }
            }
            // Literal in general, but wildcard/named in specific - not subsumed
            (Dim::Literal(_), _) => return false,
            // Expression dimensions - conservative check
            (Dim::Expr(_), _) => continue, // TODO: implement expression subsumption
            // Variadic should have been handled above
            (Dim::Variadic(_), _) => continue,
            // Global (conservative check)
            (Dim::Global(g), Dim::Global(s)) => {
                if g != s {
                    return false;
                }
            }
            (Dim::Global(_), _) => return false,
        }
    }

    // All dimensions of general are as general or more general than specific
    true
}
