use super::*;
use crate::interfaces::*;
use num_bigint::BigUint;
use num_traits::{One, Zero};

// ========================================
// Helper Functions
// ========================================

fn wildcard() -> Shape {
    Shape::new(vec![Dim::Wildcard])
}

fn literal_shape(dims: Vec<i64>) -> Shape {
    Shape::new(dims.into_iter().map(Dim::Literal).collect())
}

fn named_shape(names: Vec<&str>) -> Shape {
    Shape::new(
        names
            .into_iter()
            .map(|n| Dim::Named(n.to_string()))
            .collect(),
    )
}

// ========================================
// Shape Algebra Tests
// ========================================

// ========================================
// 1. Basic Shape Operations Tests
// ========================================

#[test]
fn test_shape_new_and_rank() {
    let s = Shape::new(vec![Dim::Literal(2), Dim::Literal(3), Dim::Literal(4)]);
    assert_eq!(s.rank(), 3);
    assert_eq!(
        s.dims,
        vec![Dim::Literal(2), Dim::Literal(3), Dim::Literal(4)]
    );
}

#[test]
fn test_shape_empty() {
    let s = Shape::new(vec![]);
    assert_eq!(s.rank(), 0);
    assert_eq!(s.size(), Some(BigUint::one()));
}

#[test]
fn test_shape_size_basic() {
    let s = Shape::new(vec![Dim::Literal(2), Dim::Literal(3), Dim::Literal(4)]);
    assert_eq!(s.size(), Some(BigUint::from(24u32)));
}

#[test]
fn test_shape_size_with_zero() {
    let s = Shape::new(vec![Dim::Literal(2), Dim::Literal(0), Dim::Literal(4)]);
    assert_eq!(s.size(), Some(BigUint::zero()));
}

#[test]
fn test_shape_size_large() {
    // Test BigUint prevents overflow: 1000 * 1000 * 1000
    let s = Shape::new(vec![
        Dim::Literal(1000),
        Dim::Literal(1000),
        Dim::Literal(1000),
    ]);
    assert_eq!(s.size(), Some(BigUint::from(1_000_000_000u64)));
}

#[test]
fn test_shape_size_very_large() {
    // Test very large product that would overflow usize on 32-bit
    let s = Shape::new(vec![
        Dim::Literal(65536),
        Dim::Literal(65536),
        Dim::Literal(100),
    ]);
    let expected = BigUint::from(65536u64) * BigUint::from(65536u64) * BigUint::from(100u64);
    assert_eq!(s.size(), Some(expected));
}

// ========================================
// 3. Shape Property Tests
// ========================================

#[test]
fn test_flatten_multidim() {
    let s = Shape::new(vec![Dim::Literal(2), Dim::Literal(3), Dim::Literal(4)]);
    let flat = s.flatten();
    assert_eq!(flat, Some(Shape::new(vec![Dim::Literal(24)])));
}

#[test]
fn test_flatten_already_flat() {
    let s = Shape::new(vec![Dim::Literal(24)]);
    let flat = s.flatten();
    assert_eq!(flat, Some(Shape::new(vec![Dim::Literal(24)])));
}

#[test]
fn test_flatten_empty() {
    let s = Shape::new(vec![]);
    let flat = s.flatten();
    assert_eq!(flat, Some(Shape::new(vec![Dim::Literal(1)]))); // Empty shape has size 1 (product of no dims)
}

#[test]
fn test_flatten_named_dims_returns_none() {
    let s = Shape::new(vec![Dim::Named("batch".to_string()), Dim::Literal(256)]);
    assert_eq!(s.flatten(), None); // Cannot flatten when dimensions are unknown
}

#[test]
fn test_flatten_wildcard_returns_none() {
    let s = Shape::new(vec![Dim::Wildcard, Dim::Literal(64)]);
    assert_eq!(s.flatten(), None); // Cannot flatten when dimensions are unknown
}

// ========================================
// Shape Inference Tests
// ========================================

#[test]
fn test_wildcard_unification() {
    let mut ctx = InferenceContext::new();
    let engine = ShapeInferenceEngine::new();

    // Wildcard matches literal
    let s1 = wildcard();
    let s2 = literal_shape(vec![512]);
    assert!(engine.unify_shapes(&s1, &s2, &mut ctx).is_ok());

    // Wildcard matches named dimension
    let s1 = wildcard();
    let s2 = named_shape(vec!["dim"]);
    assert!(engine.unify_shapes(&s1, &s2, &mut ctx).is_ok());
}

#[test]
fn test_rank_mismatch() {
    let mut ctx = InferenceContext::new();
    let engine = ShapeInferenceEngine::new();

    let s1 = literal_shape(vec![512]);
    let s2 = literal_shape(vec![512, 256]);

    let result = engine.unify_shapes(&s1, &s2, &mut ctx);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Rank mismatch"));
}

#[test]
fn test_dimension_unification() {
    let mut ctx = InferenceContext::new();
    let engine = ShapeInferenceEngine::new();

    // Named dimensions should unify
    let s1 = named_shape(vec!["batch", "dim"]);
    let s2 = named_shape(vec!["batch", "dim"]);
    assert!(engine.unify_shapes(&s1, &s2, &mut ctx).is_ok());

    // Named dimension unifies with literal
    let mut ctx = InferenceContext::new();
    let s1 = named_shape(vec!["dim"]);
    let s2 = literal_shape(vec![512]);
    assert!(engine.unify_shapes(&s1, &s2, &mut ctx).is_ok());
    assert_eq!(ctx.resolved_dims.get("dim"), Some(&512));
}

#[test]
fn test_dimension_conflict() {
    let mut ctx = InferenceContext::new();
    let engine = ShapeInferenceEngine::new();

    // First bind "dim" to 512
    let s1 = named_shape(vec!["dim"]);
    let s2 = literal_shape(vec![512]);
    assert!(engine.unify_shapes(&s1, &s2, &mut ctx).is_ok());

    // Try to bind "dim" to 256 - should fail
    let s3 = named_shape(vec!["dim"]);
    let s4 = literal_shape(vec![256]);
    let result = engine.unify_shapes(&s3, &s4, &mut ctx);
    assert!(result.is_err());
}

#[test]
fn test_variadic_matching() {
    let mut ctx = InferenceContext::new();
    let engine = ShapeInferenceEngine::new();

    // Pattern [dim, *rest] matches [512, 256, 128]
    let pattern = Shape::new(vec![
        Dim::Named("dim".to_string()),
        Dim::Variadic("rest".to_string()),
    ]);
    let concrete = literal_shape(vec![512, 256, 128]);

    assert!(engine.unify_shapes(&pattern, &concrete, &mut ctx).is_ok());
    assert_eq!(ctx.resolved_dims.get("dim"), Some(&512));
}

#[test]
fn test_has_variadic() {
    let engine = ShapeInferenceEngine::new();

    let s1 = Shape::new(vec![Dim::Variadic("rest".to_string())]);
    assert!(engine.has_variadic(&s1));

    let s2 = literal_shape(vec![512]);
    assert!(!engine.has_variadic(&s2));
}

#[test]
fn test_expr_constraint_solving_multiply() {
    let mut ctx = InferenceContext::new();

    // Test: dim * 4 = 2048  =>  dim = 512
    let expr = DimExpr {
        left: Dim::Named("dim".to_string()),
        op: BinOp::Mul,
        right: Dim::Literal(4),
    };

    assert!(ctx.solve_expr_for_unknown(&expr, 2048).is_ok());
    assert_eq!(ctx.resolved_dims.get("dim"), Some(&512));
}

#[test]
fn test_expr_constraint_solving_divide() {
    let mut ctx = InferenceContext::new();

    // Test: dim / 2 = 256  =>  dim = 512
    let expr = DimExpr {
        left: Dim::Named("dim".to_string()),
        op: BinOp::Div,
        right: Dim::Literal(2),
    };

    assert!(ctx.solve_expr_for_unknown(&expr, 256).is_ok());
    assert_eq!(ctx.resolved_dims.get("dim"), Some(&512));
}

#[test]
fn test_expr_constraint_solving_add() {
    let mut ctx = InferenceContext::new();

    // Test: dim + 100 = 612  =>  dim = 512
    let expr = DimExpr {
        left: Dim::Named("dim".to_string()),
        op: BinOp::Add,
        right: Dim::Literal(100),
    };

    assert!(ctx.solve_expr_for_unknown(&expr, 612).is_ok());
    assert_eq!(ctx.resolved_dims.get("dim"), Some(&512));
}

#[test]
fn test_expr_constraint_solving_subtract() {
    let mut ctx = InferenceContext::new();

    // Test: dim - 100 = 412  =>  dim = 512
    let expr = DimExpr {
        left: Dim::Named("dim".to_string()),
        op: BinOp::Sub,
        right: Dim::Literal(100),
    };

    assert!(ctx.solve_expr_for_unknown(&expr, 412).is_ok());
    assert_eq!(ctx.resolved_dims.get("dim"), Some(&512));
}

#[test]
fn test_expr_constraint_solving_right_operand() {
    let mut ctx = InferenceContext::new();

    // Test: 2048 / dim = 4  =>  dim = 512
    let expr = DimExpr {
        left: Dim::Literal(2048),
        op: BinOp::Div,
        right: Dim::Named("dim".to_string()),
    };

    assert!(ctx.solve_expr_for_unknown(&expr, 4).is_ok());
    assert_eq!(ctx.resolved_dims.get("dim"), Some(&512));
}

#[test]
fn test_expr_constraint_solving_invalid_division() {
    let mut ctx = InferenceContext::new();

    // Test: dim * 3 = 512  =>  Error (512 not divisible by 3)
    let expr = DimExpr {
        left: Dim::Named("dim".to_string()),
        op: BinOp::Mul,
        right: Dim::Literal(3),
    };

    let result = ctx.solve_expr_for_unknown(&expr, 512);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not divisible"));
}

#[test]
fn test_unify_expr_with_literal() {
    let mut ctx = InferenceContext::new();

    // Test unification: [dim * 4] unified with [2048] should solve for dim = 512
    let expr_dim = Dim::Expr(Box::new(DimExpr {
        left: Dim::Named("dim".to_string()),
        op: BinOp::Mul,
        right: Dim::Literal(4),
    }));

    let literal_dim = Dim::Literal(2048);

    assert!(ctx.unify(&expr_dim, &literal_dim).is_ok());
    assert_eq!(ctx.resolved_dims.get("dim"), Some(&512));
}

#[test]
fn test_is_dim_resolvable() {
    let mut ctx = InferenceContext::new();

    // Literal is always resolvable
    assert!(is_dim_resolvable(&Dim::Literal(512), &ctx));

    // Wildcard is always resolvable
    assert!(is_dim_resolvable(&Dim::Wildcard, &ctx));

    // Named dimension not yet resolved
    assert!(!is_dim_resolvable(&Dim::Named("dim".to_string()), &ctx));

    // Resolve it
    ctx.resolved_dims.insert("dim".to_string(), 512);
    assert!(is_dim_resolvable(&Dim::Named("dim".to_string()), &ctx));

    // Expression with resolvable operands
    let expr = Dim::Expr(Box::new(DimExpr {
        left: Dim::Named("dim".to_string()),
        op: BinOp::Mul,
        right: Dim::Literal(4),
    }));
    assert!(is_dim_resolvable(&expr, &ctx));

    // Expression with unresolvable operand
    let expr2 = Dim::Expr(Box::new(DimExpr {
        left: Dim::Named("unknown".to_string()),
        op: BinOp::Mul,
        right: Dim::Literal(4),
    }));
    assert!(!is_dim_resolvable(&expr2, &ctx));
}

#[test]
fn test_match_pattern_unification() {
    let mut ctx = InferenceContext::new();
    let engine = ShapeInferenceEngine::new();

    // Pattern [*, d] should match concrete shape [32, 512]
    let pattern = Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]);
    let concrete = literal_shape(vec![32, 512]);

    assert!(engine
        .unify_pattern_with_shape(&pattern, &concrete, &mut ctx)
        .is_ok());
    assert_eq!(ctx.resolved_dims.get("d"), Some(&512));
}

#[test]
fn test_match_pattern_literal_mismatch() {
    let mut ctx = InferenceContext::new();
    let engine = ShapeInferenceEngine::new();

    // Pattern [*, 512] should NOT match [32, 256]
    let pattern = Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]);
    let concrete = literal_shape(vec![32, 256]);

    let result = engine.unify_pattern_with_shape(&pattern, &concrete, &mut ctx);
    assert!(result.is_err());
}

#[test]
fn test_match_pattern_rank_mismatch() {
    let mut ctx = InferenceContext::new();
    let engine = ShapeInferenceEngine::new();

    // Pattern [*, d] should NOT match 3D shape [32, 64, 512]
    let pattern = Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]);
    let concrete = literal_shape(vec![32, 64, 512]);

    let result = engine.unify_pattern_with_shape(&pattern, &concrete, &mut ctx);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Rank mismatch"));
}

#[test]
fn test_match_variadic_pattern() {
    let mut ctx = InferenceContext::new();
    let engine = ShapeInferenceEngine::new();

    // Pattern [*batch, d] should match [32, 64, 128, 512]
    let pattern = Shape::new(vec![
        Dim::Variadic("batch".to_string()),
        Dim::Named("d".to_string()),
    ]);
    let concrete = literal_shape(vec![32, 64, 128, 512]);

    assert!(engine
        .unify_pattern_with_shape(&pattern, &concrete, &mut ctx)
        .is_ok());
    assert_eq!(ctx.resolved_dims.get("d"), Some(&512));
}

#[test]
fn test_shapes_compatible() {
    let engine = ShapeInferenceEngine::new();

    // Compatible: same literal shapes
    let s1 = literal_shape(vec![512, 256]);
    let s2 = literal_shape(vec![512, 256]);
    assert!(engine.shapes_compatible(&s1, &s2));

    // Compatible: named dimensions can unify
    let s3 = named_shape(vec!["d1", "d2"]);
    let s4 = literal_shape(vec![512, 256]);
    assert!(engine.shapes_compatible(&s3, &s4));

    // Incompatible: different literals
    let s5 = literal_shape(vec![512, 256]);
    let s6 = literal_shape(vec![512, 128]);
    assert!(!engine.shapes_compatible(&s5, &s6));

    // Incompatible: rank mismatch
    let s7 = literal_shape(vec![512]);
    let s8 = literal_shape(vec![512, 256]);
    assert!(!engine.shapes_compatible(&s7, &s8));
}

#[test]
fn test_both_variadic_unification() {
    let engine = ShapeInferenceEngine::new();

    // Compatible: [A, *x, B] vs [A, *y, C, B] — C absorbed by *x
    let s1 = Shape::new(vec![
        Dim::Literal(32),
        Dim::Variadic("x".to_string()),
        Dim::Literal(64),
    ]);
    let s2 = Shape::new(vec![
        Dim::Literal(32),
        Dim::Variadic("y".to_string()),
        Dim::Literal(128),
        Dim::Literal(64),
    ]);
    assert!(engine.shapes_compatible(&s1, &s2));

    // Compatible: [*x, D] vs [*y, D] — matching suffixes
    let s3 = Shape::new(vec![
        Dim::Variadic("x".to_string()),
        Dim::Literal(256),
    ]);
    let s4 = Shape::new(vec![
        Dim::Variadic("y".to_string()),
        Dim::Literal(256),
    ]);
    assert!(engine.shapes_compatible(&s3, &s4));

    // Incompatible: [*x, A, B] vs [*y, C, B] where A != C
    // Suffix unification catches this: [A, B] vs [C, B] with A=512, C=128
    let s5 = Shape::new(vec![
        Dim::Variadic("x".to_string()),
        Dim::Literal(512),
        Dim::Literal(64),
    ]);
    let s6 = Shape::new(vec![
        Dim::Variadic("y".to_string()),
        Dim::Literal(128),
        Dim::Literal(64),
    ]);
    assert!(!engine.shapes_compatible(&s5, &s6));

    // Compatible: [*x] vs [*y] — both fully variadic
    let s7 = Shape::new(vec![Dim::Variadic("x".to_string())]);
    let s8 = Shape::new(vec![Dim::Variadic("y".to_string())]);
    assert!(engine.shapes_compatible(&s7, &s8));
}

// ========================================
// Exhaustive Unify Arm Regression Tests
// ========================================

#[test]
fn test_unify_variadic_vs_non_variadic_errors() {
    let mut ctx = InferenceContext::new();
    // Variadic vs Literal should error
    assert!(ctx
        .unify(&Dim::Variadic("x".into()), &Dim::Literal(42))
        .is_err());
    // Variadic vs Named should error
    assert!(ctx
        .unify(&Dim::Variadic("x".into()), &Dim::Named("n".into()))
        .is_err());
    // Named vs Variadic (reversed) should also error
    assert!(ctx
        .unify(&Dim::Named("n".into()), &Dim::Variadic("x".into()))
        .is_err());
}

#[test]
fn test_unify_variadic_variadic_equivalence() {
    let mut ctx = InferenceContext::new();
    assert!(ctx
        .unify(&Dim::Variadic("a".into()), &Dim::Variadic("b".into()))
        .is_ok());
    assert_eq!(ctx.equivalences.get("a"), Some(&"b".to_string()));
}

#[test]
fn test_unify_named_expr_resolved() {
    let mut ctx = InferenceContext::new();
    // Resolve "dim" = 2048, then unify Named("dim") with Expr(x * 4)
    ctx.resolved_dims.insert("dim".into(), 2048);
    let expr = Dim::Expr(Box::new(DimExpr {
        left: Dim::Named("x".into()),
        op: BinOp::Mul,
        right: Dim::Literal(4),
    }));
    assert!(ctx.unify(&Dim::Named("dim".into()), &expr).is_ok());
    assert_eq!(ctx.resolved_dims.get("x"), Some(&512));
}

#[test]
fn test_unify_named_expr_unresolved_records_pending() {
    let mut ctx = InferenceContext::new();
    let expr = Dim::Expr(Box::new(DimExpr {
        left: Dim::Named("x".into()),
        op: BinOp::Add,
        right: Dim::Literal(1),
    }));
    // "dim" not resolved — should record pending constraint, not error
    assert!(ctx.unify(&Dim::Named("dim".into()), &expr).is_ok());
    assert_eq!(ctx.pending_constraints.len(), 1);
}

#[test]
fn test_unify_global_named_resolution() {
    let mut ctx = InferenceContext::new();
    // Global resolved, named not — should propagate
    ctx.resolved_dims.insert("g".into(), 128);
    assert!(ctx
        .unify(&Dim::Global("g".into()), &Dim::Named("n".into()))
        .is_ok());
    assert_eq!(ctx.resolved_dims.get("n"), Some(&128));
}

#[test]
fn test_unify_global_literal_resolution() {
    let mut ctx = InferenceContext::new();
    assert!(ctx
        .unify(&Dim::Global("g".into()), &Dim::Literal(256))
        .is_ok());
    assert_eq!(ctx.resolved_dims.get("g"), Some(&256));
}

#[test]
fn test_unify_global_expr_resolved() {
    let mut ctx = InferenceContext::new();
    ctx.resolved_dims.insert("g".into(), 1024);
    let expr = Dim::Expr(Box::new(DimExpr {
        left: Dim::Named("y".into()),
        op: BinOp::Mul,
        right: Dim::Literal(2),
    }));
    assert!(ctx.unify(&Dim::Global("g".into()), &expr).is_ok());
    assert_eq!(ctx.resolved_dims.get("y"), Some(&512));
}

#[test]
fn test_unify_global_expr_unresolved_records_pending() {
    let mut ctx = InferenceContext::new();
    let expr = Dim::Expr(Box::new(DimExpr {
        left: Dim::Named("y".into()),
        op: BinOp::Add,
        right: Dim::Literal(1),
    }));
    assert!(ctx.unify(&Dim::Global("g".into()), &expr).is_ok());
    assert_eq!(ctx.pending_constraints.len(), 1);
}
