//! NeuroScript Shape System
//!
//! Provides shape algebra operations and shape inference for tensor
//! dimensions in neuron graphs.

pub mod algebra;
pub mod inference;

// Re-export public API
pub use algebra::*;
pub use inference::*;

#[cfg(test)]
mod tests {
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
        Shape::new(names.into_iter().map(|n| Dim::Named(n.to_string())).collect())
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
        assert_eq!(s.dims, vec![Dim::Literal(2), Dim::Literal(3), Dim::Literal(4)]);
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
        let s = Shape::new(vec![Dim::Literal(1000), Dim::Literal(1000), Dim::Literal(1000)]);
        assert_eq!(s.size(), Some(BigUint::from(1_000_000_000u64)));
    }

    #[test]
    fn test_shape_size_very_large() {
        // Test very large product that would overflow usize on 32-bit
        let s = Shape::new(vec![Dim::Literal(65536), Dim::Literal(65536), Dim::Literal(100)]);
        let expected = BigUint::from(65536u64) * BigUint::from(65536u64) * BigUint::from(100u64);
        assert_eq!(s.size(), Some(expected));
    }

    // ========================================
    // 2. Axiswise Operations Tests
    // ========================================

    #[test]
    fn test_axiswise_le_basic() {
        let a = Shape::new(vec![Dim::Literal(2), Dim::Literal(3)]);
        let b = Shape::new(vec![Dim::Literal(4), Dim::Literal(6)]);
        assert_eq!(a.axiswise_le(&b), Some(true));
    }

    #[test]
    fn test_axiswise_le_false() {
        let a = Shape::new(vec![Dim::Literal(5), Dim::Literal(3)]);
        let b = Shape::new(vec![Dim::Literal(4), Dim::Literal(6)]);
        assert_eq!(a.axiswise_le(&b), Some(false));
    }

    #[test]
    fn test_axiswise_le_equal() {
        let a = Shape::new(vec![Dim::Literal(4), Dim::Literal(6)]);
        let b = Shape::new(vec![Dim::Literal(4), Dim::Literal(6)]);
        assert_eq!(a.axiswise_le(&b), Some(true));
    }

    #[test]
    fn test_axiswise_le_rank_mismatch() {
        let a = Shape::new(vec![Dim::Literal(2), Dim::Literal(3)]);
        let b = Shape::new(vec![Dim::Literal(4), Dim::Literal(6), Dim::Literal(8)]);
        assert_eq!(a.axiswise_le(&b), None);
    }

    #[test]
    fn test_axiswise_divides_basic() {
        let a = Shape::new(vec![Dim::Literal(2), Dim::Literal(3)]);
        let b = Shape::new(vec![Dim::Literal(4), Dim::Literal(6)]);
        assert_eq!(a.axiswise_divides(&b), Some(true));
    }

    #[test]
    fn test_axiswise_divides_false() {
        let a = Shape::new(vec![Dim::Literal(3), Dim::Literal(4)]);
        let b = Shape::new(vec![Dim::Literal(7), Dim::Literal(8)]);
        assert_eq!(a.axiswise_divides(&b), Some(false));
    }

    #[test]
    fn test_axiswise_divides_with_zero() {
        let a = Shape::new(vec![Dim::Literal(0), Dim::Literal(3)]);
        let b = Shape::new(vec![Dim::Literal(4), Dim::Literal(6)]);
        assert_eq!(a.axiswise_divides(&b), Some(false)); // 0 doesn't divide anything
    }

    #[test]
    fn test_axiswise_divides_rank_mismatch() {
        let a = Shape::new(vec![Dim::Literal(2)]);
        let b = Shape::new(vec![Dim::Literal(4), Dim::Literal(6)]);
        assert_eq!(a.axiswise_divides(&b), None);
    }

    #[test]
    fn test_axiswise_gcd_basic() {
        let a = Shape::new(vec![Dim::Literal(12), Dim::Literal(18)]);
        let b = Shape::new(vec![Dim::Literal(8), Dim::Literal(24)]);
        assert_eq!(a.axiswise_gcd(&b), Some(Shape::new(vec![Dim::Literal(4), Dim::Literal(6)])));
    }

    #[test]
    fn test_axiswise_gcd_coprime() {
        let a = Shape::new(vec![Dim::Literal(7), Dim::Literal(11)]);
        let b = Shape::new(vec![Dim::Literal(3), Dim::Literal(13)]);
        assert_eq!(a.axiswise_gcd(&b), Some(Shape::new(vec![Dim::Literal(1), Dim::Literal(1)])));
    }

    #[test]
    fn test_axiswise_gcd_rank_mismatch() {
        let a = Shape::new(vec![Dim::Literal(12), Dim::Literal(18)]);
        let b = Shape::new(vec![Dim::Literal(8)]);
        assert_eq!(a.axiswise_gcd(&b), None);
    }

    #[test]
    fn test_axiswise_lcm_basic() {
        let a = Shape::new(vec![Dim::Literal(4), Dim::Literal(6)]);
        let b = Shape::new(vec![Dim::Literal(6), Dim::Literal(9)]);
        assert_eq!(a.axiswise_lcm(&b), Some(Shape::new(vec![Dim::Literal(12), Dim::Literal(18)])));
    }

    #[test]
    fn test_axiswise_lcm_same() {
        let a = Shape::new(vec![Dim::Literal(5), Dim::Literal(7)]);
        let b = Shape::new(vec![Dim::Literal(5), Dim::Literal(7)]);
        assert_eq!(a.axiswise_lcm(&b), Some(Shape::new(vec![Dim::Literal(5), Dim::Literal(7)])));
    }

    #[test]
    fn test_axiswise_lcm_rank_mismatch() {
        let a = Shape::new(vec![Dim::Literal(4), Dim::Literal(6)]);
        let b = Shape::new(vec![Dim::Literal(6)]);
        assert_eq!(a.axiswise_lcm(&b), None);
    }

    // ========================================
    // 3. Shape Property Tests
    // ========================================

    #[test]
    fn test_flatten_multidim() {
        let s = Shape::new(vec![Dim::Literal(2), Dim::Literal(3), Dim::Literal(4)]);
        let flat = s.flatten();
        assert_eq!(flat, Shape::new(vec![Dim::Literal(24)]));
    }

    #[test]
    fn test_flatten_already_flat() {
        let s = Shape::new(vec![Dim::Literal(24)]);
        let flat = s.flatten();
        assert_eq!(flat, Shape::new(vec![Dim::Literal(24)]));
    }

    #[test]
    fn test_flatten_empty() {
        let s = Shape::new(vec![]);
        let flat = s.flatten();
        assert_eq!(flat, Shape::new(vec![Dim::Literal(1)])); // Empty shape has size 1
    }

    #[test]
    fn test_permutes_true() {
        let a = Shape::new(vec![Dim::Literal(2), Dim::Literal(3), Dim::Literal(4)]);
        let b = Shape::new(vec![Dim::Literal(4), Dim::Literal(2), Dim::Literal(3)]);
        assert!(a.permutes(&b));
    }

    #[test]
    fn test_permutes_false() {
        let a = Shape::new(vec![Dim::Literal(2), Dim::Literal(3), Dim::Literal(4)]);
        let b = Shape::new(vec![Dim::Literal(2), Dim::Literal(3), Dim::Literal(5)]);
        assert!(!a.permutes(&b));
    }

    #[test]
    fn test_permutes_different_rank() {
        let a = Shape::new(vec![Dim::Literal(2), Dim::Literal(3)]);
        let b = Shape::new(vec![Dim::Literal(2), Dim::Literal(3), Dim::Literal(1)]);
        assert!(!a.permutes(&b));
    }

    #[test]
    fn test_same_cardinality_true() {
        let a = Shape::new(vec![Dim::Literal(2), Dim::Literal(12)]);
        let b = Shape::new(vec![Dim::Literal(4), Dim::Literal(6)]);
        assert!(a.same_cardinality(&b));
    }

    #[test]
    fn test_same_cardinality_false() {
        let a = Shape::new(vec![Dim::Literal(2), Dim::Literal(12)]);
        let b = Shape::new(vec![Dim::Literal(4), Dim::Literal(7)]);
        assert!(!a.same_cardinality(&b));
    }

    #[test]
    fn test_aligned_true() {
        let s = Shape::new(vec![Dim::Literal(16), Dim::Literal(32), Dim::Literal(64)]);
        assert!(s.aligned(8));
        assert!(s.aligned(16));
    }

    #[test]
    fn test_aligned_false() {
        let s = Shape::new(vec![Dim::Literal(16), Dim::Literal(30), Dim::Literal(64)]);
        assert!(!s.aligned(8));
    }

    #[test]
    fn test_aligned_zero() {
        let s = Shape::new(vec![Dim::Literal(16), Dim::Literal(32)]);
        assert!(!s.aligned(0)); // k=0 always returns false
    }

    #[test]
    fn test_aligned_one() {
        let s = Shape::new(vec![Dim::Literal(16), Dim::Literal(32), Dim::Literal(17)]);
        assert!(s.aligned(1)); // Everything is aligned to 1
    }

    #[test]
    fn test_tiles_true() {
        let tile = Shape::new(vec![Dim::Literal(2), Dim::Literal(3)]);
        let container = Shape::new(vec![Dim::Literal(6), Dim::Literal(9)]);
        assert_eq!(tile.tiles(&container), Some(true));
    }

    #[test]
    fn test_tiles_false() {
        let tile = Shape::new(vec![Dim::Literal(2), Dim::Literal(4)]);
        let container = Shape::new(vec![Dim::Literal(6), Dim::Literal(9)]);
        assert_eq!(tile.tiles(&container), Some(false));
    }

    #[test]
    fn test_tiles_rank_mismatch() {
        let tile = Shape::new(vec![Dim::Literal(2)]);
        let container = Shape::new(vec![Dim::Literal(6), Dim::Literal(9)]);
        assert_eq!(tile.tiles(&container), None);
    }

    // ========================================
    // 4. Arithmetic Operations Tests
    // ========================================

    #[test]
    fn test_quotient_remainder_total_exact() {
        let a = Shape::new(vec![Dim::Literal(64)]);
        let b = Shape::new(vec![Dim::Literal(256)]);
        let (q, r) = quotient_remainder_total(&a, &b).unwrap();
        assert_eq!(q, 4usize.into());
        assert_eq!(r, 0usize.into());
    }

    #[test]
    fn test_quotient_remainder_total_with_remainder() {
        let a = Shape::new(vec![Dim::Literal(10)]);
        let b = Shape::new(vec![Dim::Literal(37)]);
        let (q, r) = quotient_remainder_total(&a, &b).unwrap();
        assert_eq!(q, 3usize.into());
        assert_eq!(r, 7usize.into());
    }

    #[test]
    fn test_quotient_remainder_total_zero_divisor() {
        let a = Shape::new(vec![Dim::Literal(0)]);
        let b = Shape::new(vec![Dim::Literal(100)]);
        assert!(quotient_remainder_total(&a, &b).is_none());
    }

    #[test]
    fn test_quotient_remainder_total_smaller_dividend() {
        let a = Shape::new(vec![Dim::Literal(100)]);
        let b = Shape::new(vec![Dim::Literal(50)]);
        let (q, r) = quotient_remainder_total(&a, &b).unwrap();
        assert_eq!(q, 0usize.into());
        assert_eq!(r, 50usize.into());
    }

    #[test]
    fn test_broadcastable_basic() {
        assert!(broadcastable(&Shape::new(vec![Dim::Literal(4), Dim::Literal(1), Dim::Literal(8)]), &Shape::new(vec![Dim::Literal(4), Dim::Literal(10), Dim::Literal(8)])));
        assert!(!broadcastable(&Shape::new(vec![Dim::Literal(4), Dim::Literal(3), Dim::Literal(1)]), &Shape::new(vec![Dim::Literal(4), Dim::Literal(10), Dim::Literal(8)])));
    }

    #[test]
    fn test_broadcastable_different_ranks() {
        assert!(broadcastable(&Shape::new(vec![Dim::Literal(5), Dim::Literal(6)]), &Shape::new(vec![Dim::Literal(6)])));
        assert!(broadcastable(&Shape::new(vec![Dim::Literal(1)]), &Shape::new(vec![Dim::Literal(8), Dim::Literal(1), Dim::Literal(6), Dim::Literal(1)])));
    }

    #[test]
    fn test_broadcastable_all_ones() {
        assert!(broadcastable(&Shape::new(vec![Dim::Literal(1), Dim::Literal(1), Dim::Literal(1)]), &Shape::new(vec![Dim::Literal(8), Dim::Literal(4), Dim::Literal(6)])));
    }

    #[test]
    fn test_broadcastable_incompatible() {
        assert!(!broadcastable(&Shape::new(vec![Dim::Literal(3), Dim::Literal(4)]), &Shape::new(vec![Dim::Literal(2), Dim::Literal(5)])));
    }

    #[test]
    fn test_reshapeable_same_size() {
        let a = Shape::new(vec![Dim::Literal(2), Dim::Literal(12)]);
        let b = Shape::new(vec![Dim::Literal(4), Dim::Literal(6)]);
        assert!(reshapeable(&a, &b));
    }

    #[test]
    fn test_reshapeable_different_size() {
        let a = Shape::new(vec![Dim::Literal(2), Dim::Literal(12)]);
        let b = Shape::new(vec![Dim::Literal(4), Dim::Literal(7)]);
        assert!(!reshapeable(&a, &b));
    }

    // ========================================
    // 5. Transformation Tests
    // ========================================

    #[test]
    fn test_refine_axis_basic() {
        let s = Shape::new(vec![Dim::Literal(64)]);
        let r = refine_axis(&s, 0, &[8, 8]).unwrap();
        assert_eq!(r, Shape::new(vec![Dim::Literal(8), Dim::Literal(8)]));
    }

    #[test]
    fn test_refine_axis_middle() {
        let s = Shape::new(vec![Dim::Literal(2), Dim::Literal(12), Dim::Literal(4)]);
        let r = refine_axis(&s, 1, &[3, 4]).unwrap();
        assert_eq!(r, Shape::new(vec![Dim::Literal(2), Dim::Literal(3), Dim::Literal(4), Dim::Literal(4)]));
    }

    #[test]
    fn test_refine_axis_three_factors() {
        let s = Shape::new(vec![Dim::Literal(24)]);
        let r = refine_axis(&s, 0, &[2, 3, 4]).unwrap();
        assert_eq!(r, Shape::new(vec![Dim::Literal(2), Dim::Literal(3), Dim::Literal(4)]));
    }

    #[test]
    fn test_refine_axis_invalid_axis() {
        let s = Shape::new(vec![Dim::Literal(2), Dim::Literal(3)]);
        let r = refine_axis(&s, 2, &[2, 3]);
        assert!(r.is_none());
    }

    #[test]
    fn test_refine_axis_product_mismatch() {
        let s = Shape::new(vec![Dim::Literal(12)]);
        let r = refine_axis(&s, 0, &[2, 5]);
        assert!(r.is_none()); // 2*5=10 != 12
    }

    #[test]
    fn test_coarsen_axes_basic() {
        let s = Shape::new(vec![Dim::Literal(8), Dim::Literal(8)]);
        let c = coarsen_axes(&s, 0..2).unwrap();
        assert_eq!(c, Shape::new(vec![Dim::Literal(64)]));
    }

    #[test]
    fn test_coarsen_axes_middle() {
        let s = Shape::new(vec![Dim::Literal(2), Dim::Literal(3), Dim::Literal(4), Dim::Literal(5)]);
        let c = coarsen_axes(&s, 1..3).unwrap();
        assert_eq!(c, Shape::new(vec![Dim::Literal(2), Dim::Literal(12), Dim::Literal(5)]));
    }

    #[test]
    fn test_coarsen_axes_single() {
        let s = Shape::new(vec![Dim::Literal(2), Dim::Literal(3), Dim::Literal(4)]);
        let c = coarsen_axes(&s, 1..2).unwrap();
        assert_eq!(c, Shape::new(vec![Dim::Literal(2), Dim::Literal(3), Dim::Literal(4)]));
    }

    #[test]
    fn test_coarsen_axes_invalid_range() {
        let s = Shape::new(vec![Dim::Literal(2), Dim::Literal(3), Dim::Literal(4)]);
        assert!(coarsen_axes(&s, 2..2).is_none()); // empty range
        assert!(coarsen_axes(&s, 1..5).is_none()); // out of bounds
    }

    #[test]
    fn test_refine_coarsen_roundtrip() {
        let s = Shape::new(vec![Dim::Literal(64)]);
        let r = refine_axis(&s, 0, &[8, 8]).unwrap();
        assert_eq!(r, Shape::new(vec![Dim::Literal(8), Dim::Literal(8)]));
        let c = coarsen_axes(&r, 0..2).unwrap();
        assert_eq!(c, Shape::new(vec![Dim::Literal(64)]));
    }

    // ========================================
    // 6. Utility Function Tests
    // ========================================

    #[test]
    fn test_is_power_of_two_true() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(4));
        assert!(is_power_of_two(8));
        assert!(is_power_of_two(1024));
    }

    #[test]
    fn test_is_power_of_two_false() {
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(6));
        assert!(!is_power_of_two(1000));
    }

    #[test]
    fn test_prime_factors_small() {
        assert_eq!(prime_factors(1), vec![]);
        assert_eq!(prime_factors(2), vec![2]);
        assert_eq!(prime_factors(3), vec![3]);
        assert_eq!(prime_factors(4), vec![2, 2]);
        assert_eq!(prime_factors(6), vec![2, 3]);
    }

    #[test]
    fn test_prime_factors_composite() {
        assert_eq!(prime_factors(12), vec![2, 2, 3]);
        assert_eq!(prime_factors(100), vec![2, 2, 5, 5]);
        assert_eq!(prime_factors(60), vec![2, 2, 3, 5]);
    }

    #[test]
    fn test_prime_factors_prime() {
        assert_eq!(prime_factors(13), vec![13]);
        assert_eq!(prime_factors(97), vec![97]);
    }

    #[test]
    fn test_small_factorizations_basic() {
        let facs = small_factorizations(12, 3);
        assert!(facs.contains(&vec![12])); // trivial
        assert!(facs.contains(&vec![2, 6]));
        assert!(facs.contains(&vec![3, 4]));
    }

    #[test]
    fn test_small_factorizations_prime() {
        let facs = small_factorizations(13, 3);
        // For prime 13, expect [13] and [13, 1] since 13 = 13 * 1
        assert!(facs.contains(&vec![13]));
        assert!(facs.len() >= 1);
    }

    #[test]
    fn test_small_factorizations_one() {
        let facs = small_factorizations(1, 3);
        assert_eq!(facs, vec![vec![1]]);
    }

    #[test]
    fn test_tile_count_basic() {
        let tile = Shape::new(vec![Dim::Literal(2), Dim::Literal(3)]);
        let container = Shape::new(vec![Dim::Literal(6), Dim::Literal(9)]);
        assert_eq!(tile_count(&tile, &container), Some(vec![3, 3]));
    }

    #[test]
    fn test_tile_count_exact() {
        let tile = Shape::new(vec![Dim::Literal(4), Dim::Literal(4)]);
        let container = Shape::new(vec![Dim::Literal(4), Dim::Literal(4)]);
        assert_eq!(tile_count(&tile, &container), Some(vec![1, 1]));
    }

    #[test]
    fn test_tile_count_non_divisible() {
        let tile = Shape::new(vec![Dim::Literal(3), Dim::Literal(4)]);
        let container = Shape::new(vec![Dim::Literal(7), Dim::Literal(10)]);
        assert_eq!(tile_count(&tile, &container), Some(vec![2, 2])); // floor division
    }

    #[test]
    fn test_tile_count_zero_tile() {
        let tile = Shape::new(vec![Dim::Literal(0), Dim::Literal(3)]);
        let container = Shape::new(vec![Dim::Literal(6), Dim::Literal(9)]);
        assert!(tile_count(&tile, &container).is_none());
    }

    #[test]
    fn test_tile_count_rank_mismatch() {
        let tile = Shape::new(vec![Dim::Literal(2)]);
        let container = Shape::new(vec![Dim::Literal(6), Dim::Literal(9)]);
        assert!(tile_count(&tile, &container).is_none());
    }

    // ========================================
    // 7. Pattern Matching Tests
    // ========================================

    #[test]
    fn test_pattern_literal_exact() {
        let pat = Pattern::from_tokens(vec![PatToken::Lit(32), PatToken::Lit(64)]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(64)])));
        assert!(!pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(65)])));
    }

    #[test]
    fn test_pattern_literal_rank_mismatch() {
        let pat = Pattern::from_tokens(vec![PatToken::Lit(32), PatToken::Lit(64)]);
        assert!(!pat.matches(&Shape::new(vec![Dim::Literal(32)])));
        assert!(!pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(64), Dim::Literal(128)])));
    }

    #[test]
    fn test_pattern_any_single() {
        let pat = Pattern::from_tokens(vec![PatToken::Any]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(42)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(1000)])));
    }

    #[test]
    fn test_pattern_any_multiple() {
        let pat = Pattern::from_tokens(vec![PatToken::Any, PatToken::Any, PatToken::Any]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(1), Dim::Literal(2), Dim::Literal(3)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(100), Dim::Literal(200), Dim::Literal(300)])));
    }

    #[test]
    fn test_pattern_any_with_literal() {
        let pat = Pattern::from_tokens(vec![PatToken::Any, PatToken::Lit(1), PatToken::Any]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(1), Dim::Literal(256)])));
        assert!(!pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(2), Dim::Literal(256)])));
    }

    #[test]
    fn test_pattern_ignore() {
        let pat = Pattern::from_tokens(vec![PatToken::Ignore, PatToken::Lit(64)]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(123), Dim::Literal(64)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(999), Dim::Literal(64)])));
        assert!(!pat.matches(&Shape::new(vec![Dim::Literal(123), Dim::Literal(65)])));
    }

    #[test]
    fn test_pattern_capture_any() {
        let pat = Pattern::from_tokens(vec![PatToken::Any, PatToken::Lit(64), PatToken::Any]);
        let shape = Shape::new(vec![Dim::Literal(32), Dim::Literal(64), Dim::Literal(128)]);
        let captured = pat.matches_and_capture(&shape, true).unwrap();
        assert_eq!(captured, vec![32, 128]);
    }

    #[test]
    fn test_pattern_capture_ignore_not_captured() {
        let pat = Pattern::from_tokens(vec![PatToken::Ignore, PatToken::Any, PatToken::Lit(64)]);
        let shape = Shape::new(vec![Dim::Literal(32), Dim::Literal(128), Dim::Literal(64)]);
        let captured = pat.matches_and_capture(&shape, true).unwrap();
        assert_eq!(captured, vec![128]); // Only Any captured, not Ignore
    }

    #[test]
    fn test_pattern_rest_at_end() {
        let pat = Pattern::from_tokens(vec![PatToken::Lit(32), PatToken::Rest]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(64)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(64), Dim::Literal(128), Dim::Literal(256)])));
        assert!(!pat.matches(&Shape::new(vec![Dim::Literal(64)])));
    }

    #[test]
    fn test_pattern_rest_at_start() {
        let pat = Pattern::from_tokens(vec![PatToken::Rest, PatToken::Lit(256)]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(256)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(256)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(64), Dim::Literal(128), Dim::Literal(256)])));
        assert!(!pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(64)])));
    }

    #[test]
    fn test_pattern_rest_middle() {
        let pat = Pattern::from_tokens(vec![PatToken::Lit(32), PatToken::Rest, PatToken::Lit(256)]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(256)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(64), Dim::Literal(256)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(64), Dim::Literal(128), Dim::Literal(256)])));
        assert!(!pat.matches(&Shape::new(vec![Dim::Literal(32)])));
        assert!(!pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(64), Dim::Literal(128)])));
    }

    #[test]
    fn test_pattern_rest_capture_prefix_suffix() {
        let pat = Pattern::from_tokens(vec![PatToken::Any, PatToken::Rest, PatToken::Any]);
        let shape = Shape::new(vec![Dim::Literal(32), Dim::Literal(64), Dim::Literal(128), Dim::Literal(256)]);
        let captured = pat.matches_and_capture(&shape, true).unwrap();
        assert_eq!(captured, vec![32, 256]); // First and last
    }

    // ========================================
    // 8. Neuron/Tensor Pattern Matching Tests
    // ========================================

    #[test]
    fn test_tensor_batch_pattern() {
        // Pattern: [batch, ...] - match any tensor with batch dimension
        let pat = Pattern::from_tokens(vec![PatToken::Any, PatToken::Rest]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(784)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(3), Dim::Literal(224), Dim::Literal(224)])));

        let captured = pat.matches_and_capture(&Shape::new(vec![Dim::Literal(16), Dim::Literal(3), Dim::Literal(224), Dim::Literal(224)]), true).unwrap();
        assert_eq!(captured, vec![16]); // Batch size captured
    }

    #[test]
    fn test_tensor_channel_last_pattern() {
        // Pattern: [..., C] - match tensors with channel dimension at end
        let pat = Pattern::from_tokens(vec![PatToken::Rest, PatToken::Any]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(3)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(224), Dim::Literal(224), Dim::Literal(3)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(224), Dim::Literal(224), Dim::Literal(3)])));

        let captured = pat.matches_and_capture(&Shape::new(vec![Dim::Literal(32), Dim::Literal(224), Dim::Literal(224), Dim::Literal(3)]), true).unwrap();
        assert_eq!(captured, vec![3]); // Channels captured
    }

    #[test]
    fn test_tensor_spatial_pattern() {
        // Pattern: [*, H, W, *] - match 4D tensors with specific spatial structure
        let pat = Pattern::from_tokens(vec![
            PatToken::Any,
            PatToken::Any,
            PatToken::Any,
            PatToken::Any
        ]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(224), Dim::Literal(224), Dim::Literal(3)])));

        let captured = pat.matches_and_capture(&Shape::new(vec![Dim::Literal(16), Dim::Literal(128), Dim::Literal(128), Dim::Literal(64)]), true).unwrap();
        assert_eq!(captured, vec![16, 128, 128, 64]); // B, H, W, C
    }

    #[test]
    fn test_tensor_fixed_spatial_pattern() {
        // Pattern: [*, 224, 224, *] - match specific spatial size
        let pat = Pattern::from_tokens(vec![
            PatToken::Any,
            PatToken::Lit(224),
            PatToken::Lit(224),
            PatToken::Any
        ]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(224), Dim::Literal(224), Dim::Literal(3)])));
        assert!(!pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(128), Dim::Literal(128), Dim::Literal(3)])));

        let captured = pat.matches_and_capture(&Shape::new(vec![Dim::Literal(16), Dim::Literal(224), Dim::Literal(224), Dim::Literal(64)]), true).unwrap();
        assert_eq!(captured, vec![16, 64]); // Batch and channels
    }

    #[test]
    fn test_attention_head_pattern() {
        // Pattern: [B, H, *, D] - multi-head attention structure
        let pat = Pattern::from_tokens(vec![
            PatToken::Any,  // Batch
            PatToken::Any,  // Heads
            PatToken::Any,  // Sequence length
            PatToken::Any   // Head dimension
        ]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(8), Dim::Literal(512), Dim::Literal(64)])));

        let captured = pat.matches_and_capture(&Shape::new(vec![Dim::Literal(16), Dim::Literal(12), Dim::Literal(256), Dim::Literal(64)]), true).unwrap();
        assert_eq!(captured, vec![16, 12, 256, 64]); // B, H, Seq, D
    }

    #[test]
    fn test_attention_fixed_head_dim_pattern() {
        // Pattern: [*, *, *, 64] - attention with fixed head dimension
        let pat = Pattern::from_tokens(vec![
            PatToken::Any,
            PatToken::Any,
            PatToken::Any,
            PatToken::Lit(64)
        ]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(8), Dim::Literal(512), Dim::Literal(64)])));
        assert!(!pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(8), Dim::Literal(512), Dim::Literal(128)])));
    }

    #[test]
    fn test_conv_kernel_pattern() {
        // Pattern: [*, *, 3, 3] - 3x3 conv kernel
        let pat = Pattern::from_tokens(vec![
            PatToken::Any,
            PatToken::Any,
            PatToken::Lit(3),
            PatToken::Lit(3)
        ]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(64), Dim::Literal(128), Dim::Literal(3), Dim::Literal(3)])));
        assert!(!pat.matches(&Shape::new(vec![Dim::Literal(64), Dim::Literal(128), Dim::Literal(5), Dim::Literal(5)])));
    }

    #[test]
    fn test_broadcastable_middle_axis() {
        // Pattern: [*, 1, *] - broadcastable middle dimension
        let pat = Pattern::from_tokens(vec![PatToken::Any, PatToken::Lit(1), PatToken::Any]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(1), Dim::Literal(256)])));
        assert!(!pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(2), Dim::Literal(256)])));
    }

    #[test]
    fn test_sequence_to_sequence_pattern() {
        // Pattern: [Batch, SeqLen, Hidden] - transformer-style
        let pat = Pattern::from_tokens(vec![PatToken::Any, PatToken::Any, PatToken::Any]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(512), Dim::Literal(768)])));

        let captured = pat.matches_and_capture(&Shape::new(vec![Dim::Literal(16), Dim::Literal(256), Dim::Literal(1024)]), true).unwrap();
        assert_eq!(captured, vec![16, 256, 1024]);
    }

    #[test]
    fn test_embedding_pattern() {
        // Pattern: [VocabSize, EmbedDim] - embedding matrix
        let pat = Pattern::from_tokens(vec![PatToken::Any, PatToken::Any]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(50000), Dim::Literal(768)])));

        let captured = pat.matches_and_capture(&Shape::new(vec![Dim::Literal(30000), Dim::Literal(512)]), true).unwrap();
        assert_eq!(captured, vec![30000, 512]);
    }

    #[test]
    fn test_flatten_pattern() {
        // Pattern: [...] - match any rank
        let pat = Pattern::from_tokens(vec![PatToken::Rest]);
        assert!(pat.matches(&Shape::new(vec![])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(42)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(64), Dim::Literal(128)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(1), Dim::Literal(2), Dim::Literal(3), Dim::Literal(4), Dim::Literal(5)])));
    }

    #[test]
    fn test_specific_batch_size_pattern() {
        // Pattern: [32, ...] - require batch size of 32
        let pat = Pattern::from_tokens(vec![PatToken::Lit(32), PatToken::Rest]);
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(784)])));
        assert!(pat.matches(&Shape::new(vec![Dim::Literal(32), Dim::Literal(3), Dim::Literal(224), Dim::Literal(224)])));
        assert!(!pat.matches(&Shape::new(vec![Dim::Literal(16), Dim::Literal(3), Dim::Literal(224), Dim::Literal(224)])));
    }

    #[test]
    fn test_resnet_residual_pattern() {
        // Check if two shapes can be added (must be identical or broadcastable)
        let a = Shape::new(vec![Dim::Literal(32), Dim::Literal(64), Dim::Literal(56), Dim::Literal(56)]);
        let b = Shape::new(vec![Dim::Literal(32), Dim::Literal(64), Dim::Literal(56), Dim::Literal(56)]);
        assert!(broadcastable(&a, &b));

        // With 1x1 broadcast
        let a = Shape::new(vec![Dim::Literal(32), Dim::Literal(64), Dim::Literal(56), Dim::Literal(56)]);
        let b = Shape::new(vec![Dim::Literal(1), Dim::Literal(64), Dim::Literal(1), Dim::Literal(1)]);
        assert!(broadcastable(&a, &b));
    }

    #[test]
    fn test_tensor_reshape_compatibility() {
        // Check if [32, 784] can be reshaped to [32, 28, 28]
        let a = Shape::new(vec![Dim::Literal(32), Dim::Literal(784)]);
        let b = Shape::new(vec![Dim::Literal(32), Dim::Literal(28), Dim::Literal(28)]);
        assert!(reshapeable(&a, &b));

        // [32, 3, 224, 224] can be flattened to [32, 150528]
        let a = Shape::new(vec![Dim::Literal(32), Dim::Literal(3), Dim::Literal(224), Dim::Literal(224)]);
        let b = Shape::new(vec![Dim::Literal(32), Dim::Literal(150528)]);
        assert!(reshapeable(&a, &b));
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
    fn test_has_wildcard() {
        let engine = ShapeInferenceEngine::new();

        let s1 = wildcard();
        assert!(engine.has_wildcard(&s1));

        let s2 = literal_shape(vec![512]);
        assert!(!engine.has_wildcard(&s2));

        let s3 = Shape::new(vec![Dim::Literal(512), Dim::Wildcard]);
        assert!(engine.has_wildcard(&s3));
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
        let engine = ShapeInferenceEngine::new();

        // Literal is always resolvable
        assert!(engine.is_dim_resolvable(&Dim::Literal(512), &ctx));

        // Wildcard is always resolvable
        assert!(engine.is_dim_resolvable(&Dim::Wildcard, &ctx));

        // Named dimension not yet resolved
        assert!(!engine.is_dim_resolvable(&Dim::Named("dim".to_string()), &ctx));

        // Resolve it
        ctx.resolved_dims.insert("dim".to_string(), 512);
        assert!(engine.is_dim_resolvable(&Dim::Named("dim".to_string()), &ctx));

        // Expression with resolvable operands
        let expr = Dim::Expr(Box::new(DimExpr {
            left: Dim::Named("dim".to_string()),
            op: BinOp::Mul,
            right: Dim::Literal(4),
        }));
        assert!(engine.is_dim_resolvable(&expr, &ctx));

        // Expression with unresolvable operand
        let expr2 = Dim::Expr(Box::new(DimExpr {
            left: Dim::Named("unknown".to_string()),
            op: BinOp::Mul,
            right: Dim::Literal(4),
        }));
        assert!(!engine.is_dim_resolvable(&expr2, &ctx));
    }

    #[test]
    fn test_match_pattern_unification() {
        let mut ctx = InferenceContext::new();
        let engine = ShapeInferenceEngine::new();

        // Pattern [*, d] should match concrete shape [32, 512]
        let pattern = Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]);
        let concrete = literal_shape(vec![32, 512]);

        assert!(engine.unify_pattern_with_shape(&pattern, &concrete, &mut ctx).is_ok());
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
            Dim::Named("d".to_string())
        ]);
        let concrete = literal_shape(vec![32, 64, 128, 512]);

        assert!(engine.unify_pattern_with_shape(&pattern, &concrete, &mut ctx).is_ok());
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
}
