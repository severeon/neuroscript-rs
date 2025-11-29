use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};
use crate::interfaces::*;

impl Shape {
    pub fn from_dims(dims: Vec<usize>) -> Self {
        Shape {
            dims: dims.into_iter().map(|d| Dim::Literal(d as i64)).collect(),
        }
    }

    /// rank = number of axes
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// total number of elements as BigUint (product of dims)
    /// Returns None if any dimension is not a literal value
    pub fn size(&self) -> Option<BigUint> {
        let mut p = BigUint::one();
        for d in &self.dims {
            match d {
                Dim::Literal(n) => {
                    if *n < 0 {
                        return None; // Negative dimensions don't make sense for size
                    }
                    p *= *n as u64;
                }
                _ => return None, // Non-literal dimensions can't be computed
            }
        }
        Some(p)
    }

    /// Try to get the size, returning 0 for unknown dimensions
    pub fn size_or_zero(&self) -> BigUint {
        self.size().unwrap_or(BigUint::zero())
    }

    /// axiswise check: A[i] <= B[i] for all i (requires same rank)
    /// Only works for literal dimensions
    pub fn axiswise_le(&self, other: &Shape) -> Option<bool> {
        if self.rank() != other.rank() {
            return None;
        }
        Some(self.dims.iter().zip(&other.dims).all(|(a, b)| {
            match (a.as_literal(), b.as_literal()) {
                (Some(a_val), Some(b_val)) => a_val <= b_val,
                _ => false, // Non-literal dimensions can't be compared
            }
        }))
    }

    /// axiswise divisibility: a[i] divides b[i] for all i (requires same rank)
    /// Only works for literal dimensions
    pub fn axiswise_divides(&self, other: &Shape) -> Option<bool> {
        if self.rank() != other.rank() {
            return None;
        }
        Some(self.dims.iter().zip(&other.dims).all(|(a, b)| {
            match (a.as_literal(), b.as_literal()) {
                (Some(a_val), Some(b_val)) => {
                    if a_val == 0 { false } else { b_val % a_val == 0 }
                }
                _ => false, // Non-literal dimensions can't be divided
            }
        }))
    }

    /// axiswise gcd (returns None if ranks mismatch or non-literal dimensions)
    /// Only works for literal dimensions
    pub fn axiswise_gcd(&self, other: &Shape) -> Option<Shape> {
        if self.rank() != other.rank() {
            return None;
        }
        let dims: Option<Vec<Dim>> = self
            .dims
            .iter()
            .zip(&other.dims)
            .map(|(a, b)| {
                match (a.as_literal(), b.as_literal()) {
                    (Some(a_val), Some(b_val)) => Some(Dim::Literal(num_integer::gcd(a_val, b_val))),
                    _ => None,
                }
            })
            .collect();
        dims.map(Shape::new)
    }

    /// axiswise lcm (None if rank mismatch or non-literal dimensions)
    /// Only works for literal dimensions
    pub fn axiswise_lcm(&self, other: &Shape) -> Option<Shape> {
        if self.rank() != other.rank() {
            return None;
        }
        let dims: Option<Vec<Dim>> = self
            .dims
            .iter()
            .zip(&other.dims)
            .map(|(a, b)| {
                match (a.as_literal(), b.as_literal()) {
                    (Some(a_val), Some(b_val)) => Some(Dim::Literal(num_integer::lcm(a_val, b_val))),
                    _ => None,
                }
            })
            .collect();
        dims.map(Shape::new)
    }

    /// flatten to rank-1 shape [size]
    pub fn flatten(&self) -> Shape {
        let size = self.size().unwrap_or(BigUint::zero()).to_usize().unwrap_or(0);
        Shape::new(vec![Dim::Literal(size as i64)]) // best-effort for small sizes
    }

    /// check if two shapes are permutations of each other (same multiset of dims)
    pub fn permutes(&self, other: &Shape) -> bool {
        if self.rank() != other.rank() {
            return false;
        }
        let mut a = self.dims.clone();
        let mut b = other.dims.clone();
        a.sort_unstable();
        b.sort_unstable();
        a == b
    }

    /// check if the two shapes have same total cardinality
    pub fn same_cardinality(&self, other: &Shape) -> bool {
        self.size() == other.size()
    }

    /// is shape 'aligned' with k? i.e., every axis is multiple of k
    /// Only works for literal dimensions
    pub fn aligned(&self, k: usize) -> bool {
        if k == 0 {
            return false;
        }
        self.dims.iter().all(|d| {
            match d.as_literal() {
                Some(n) => n % (k as i64) == 0,
                _ => false, // Non-literal dimensions can't be checked for alignment
            }
        })
    }

    /// `tiles` returns true if self tiles other (i.e., other is integral multiple per-axis)
    /// requires same rank.
    pub fn tiles(&self, other: &Shape) -> Option<bool> {
        self.axiswise_divides(other)
    }
}

/// Quotient and remainder of total sizes: returns (q, r) such that
/// size_b = q * size_a + r and 0 <= r < size_a.
/// Returns None if size(A) == 0, non-literal dimensions, or size(A) == 0.
pub fn quotient_remainder_total(a: &Shape, b: &Shape) -> Option<(BigUint, BigUint)> {
    let sa = a.size()?;
    if sa.is_zero() {
        return None;
    }
    let sb = b.size()?;
    let (q, r) = sb.div_rem(&sa);
    Some((q, r))
}

/// Returns true if shapes are broadcastable together (numpy-style).
/// Broadcasting works from right-to-left:
/// for each axis (from end), either equal, or one of them is 1; missing axes are treated as 1.
/// Only works for literal dimensions.
pub fn broadcastable(a: &Shape, b: &Shape) -> bool {
    let ar = a.rank();
    let br = b.rank();
    let r = ar.max(br);
    for i in 0..r {
        // index from right
        let ai = if i < ar { &a.dims[ar - 1 - i] } else { &Dim::Literal(1) };
        let bi = if i < br { &b.dims[br - 1 - i] } else { &Dim::Literal(1) };
        if !ai.broadcastable_with(bi) {
            return false;
        }
    }
    true
}

/// reshapeable: total cardinalities are equal
pub fn reshapeable(a: &Shape, b: &Shape) -> bool {
    a.size() == b.size()
}

/// refine an axis: split axis `axis` (0-based) into `factors` if product matches.
/// Example: [64] refine_axis 0 by [8,8] -> [8,8]
/// Returns None if axis is out of range, factors product != axis size, or non-literal dimensions.
pub fn refine_axis(shape: &Shape, axis: usize, factors: &[usize]) -> Option<Shape> {
    if axis >= shape.rank() {
        return None;
    }
    let axis_size = shape.dims[axis].as_literal()?;
    let mut prod = 1usize;
    for &f in factors {
        prod = prod.checked_mul(f)?;
    }
    if prod as i64 != axis_size {
        return None;
    }
    let mut out = Vec::with_capacity(shape.rank() - 1 + factors.len());
    out.extend_from_slice(&shape.dims[..axis]);
    for &f in factors {
        out.push(Dim::Literal(f as i64));
    }
    out.extend_from_slice(&shape.dims[axis + 1..]);
    Some(Shape { dims: out })
}

/// coarsen axes [a..b) (exclusive end). Merge dims in that range.
///
/// Example: coarsen_axes([2,3,4,5], 1..3) => [2, 12, 5]
/// Only works for literal dimensions.
pub fn coarsen_axes(shape: &Shape, range: std::ops::Range<usize>) -> Option<Shape> {
    if range.end > shape.rank() || range.start >= range.end {
        return None;
    }
    let mut merged: usize = 1;
    for d in &shape.dims[range.start..range.end] {
        let val = d.as_literal()? as usize;
        merged = merged.checked_mul(val)?;
    }
    let mut out = Vec::with_capacity(shape.rank() - (range.end - range.start) + 1);
    out.extend_from_slice(&shape.dims[..range.start]);
    out.push(Dim::Literal(merged as i64));
    out.extend_from_slice(&shape.dims[range.end..]);
    Some(Shape { dims: out })
}

/// tile_count: axiswise floor division other / self (requires same rank).
/// Returns None if ranks mismatch, any self axis is 0, or non-literal dimensions.
pub fn tile_count(tile: &Shape, container: &Shape) -> Option<Vec<usize>> {
    if tile.rank() != container.rank() {
        return None;
    }
    let mut out = Vec::with_capacity(tile.rank());
    for (t, c) in tile.dims.iter().zip(&container.dims) {
        let t_val = t.as_literal()? as usize;
        let c_val = c.as_literal()? as usize;
        if t_val == 0 { return None; }
        out.push(c_val / t_val);
    }
    Some(out)
}

/// is n a power of two?
pub fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// integer factorization helper (trial division). Returns prime factors with multiplicity.
/// This is intentionally simple; for very large integers use a specialized crate.
/// Returns empty vec if n <= 1.
pub fn prime_factors(mut n: usize) -> Vec<usize> {
    let mut res = Vec::new();
    if n <= 1 {
        return res;
    }
    while n % 2 == 0 {
        res.push(2);
        n /= 2;
    }
    let mut f = 3usize;
    while f * f <= n {
        while n % f == 0 {
            res.push(f);
            n /= f;
        }
        f += 2;
    }
    if n > 1 {
        res.push(n);
    }
    res
}

/// Factorizations: returns factorizations as lists of factor-vectors whose product equals `n`.
/// WARNING: combinatorial blowup; this generator is simple and returns a few small decompositions
/// rather than all partitions. Meant as a util for "try split axis" not a complete enumerator.
pub fn small_factorizations(n: usize, max_factors: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if n <= 1 {
        out.push(vec![n]);
        return out;
    }
    // trivial: [n]
    out.push(vec![n]);
    // try binary splits into two factors where both >1
    for a in 2..=n {
        if n % a == 0 {
            let b = n / a;
            out.push(vec![a, b]);
            if max_factors >= 3 {
                // if b composite, try split b too
                for c in 2..=b {
                    if b % c == 0 {
                        out.push(vec![a, c, b / c]);
                    }
                }
            }
        }
        if out.len() > 64 {
            break;
        }
    }
    out
}

impl Pattern {
    pub fn new(tokens: Vec<PatToken>) -> Self {
        Pattern { tokens }
    }

    /// convenience
    pub fn from_tokens(tokens: Vec<PatToken>) -> Self {
        Self::new(tokens)
    }

    /// match shape: if capture==true, return the vector of dims bound to `Any` tokens (in order).
    /// Note: `Ignore` tokens don't bind. `Rest` can appear only once and matches remaining axes.
    /// Only works for literal dimensions.
    pub fn matches_and_capture(&self, shape: &Shape, capture: bool) -> Option<Vec<usize>> {
        let mut captures: Vec<usize> = Vec::new();
        // find Rest if any
        let rest_pos = self.tokens.iter().position(|t| *t == PatToken::Rest);
        match rest_pos {
            None => {
                // lengths must match
                if self.tokens.len() != shape.rank() {
                    return None;
                }
                for (tok, dim) in self.tokens.iter().zip(&shape.dims) {
                    match tok {
                        PatToken::Any => {
                            if capture {
                                if let Some(val) = dim.as_literal() {
                                    captures.push(val as usize);
                                } else {
                                    return None; // Can't capture non-literal dimensions
                                }
                            }
                        }
                        PatToken::Ignore => { /* nothing */ }
                        PatToken::Lit(n) => {
                            match dim.as_literal() {
                                Some(val) if *n as i64 == val => {}
                                _ => return None,
                            }
                        }
                        PatToken::Rest => unreachable!(),
                    }
                }
                Some(captures)
            }
            Some(rpos) => {
                // tokens[..rpos] match prefix; tokens[rpos+1..] match suffix against end
                let prefix = &self.tokens[..rpos];
                let suffix = &self.tokens[rpos + 1..];
                if shape.rank() < prefix.len() + suffix.len() {
                    return None;
                }
                // prefix
                for (tok, dim) in prefix.iter().zip(&shape.dims[..prefix.len()]) {
                    match tok {
                        PatToken::Any => {
                            if capture {
                                if let Some(val) = dim.as_literal() {
                                    captures.push(val as usize);
                                } else {
                                    return None; // Can't capture non-literal dimensions
                                }
                            }
                        }
                        PatToken::Ignore => {}
                        PatToken::Lit(n) => {
                            match dim.as_literal() {
                                Some(val) if *n as i64 == val => {}
                                _ => return None,
                            }
                        }
                        PatToken::Rest => unreachable!(),
                    }
                }
                // suffix
                let start_of_suffix = shape.rank() - suffix.len();
                for (tok, dim) in suffix.iter().zip(&shape.dims[start_of_suffix..]) {
                    match tok {
                        PatToken::Any => {
                            if capture {
                                if let Some(val) = dim.as_literal() {
                                    captures.push(val as usize);
                                } else {
                                    return None; // Can't capture non-literal dimensions
                                }
                            }
                        }
                        PatToken::Ignore => {}
                        PatToken::Lit(n) => {
                            match dim.as_literal() {
                                Some(val) if *n as i64 == val => {}
                                _ => return None,
                            }
                        }
                        PatToken::Rest => unreachable!(),
                    }
                }
                Some(captures)
            }
        }
    }

    /// convenience boolean-only match
    pub fn matches(&self, shape: &Shape) -> bool {
        self.matches_and_capture(shape, false).is_some()
    }
}

// ------------------------------
// Comprehensive test suite
// ------------------------------
#[cfg(test)]
mod tests {
    use super::*;

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
}
