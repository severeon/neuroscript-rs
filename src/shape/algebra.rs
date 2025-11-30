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
