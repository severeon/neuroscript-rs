use crate::interfaces::*;
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};

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

    /// Flatten to rank-1 shape [size].
    /// Returns `None` if the shape contains non-literal dimensions (named, variadic, wildcard)
    /// or if the total size overflows `usize`.
    pub fn flatten(&self) -> Option<Shape> {
        let size = self.size()?.to_usize()?;
        Some(Shape::new(vec![Dim::Literal(size as i64)]))
    }
}
