use super::fixtures::*;
use crate::interfaces::*;
use num_bigint::BigUint;

#[test]
fn test_shape_mismatch_literal() {
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("Out512", wildcard(), shape_512())
        .with_simple_neuron("In256", shape_256(), wildcard())
        .with_composite(
            "Composite",
            vec![connection(call_endpoint("Out512"), call_endpoint("In256"))],
            Some(10),
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(e, ValidationError::PortMismatch { .. })
    });
}

#[test]
fn test_shape_mismatch_multi_dim() {
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("Out512", wildcard(), shape_batch_512())
        .with_simple_neuron("In256", shape_batch_256(), wildcard())
        .with_composite(
            "Composite",
            vec![connection(call_endpoint("Out512"), call_endpoint("In256"))],
            Some(10),
        )
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(e, ValidationError::PortMismatch { .. })
    });
}

#[test]
fn test_shape_match_exact() {
    let mut program = ProgramBuilder::new()
        .with_simple_neuron("Out512", wildcard(), shape_512())
        .with_simple_neuron("In512", shape_512(), wildcard())
        .with_composite(
            "Composite",
            vec![connection(call_endpoint("Out512"), call_endpoint("In512"))],
            Some(10),
        )
        .build();

    assert_validation_ok(&mut program);
}

#[test]
fn test_biguint_literal_product_overflow() {
    // A shape whose product overflows i64 but is representable with BigUint.
    // i64::MAX ≈ 9.2e18; 100_000 * 100_000 * 100_000 * 100_000 = 1e20 > i64::MAX.
    let large_shape = Shape::new(vec![
        Dim::Literal(100_000),
        Dim::Literal(100_000),
        Dim::Literal(100_000),
        Dim::Literal(100_000),
    ]);

    // BigUint-based Shape::size() should succeed where i64 would overflow.
    let size = large_shape.size();
    assert!(
        size.is_some(),
        "Shape::size() should return Some for large literal shapes"
    );
    let expected = BigUint::from(100_000u64)
        * BigUint::from(100_000u64)
        * BigUint::from(100_000u64)
        * BigUint::from(100_000u64);
    assert_eq!(size.unwrap(), expected);

    // Verify this would indeed overflow i64
    let overflow_check = 100_000i64
        .checked_mul(100_000)
        .and_then(|v| v.checked_mul(100_000))
        .and_then(|v| v.checked_mul(100_000));
    assert!(
        overflow_check.is_none(),
        "i64 should overflow for this product, confirming BigUint is needed"
    );
}
