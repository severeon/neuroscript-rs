use super::fixtures::*;
use crate::interfaces::ValidationError;

#[test]
fn test_dunder_name_rejected() {
    // __foo__ has len 7 > 4, starts with __, ends with __ => reserved
    let mut program = ProgramBuilder::new()
        .with_composite("__foo__", vec![], Some(10))
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(e, ValidationError::ReservedName { name } if name == "__foo__")
    });
}

#[test]
fn test_single_underscore_name_passes() {
    // _foo_ does NOT start/end with double underscores => allowed
    let mut program = ProgramBuilder::new()
        .with_composite("_foo_", vec![], Some(10))
        .build();

    assert_validation_ok(&mut program);
}

#[test]
fn test_four_char_dunder_passes() {
    // ____ has len 4 which is NOT > 4 => allowed (boundary case)
    let mut program = ProgramBuilder::new()
        .with_composite("____", vec![], Some(10))
        .build();

    assert_validation_ok(&mut program);
}

#[test]
fn test_five_char_dunder_rejected() {
    // _____ has len 5 > 4, starts with __, ends with __ => reserved
    let mut program = ProgramBuilder::new()
        .with_composite("_____", vec![], Some(10))
        .build();

    assert_validation_error(&mut program, |e| {
        matches!(e, ValidationError::ReservedName { name } if name == "_____")
    });
}

#[test]
fn test_normal_name_passes() {
    let mut program = ProgramBuilder::new()
        .with_composite("MyNeuron", vec![], Some(10))
        .build();

    assert_validation_ok(&mut program);
}
