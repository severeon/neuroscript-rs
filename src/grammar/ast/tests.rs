use super::*;
use crate::grammar::NeuroScriptParser;
use pest::Parser;

fn parse_program(input: &str) -> Result<Program, ParseError> {
    let pairs = NeuroScriptParser::parse(Rule::program, input).map_err(error::from_pest_error)?;

    let mut builder = AstBuilder::new();
    builder.build_program(pairs.into_iter().next().unwrap())
}

#[test]
fn test_simple_neuron() {
    let input = r#"neuron Linear(in_dim, out_dim):
  in: [*, in_dim]
  out: [*, out_dim]
  impl: core,nn/Linear
"#;
    let program = parse_program(input).expect("Failed to parse");
    assert_eq!(program.neurons.len(), 1);
    assert!(program.neurons.contains_key("Linear"));

    let neuron = &program.neurons["Linear"];
    assert_eq!(neuron.params.len(), 2);
    assert_eq!(neuron.inputs.len(), 1);
    assert_eq!(neuron.outputs.len(), 1);
    assert!(matches!(neuron.body, NeuronBody::Primitive(_)));
}

#[test]
fn test_use_stmt() {
    let input = r#"use core,nn/*

neuron Test:
  in: [*]
  out: [*]
  impl: core,nn/Test
"#;
    let program = parse_program(input).expect("Failed to parse");
    assert_eq!(program.uses.len(), 1);
    assert_eq!(program.uses[0].source, "core");
    assert_eq!(program.uses[0].path, vec!["nn", "*"]);
}

#[test]
fn test_composite_neuron() {
    let input = r#"neuron MLP(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out
"#;
    let program = parse_program(input).expect("Failed to parse");
    let neuron = &program.neurons["MLP"];

    if let NeuronBody::Graph { connections, .. } = &neuron.body {
        assert_eq!(connections.len(), 2); // in->Linear, Linear->out
    } else {
        panic!("Expected Graph body");
    }
}

// === Build tests for example files ===

fn test_build(input: &str, name: &str) {
    let build_result = parse_program(input);
    assert!(
        build_result.is_ok(),
        "{}: build failed: {:?}",
        name,
        build_result.err()
    );
}

#[test]
fn test_build_stdlib_residual() {
    let input = include_str!("../../../examples/stdlib/residual.ns");
    test_build(input, "stdlib/residual.ns");
}

#[test]
fn test_build_shape_inference() {
    let input = include_str!("../../../examples/tutorials/01_shape_inference.ns");
    test_build(input, "tutorials/01_shape_inference.ns");
}

#[test]
fn test_build_primitives_basics() {
    let input = include_str!("../../../examples/primitives/basics.ns");
    test_build(input, "primitives/basics.ns");
}

#[test]
fn test_build_stdlib_feedforward() {
    let input = include_str!("../../../examples/stdlib/feedforward.ns");
    test_build(input, "stdlib/feedforward.ns");
}

#[test]
fn test_build_tutorials_fork_join() {
    let input = include_str!("../../../examples/tutorials/02_fork_join.ns");
    test_build(input, "tutorials/02_fork_join.ns");
}

#[test]
fn test_build_match_inline() {
    let input = include_str!("../../../examples/match_inline_test.ns");
    test_build(input, "match_inline_test.ns");
}

#[test]
fn test_build_match_multiline() {
    let input = include_str!("../../../examples/match_multiline.ns");
    test_build(input, "match_multiline.ns");
}

// Run build check on all numbered example files
macro_rules! check_example_build {
    ($name:ident, $file:expr) => {
        #[test]
        fn $name() {
            let input = include_str!(concat!("../../../examples/", $file));
            test_build(input, $file);
        }
    };
}

check_example_build!(build_primitives_activations, "primitives/activations.ns");
check_example_build!(build_primitives_structural, "primitives/structural.ns");
check_example_build!(build_primitives_operations, "primitives/operations.ns");
check_example_build!(build_primitives_attention, "primitives/attention.ns");
check_example_build!(build_stdlib_attention, "stdlib/attention.ns");
check_example_build!(build_stdlib_transformer_block, "stdlib/transformer_block.ns");
check_example_build!(build_tutorials_fork_join, "tutorials/02_fork_join.ns");
check_example_build!(build_real_world_resnet, "real_world/resnet.ns");
check_example_build!(build_dropout, "dropout.ns");
check_example_build!(build_mlp_with_dropout, "mlp_with_dropout.ns");
check_example_build!(build_dilated_conv, "DilatedConv.ns");
