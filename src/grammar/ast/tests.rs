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
fn test_build_residual() {
    let input = include_str!("../../../examples/residual.ns");
    test_build(input, "residual.ns");
}

#[test]
fn test_build_01_comments() {
    let input = include_str!("../../../examples/01-comments.ns");
    test_build(input, "01-comments.ns");
}

#[test]
fn test_build_03_parameters() {
    let input = include_str!("../../../examples/03-parameters.ns");
    test_build(input, "03-parameters.ns");
}

#[test]
fn test_build_07_pipelines() {
    let input = include_str!("../../../examples/07-pipelines.ns");
    test_build(input, "07-pipelines.ns");
}

#[test]
fn test_build_10_match() {
    let input = include_str!("../../../examples/10-match.ns");
    test_build(input, "10-match.ns");
}

#[test]
fn test_build_22_xor() {
    let input = include_str!("../../../examples/22-xor.ns");
    test_build(input, "22-xor.ns");
}

#[test]
fn test_build_28_context() {
    let input = include_str!("../../../examples/28-context_basic.ns");
    test_build(input, "28-context_basic.ns");
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

check_example_build!(build_02_imports, "02-imports.ns");
check_example_build!(build_04_shapes, "04-shapes.ns");
check_example_build!(build_05_ports, "05-ports.ns");
check_example_build!(build_06_impl_refs, "06-impl-refs.ns");
check_example_build!(build_08_tuples, "08-tuples.ns");
check_example_build!(build_09_port_access, "09-port-access.ns");
check_example_build!(build_11_calls, "11-calls.ns");
check_example_build!(build_12_expressions, "12-expressions.ns");
check_example_build!(build_13_values, "13-values.ns");
check_example_build!(build_14_composite, "14-composite.ns");
check_example_build!(build_15_edge_cases, "15-edge-cases.ns");
