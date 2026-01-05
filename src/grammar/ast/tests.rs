use super::*;
use crate::grammar::NeuroScriptParser;
use crate::interfaces::Parser as OldParser;
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

// === Comparison tests: pest vs handwritten parser ===

fn compare_parsers(input: &str, name: &str) {
    let pest_result = parse_program(input);
    let old_result = OldParser::parse(input);

    match (&pest_result, &old_result) {
        (Ok(pest_prog), Ok(old_prog)) => {
            // Compare neuron counts
            assert_eq!(
                pest_prog.neurons.len(),
                old_prog.neurons.len(),
                "{}: neuron count mismatch",
                name
            );

            // Compare use statement counts
            assert_eq!(
                pest_prog.uses.len(),
                old_prog.uses.len(),
                "{}: use statement count mismatch",
                name
            );

            // Compare each neuron
            for (neuron_name, old_neuron) in &old_prog.neurons {
                let pest_neuron = pest_prog
                    .neurons
                    .get(neuron_name)
                    .unwrap_or_else(|| panic!("{}: missing neuron {}", name, neuron_name));

                // Compare params
                assert_eq!(
                    pest_neuron.params.len(),
                    old_neuron.params.len(),
                    "{}: param count mismatch for {}",
                    name,
                    neuron_name
                );

                // Compare inputs
                assert_eq!(
                    pest_neuron.inputs.len(),
                    old_neuron.inputs.len(),
                    "{}: input count mismatch for {}",
                    name,
                    neuron_name
                );

                // Compare outputs
                assert_eq!(
                    pest_neuron.outputs.len(),
                    old_neuron.outputs.len(),
                    "{}: output count mismatch for {}",
                    name,
                    neuron_name
                );

                // Compare body type
                match (&pest_neuron.body, &old_neuron.body) {
                    (NeuronBody::Primitive(_), NeuronBody::Primitive(_)) => {}
                    (
                        NeuronBody::Graph {
                            connections: pc, ..
                        },
                        NeuronBody::Graph {
                            connections: oc, ..
                        },
                    ) => {
                        assert_eq!(
                            pc.len(),
                            oc.len(),
                            "{}: connection count mismatch for {}",
                            name,
                            neuron_name
                        );
                    }
                    _ => panic!(
                        "{}: body type mismatch for {}: {:?} vs {:?}",
                        name,
                        neuron_name,
                        std::mem::discriminant(&pest_neuron.body),
                        std::mem::discriminant(&old_neuron.body)
                    ),
                }
            }
        }
        (Err(e), Ok(_)) => panic!("{}: pest failed but old succeeded: {:?}", name, e),
        (Ok(_), Err(e)) => panic!("{}: old failed but pest succeeded: {:?}", name, e),
        (Err(_), Err(_)) => {
            // Both failed - that's ok, they might have the same error
        }
    }
}

#[test]
fn test_compare_residual() {
    let input = include_str!("../../../examples/residual.ns");
    compare_parsers(input, "residual.ns");
}

#[test]
fn test_compare_01_comments() {
    let input = include_str!("../../../examples/01-comments.ns");
    compare_parsers(input, "01-comments.ns");
}

#[test]
fn test_compare_03_parameters() {
    let input = include_str!("../../../examples/03-parameters.ns");
    compare_parsers(input, "03-parameters.ns");
}

#[test]
fn test_compare_07_pipelines() {
    let input = include_str!("../../../examples/07-pipelines.ns");
    compare_parsers(input, "07-pipelines.ns");
}

#[test]
fn test_compare_10_match() {
    let input = include_str!("../../../examples/10-match.ns");
    compare_parsers(input, "10-match.ns");
}

#[test]
fn test_compare_22_xor() {
    let input = include_str!("../../../examples/22-xor.ns");
    compare_parsers(input, "22-xor.ns");
}

#[test]
fn test_compare_28_context() {
    let input = include_str!("../../../examples/28-context_basic.ns");
    compare_parsers(input, "28-context_basic.ns");
}

// Run comparison on all numbered example files
macro_rules! compare_example {
    ($name:ident, $file:expr) => {
        #[test]
        fn $name() {
            let input = include_str!(concat!("../../../examples/", $file));
            compare_parsers(input, $file);
        }
    };
}

compare_example!(compare_02_imports, "02-imports.ns");
compare_example!(compare_04_shapes, "04-shapes.ns");
compare_example!(compare_05_ports, "05-ports.ns");
compare_example!(compare_06_impl_refs, "06-impl-refs.ns");
compare_example!(compare_08_tuples, "08-tuples.ns");
compare_example!(compare_09_port_access, "09-port-access.ns");
compare_example!(compare_11_calls, "11-calls.ns");
compare_example!(compare_12_expressions, "12-expressions.ns");
compare_example!(compare_13_values, "13-values.ns");
compare_example!(compare_14_composite, "14-composite.ns");
compare_example!(compare_15_edge_cases, "15-edge-cases.ns");
