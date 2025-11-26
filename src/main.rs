//! NeuroScript CLI

use neuroscript::{parse, NeuronBody};
use std::env;
use std::fs;
use miette::{IntoDiagnostic, WrapErr, NamedSource};

fn main() -> miette::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut filename = None;
    let mut validate_flag = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--validate" => {
                validate_flag = true;
                i += 1;
            }
            arg if !arg.starts_with('-') => {
                filename = Some(arg.to_string());
                i += 1;
            }
            _ => {
                eprintln!("Usage: neuroscript [--validate] <file.ns>");
                std::process::exit(1);
            }
        }
    }

    let filename = filename.unwrap_or_else(|| {
        eprintln!("Usage: neuroscript [--validate] <file.ns>");
        std::process::exit(1);
    });

    let source = fs::read_to_string(&filename)
        .into_diagnostic()
        .wrap_err_with(|| format!("Failed to read {}", filename))?;

    match parse(&source) {
        Ok(program) => {
            println!("Parsed {} imports and {} neurons:\n", program.uses.len(), program.neurons.len());

            for use_stmt in &program.uses {
                println!("  use {},{}", use_stmt.source, use_stmt.path.join("/"));
            }

            if !program.uses.is_empty() {
                println!();
            }

            for (name, neuron) in &program.neurons {
                let kind = match &neuron.body {
                    NeuronBody::Primitive(_) => "primitive",
                    NeuronBody::Graph(_) => "composite",
                };

                let inputs: Vec<_> = neuron.inputs.iter()
                    .map(|p| format!("{}: {}", p.name, p.shape))
                    .collect();

                let outputs: Vec<_> = neuron.outputs.iter()
                    .map(|p| format!("{}: {}", p.name, p.shape))
                    .collect();

                println!("  {} ({})", name, kind);
                println!("    in:  {}", inputs.join(", "));
                println!("    out: {}", outputs.join(", "));

                if let NeuronBody::Graph(conns) = &neuron.body {
                    println!("    connections: {:?}", conns);
                }

                println!();
            }

            // Run validation if requested
            if validate_flag {
                println!("Validating program...");
                match neuroscript::validate(&program) {
                    Ok(()) => {
                        println!("✓ Program is valid!");
                    }
                    Err(errors) => {
                        println!("✗ Validation failed with {} errors:", errors.len());
                        for error in errors {
                            println!("  {}", error);
                        }
                        std::process::exit(1);
                    }
                }
            }

            Ok(())
        }
        Err(e) => {
            let source = NamedSource::new(&filename, source);
            Err(miette::Report::from(e).with_source_code(source))
        }
    }
}
