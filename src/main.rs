//! NeuroScript CLI - Neural architecture composition language compiler

use clap::{Parser, Subcommand};
use miette::{IntoDiagnostic, NamedSource, WrapErr};
use neuroscript::{parse, validate, generate_pytorch, stdlib, NeuronBody};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "neuroscript")]
#[command(about = "Neural architecture composition language compiler", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse a NeuroScript file and show its structure
    Parse {
        /// Input NeuroScript file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Show detailed IR structure
        #[arg(short, long)]
        verbose: bool,
    },

    /// Validate a NeuroScript file
    Validate {
        /// Input NeuroScript file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Show IR structure and validation details
        #[arg(short, long)]
        verbose: bool,

        /// Skip loading standard library
        #[arg(long)]
        no_stdlib: bool,
    },

    /// Compile NeuroScript to PyTorch
    Compile {
        /// Input NeuroScript file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Neuron to compile (defaults to file name in PascalCase)
        #[arg(short = 'n', long, value_name = "NEURON")]
        neuron: Option<String>,

        /// Write output to file instead of stdout
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Disable all optimizations
        #[arg(long)]
        no_optimize: bool,

        /// Disable dead branch elimination only
        #[arg(long)]
        no_dead_elim: bool,

        /// Show optimization details and timing
        #[arg(short, long)]
        verbose: bool,

        /// Skip loading standard library
        #[arg(long)]
        no_stdlib: bool,
    },

    /// List all neurons in a file
    List {
        /// Input NeuroScript file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Show additional details (connections, match expressions)
        #[arg(short, long)]
        verbose: bool,
    },
}

fn main() -> miette::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Parse { file, verbose } => cmd_parse(file, verbose),
        Commands::Validate { file, verbose, no_stdlib } => cmd_validate(file, verbose, no_stdlib),
        Commands::Compile {
            file,
            neuron,
            output,
            no_optimize,
            no_dead_elim,
            verbose,
            no_stdlib,
        } => cmd_compile(file, neuron, output, no_optimize, no_dead_elim, verbose, no_stdlib),
        Commands::List { file, verbose } => cmd_list(file, verbose),
    }
}

/// Parse command: Read and display the IR structure
fn cmd_parse(file: PathBuf, verbose: bool) -> miette::Result<()> {
    let source = read_source(&file)?;
    let program = parse(&source).map_err(|e| {
        let source_named = NamedSource::new(file.to_string_lossy(), source);
        miette::Report::from(e).with_source_code(source_named)
    })?;

    if verbose {
        println!(
            "Parsed {} imports and {} neurons:\n",
            program.uses.len(),
            program.neurons.len()
        );

        for use_stmt in &program.uses {
            println!("  use {},{}", use_stmt.source, use_stmt.path.join("/"));
        }

        if !program.uses.is_empty() {
            println!();
        }
    } else {
        println!("✓ Successfully parsed {}", file.display());
    }

    print_neuron_summary(&program, verbose);

    Ok(())
}

/// Validate command: Parse and validate the program
fn cmd_validate(file: PathBuf, verbose: bool, no_stdlib: bool) -> miette::Result<()> {
    let source = read_source(&file)?;
    let user_program = parse(&source).map_err(|e| {
        let source_named = NamedSource::new(file.to_string_lossy(), source);
        miette::Report::from(e).with_source_code(source_named)
    })?;

    // Load and merge stdlib if not disabled
    let mut program = if no_stdlib {
        if verbose {
            println!("Skipping stdlib loading (--no-stdlib)");
        }
        user_program
    } else {
        if verbose {
            println!("Loading standard library...");
        }
        match stdlib::load_stdlib() {
            Ok(stdlib_program) => {
                if verbose {
                    println!("✓ Loaded {} stdlib neurons", stdlib_program.neurons.len());
                }
                stdlib::merge_programs(stdlib_program, user_program)
            }
            Err(e) => {
                eprintln!("Warning: Failed to load stdlib: {}", e);
                eprintln!("Continuing without stdlib...");
                user_program
            }
        }
    };

    if verbose {
        println!(
            "Parsed {} imports and {} neurons total\n",
            program.uses.len(),
            program.neurons.len()
        );
    }

    if verbose {
        println!("Running validation...");
    }

    match validate(&mut program) {
        Ok(()) => {
            if verbose {
                println!("✓ Program is valid!");
            } else {
                println!("✓ Valid");
            }
            Ok(())
        }
        Err(errors) => {
            println!("✗ Validation failed with {} error(s):", errors.len());
            for error in errors {
                println!("  {}", error);
            }
            std::process::exit(1);
        }
    }
}

/// Compile command: Full pipeline - parse, validate, optimize, codegen
fn cmd_compile(
    file: PathBuf,
    neuron: Option<String>,
    output: Option<PathBuf>,
    no_optimize: bool,
    no_dead_elim: bool,
    verbose: bool,
    no_stdlib: bool,
) -> miette::Result<()> {
    let source = read_source(&file)?;
    let user_program = parse(&source).map_err(|e| {
        let source_named = NamedSource::new(file.to_string_lossy(), source);
        miette::Report::from(e).with_source_code(source_named)
    })?;

    // Load and merge stdlib if not disabled
    let mut program = if no_stdlib {
        if verbose {
            println!("Skipping stdlib loading (--no-stdlib)");
        }
        user_program
    } else {
        if verbose {
            println!("Loading standard library...");
        }
        match stdlib::load_stdlib() {
            Ok(stdlib_program) => {
                if verbose {
                    println!("✓ Loaded {} stdlib neurons", stdlib_program.neurons.len());
                }
                stdlib::merge_programs(stdlib_program, user_program)
            }
            Err(e) => {
                eprintln!("Warning: Failed to load stdlib: {}", e);
                eprintln!("Continuing without stdlib...");
                user_program
            }
        }
    };

    if verbose {
        println!(
            "Parsed {} imports and {} neurons total",
            program.uses.len(),
            program.neurons.len()
        );
    }

    // Validate
    if let Err(errors) = validate(&mut program) {
        println!("✗ Validation failed with {} error(s):", errors.len());
        for error in errors {
            println!("  {}", error);
        }
        std::process::exit(1);
    }
    if verbose {
        println!("✓ Validation passed");
    }

    // Infer neuron name if not provided
    let neuron_name = if let Some(n) = neuron {
        n
    } else {
        infer_neuron_name(&file, &program)?
    };

    // Check neuron exists
    if !program.neurons.contains_key(&neuron_name) {
        let available: Vec<&str> = program.neurons.keys().map(|s| s.as_str()).collect();
        eprintln!("✗ Neuron '{}' not found", neuron_name);
        eprintln!("  Available neurons: {}", available.join(", "));
        std::process::exit(1);
    }

    // Optimize
    if !no_optimize {
        let reordered = neuroscript::optimizer::reorder_match_arms(&mut program);
        let pruned = neuroscript::optimizer::optimize_matches(&mut program, !no_dead_elim);
        if verbose {
            if reordered > 0 {
                println!("  Pattern reordering: optimized {} match expressions", reordered);
            }
            if pruned > 0 {
                println!("  Dead branch elimination: pruned {} arms", pruned);
            }
        }
    } else if verbose {
        println!("  Optimizations disabled");
    }

    // Codegen
    match generate_pytorch(&program, &neuron_name) {
        Ok(python_code) => {
            if let Some(output_path) = output {
                fs::write(&output_path, python_code)
                    .into_diagnostic()
                    .wrap_err_with(|| format!("Failed to write to {}", output_path.display()))?;
                if verbose {
                    println!("✓ Generated PyTorch code for '{}' → {}", neuron_name, output_path.display());
                } else {
                    println!("✓ Compiled to {}", output_path.display());
                }
            } else {
                println!("# Generated PyTorch code for '{}'", neuron_name);
                println!("{}", python_code);
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("✗ Codegen failed: {}", e);
            std::process::exit(1);
        }
    }
}

/// List command: Show all neurons and their signatures
fn cmd_list(file: PathBuf, verbose: bool) -> miette::Result<()> {
    let source = read_source(&file)?;
    let program = parse(&source).map_err(|e| {
        let source_named = NamedSource::new(file.to_string_lossy(), source);
        miette::Report::from(e).with_source_code(source_named)
    })?;

    if program.neurons.is_empty() {
        println!("No neurons found in {}", file.display());
        return Ok(());
    }

    println!(
        "Neurons in {} ({} total):\n",
        file.display(),
        program.neurons.len()
    );

    for (name, neuron) in &program.neurons {
        let kind = match &neuron.body {
            NeuronBody::Primitive(_) => "primitive",
            NeuronBody::Graph { .. } => "composite",
        };

        let inputs: Vec<String> = neuron
            .inputs
            .iter()
            .map(|p| format!("{}: {}", p.name, p.shape))
            .collect();

        let outputs: Vec<String> = neuron
            .outputs
            .iter()
            .map(|p| format!("{}: {}", p.name, p.shape))
            .collect();

        println!("  {} ({})", name, kind);
        println!("    inputs:  {}", inputs.join(", "));
        println!("    outputs: {}", outputs.join(", "));

        if verbose {
            if let NeuronBody::Graph { connections, .. } = &neuron.body {
                println!("    connections: {} ", connections.len());
                for conn in connections.iter().take(3) {
                    println!("      - {:?}", conn);
                }
                if connections.len() > 3 {
                    println!("      ... and {} more", connections.len() - 3);
                }
            }
        }

        println!();
    }

    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Read source file with error handling
fn read_source(file: &PathBuf) -> miette::Result<String> {
    fs::read_to_string(file)
        .into_diagnostic()
        .wrap_err_with(|| format!("Failed to read {}", file.display()))
}

/// Print a summary of all neurons
fn print_neuron_summary(program: &neuroscript::Program, verbose: bool) {
    for (name, neuron) in &program.neurons {
        let kind = match &neuron.body {
            NeuronBody::Primitive(_) => "primitive",
            NeuronBody::Graph { .. } => "composite",
        };

        let inputs: Vec<String> = neuron
            .inputs
            .iter()
            .map(|p| format!("{}: {}", p.name, p.shape))
            .collect();

        let outputs: Vec<String> = neuron
            .outputs
            .iter()
            .map(|p| format!("{}: {}", p.name, p.shape))
            .collect();

        println!("  {} ({})", name, kind);
        println!("    in:  {}", inputs.join(", "));
        println!("    out: {}", outputs.join(", "));

        if verbose {
            if let NeuronBody::Graph { connections: conns, .. } = &neuron.body {
                println!("    connections: {:?}", conns);
            }
        }

        println!();
    }
}

/// Infer neuron name from filename, or fail with helpful message
fn infer_neuron_name(file: &PathBuf, program: &neuroscript::Program) -> miette::Result<String> {
    // Extract filename without extension
    let filename = file
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| {
            miette::miette!("Invalid file path: {}", file.display())
        })?;

    // Convert snake_case or kebab-case to PascalCase
    let neuron_name = filename
        .split(|c: char| c == '_' || c == '-')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect::<String>();

    // Check if this neuron exists
    if program.neurons.contains_key(&neuron_name) {
        Ok(neuron_name)
    } else {
        // Provide helpful message with available neurons
        let available: Vec<&str> = program.neurons.keys().map(|s| s.as_str()).collect();
        eprintln!(
            "✗ No neuron matching filename '{}' found (tried: '{}')",
            file.display(),
            neuron_name
        );
        if !available.is_empty() {
            eprintln!("  Available neurons: {}", available.join(", "));
        }
        eprintln!("  Use --neuron <NAME> to specify explicitly");
        std::process::exit(1);
    }
}
