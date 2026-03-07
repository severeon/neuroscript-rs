//! NeuroScript CLI - Neural architecture composition language compiler

use clap::{Parser, Subcommand};
use miette::{IntoDiagnostic, NamedSource, WrapErr};
use neuroscript::package::{self, DependencyContext};
use neuroscript::{parse, stdlib, validate, NeuronBody, StdlibRegistry};
use std::fs;
use std::path::{Path, PathBuf};

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
    /// Initialize a new NeuroScript package
    Init {
        /// Package name
        #[arg(value_name = "NAME")]
        name: String,

        /// Create in this directory (defaults to NAME)
        #[arg(long, value_name = "PATH")]
        path: Option<PathBuf>,

        /// Package version
        #[arg(long, default_value = "0.1.0")]
        version: String,

        /// Author name and email
        #[arg(long)]
        author: Option<String>,

        /// License
        #[arg(long, default_value = "MIT")]
        license: String,

        /// Create a binary package (with examples)
        #[arg(long)]
        bin: bool,
    },

    /// Add a dependency to Axon.toml
    Add {
        /// Package name
        #[arg(value_name = "PACKAGE")]
        package: String,

        /// Version requirement (e.g., "1.0", "^1.2.3")
        #[arg(long)]
        version: Option<String>,

        /// Git repository URL
        #[arg(long)]
        git: Option<String>,

        /// Git branch
        #[arg(long)]
        branch: Option<String>,

        /// Git tag
        #[arg(long)]
        tag: Option<String>,

        /// Git revision (commit hash)
        #[arg(long)]
        rev: Option<String>,

        /// Local filesystem path
        #[arg(long)]
        path: Option<PathBuf>,
    },

    /// Fetch all dependencies
    Fetch {
        /// Show verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Update dependencies to latest compatible versions
        #[arg(long)]
        update: bool,
    },

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

        /// Skip loading fetched dependencies
        #[arg(long)]
        no_deps: bool,
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

        /// Skip loading fetched dependencies
        #[arg(long)]
        no_deps: bool,

        /// Bundle primitive definitions inline (no neuroscript_runtime dependency)
        #[arg(long)]
        bundle: bool,
    },

    /// List all neurons in a file, stdlib, or fetched packages
    List {
        /// Input NeuroScript file
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,

        /// Show additional details (connections, match expressions)
        #[arg(short, long)]
        verbose: bool,

        /// List all stdlib neurons (primitives + composites)
        #[arg(long)]
        stdlib: bool,

        /// List neurons from a specific fetched dependency
        #[arg(long, value_name = "NAME")]
        package: Option<String>,

        /// List all available neurons (stdlib + all fetched deps)
        #[arg(long)]
        available: bool,
    },

    /// Generate an Ed25519 keypair for package signing
    Keygen {
        /// Key name (used for file naming: ~/.neuroscript/keys/{name}.key)
        #[arg(value_name = "NAME")]
        name: String,
    },

    /// Sign and prepare a package for distribution
    Publish {
        /// Package directory (defaults to current directory)
        #[arg(value_name = "PATH")]
        path: Option<PathBuf>,

        /// Path to signing key file
        #[arg(long, value_name = "FILE")]
        key: Option<PathBuf>,

        /// Skip signing (only compute checksums)
        #[arg(long)]
        no_sign: bool,

        /// Show detailed output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Verify package checksums and signature
    Verify {
        /// Package directory (defaults to current directory)
        #[arg(value_name = "PATH")]
        path: Option<PathBuf>,

        /// Show detailed output
        #[arg(short, long)]
        verbose: bool,
    },
}

fn main() -> miette::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init {
            name,
            path,
            version,
            author,
            license,
            bin,
        } => cmd_init(name, path, version, author, license, bin),
        Commands::Add {
            package,
            version,
            git,
            branch,
            tag,
            rev,
            path,
        } => cmd_add(package, version, git, branch, tag, rev, path),
        Commands::Fetch { verbose, update } => cmd_fetch(verbose, update),
        Commands::Parse { file, verbose } => cmd_parse(file, verbose),
        Commands::Validate {
            file,
            verbose,
            no_stdlib,
            no_deps,
        } => cmd_validate(file, verbose, no_stdlib, no_deps),
        Commands::Compile {
            file,
            neuron,
            output,
            no_optimize,
            no_dead_elim,
            verbose,
            no_stdlib,
            no_deps,
            bundle,
        } => cmd_compile(
            file,
            neuron,
            output,
            no_optimize,
            no_dead_elim,
            verbose,
            no_stdlib,
            no_deps,
            bundle,
        ),
        Commands::List {
            file,
            verbose,
            stdlib: list_stdlib,
            package: list_package,
            available,
        } => cmd_list(file, verbose, list_stdlib, list_package, available),
        Commands::Keygen { name } => cmd_keygen(name),
        Commands::Publish {
            path,
            key,
            no_sign,
            verbose,
        } => cmd_publish(path, key, no_sign, verbose),
        Commands::Verify { path, verbose } => cmd_verify(path, verbose),
    }
}

/// Init command: Create a new NeuroScript package
fn cmd_init(
    name: String,
    path: Option<PathBuf>,
    version: String,
    author: Option<String>,
    license: String,
    bin: bool,
) -> miette::Result<()> {
    use neuroscript::package::{init_package, InitOptions};

    let options = InitOptions {
        name: name.clone(),
        path,
        version,
        author,
        license: Some(license),
        bin,
    };

    match init_package(&options) {
        Ok(package_dir) => {
            println!("✓ Created new package '{}' at {}", name, package_dir.display());
            println!("\nNext steps:");
            println!("  cd {}", package_dir.display());
            println!("  # Edit src/*.ns with your neuron definitions");
            println!("  neuroscript build");
            Ok(())
        }
        Err(e) => {
            return Err(miette::miette!("Failed to initialize package: {}", e));
        }
    }
}

/// Add command: Add a dependency to Axon.toml
fn cmd_add(
    package: String,
    version: Option<String>,
    git: Option<String>,
    branch: Option<String>,
    tag: Option<String>,
    rev: Option<String>,
    path: Option<PathBuf>,
) -> miette::Result<()> {
    use neuroscript::package::{Dependency, DependencyDetail, Manifest};

    // Find Axon.toml in current directory or parents
    let manifest_path = Manifest::find_in_directory(".")
        .map_err(|e| miette::miette!("No Axon.toml found in current directory or parents: {}", e))?;

    // Load manifest
    let mut manifest = Manifest::from_path(&manifest_path)
        .into_diagnostic()
        .wrap_err("Failed to load Axon.toml")?;

    // Create dependency specification
    let dep = if let Some(git_url) = git {
        Dependency::Detailed(DependencyDetail {
            version: version.clone(),
            git: Some(git_url),
            branch,
            tag,
            rev,
            path: None,
            optional: false,
        })
    } else if let Some(local_path) = path {
        Dependency::Detailed(DependencyDetail {
            version: None,
            git: None,
            branch: None,
            tag: None,
            rev: None,
            path: Some(local_path),
            optional: false,
        })
    } else if let Some(ver) = version {
        Dependency::Simple(ver)
    } else {
        // Default to latest version
        Dependency::Simple("*".to_string())
    };

    // Add to manifest
    manifest.dependencies.insert(package.clone(), dep);

    // Save manifest
    let toml_string = toml::to_string_pretty(&manifest)
        .into_diagnostic()
        .wrap_err("Failed to serialize manifest")?;

    fs::write(&manifest_path, toml_string)
        .into_diagnostic()
        .wrap_err("Failed to write Axon.toml")?;

    println!("✓ Added dependency: {}", package);
    println!("  Updated: {}", manifest_path.display());
    println!("\nRun `neuroscript fetch` to download dependencies");

    Ok(())
}

/// Fetch command: Fetch all dependencies
fn cmd_fetch(verbose: bool, update: bool) -> miette::Result<()> {
    use neuroscript::package::{Lockfile, Manifest, Registry};

    // Find Axon.toml
    let manifest_path = Manifest::find_in_directory(".")
        .map_err(|e| miette::miette!("No Axon.toml found in current directory or parents: {}", e))?;

    let manifest_dir = manifest_path
        .parent()
        .ok_or_else(|| miette::miette!("Invalid manifest path"))?;

    // Load manifest
    let manifest = Manifest::from_path(&manifest_path)
        .into_diagnostic()
        .wrap_err("Failed to load Axon.toml")?;

    if verbose {
        println!("Loading manifest from {}", manifest_path.display());
        println!("Package: {} v{}", manifest.package.name, manifest.package.version);
    }

    // Check if we have dependencies
    if manifest.dependencies.is_empty() {
        println!("No dependencies to fetch");
        return Ok(());
    }

    // Initialize registry
    let registry = Registry::new()
        .into_diagnostic()
        .wrap_err("Failed to initialize registry")?;

    registry
        .init()
        .into_diagnostic()
        .wrap_err("Failed to initialize cache")?;

    if verbose {
        let cache_dir = Registry::default_cache_dir()
            .into_diagnostic()
            .wrap_err("Failed to get cache directory")?;
        println!("Cache directory: {}", cache_dir.display());
    }

    // Check lockfile
    let lockfile_path = manifest_dir.join("Axon.lock");
    let needs_update = if lockfile_path.exists() && !update {
        let lockfile = Lockfile::from_path(&lockfile_path)
            .into_diagnostic()
            .wrap_err("Failed to load Axon.lock")?;

        if lockfile.is_up_to_date(&manifest) {
            if verbose {
                println!("Lockfile is up-to-date, using existing resolutions");
            }
            false
        } else {
            if verbose {
                println!("Lockfile is outdated, resolving dependencies");
            }
            true
        }
    } else {
        true
    };

    // Fetch dependencies
    println!("Fetching {} dependencies...", manifest.dependencies.len());

    let fetched = registry
        .fetch_dependencies(&manifest)
        .into_diagnostic()
        .wrap_err("Failed to fetch dependencies")?;

    for (name, path) in &fetched {
        println!("  ✓ {} -> {}", name, path.display());
    }

    // Generate/update lockfile if needed
    if needs_update {
        // For now, create a simple lockfile with fetched dependencies
        // Full resolution will be implemented when we have a registry
        let mut lockfile = Lockfile::new();

        for (name, path) in &fetched {
            // Load the dependency's manifest to get version
            let dep_manifest_path = path.join("Axon.toml");
            if let Ok(dep_manifest) = Manifest::from_path(dep_manifest_path) {
                let locked = neuroscript::package::LockedPackage::from_path(
                    name.clone(),
                    dep_manifest.package.version.clone(),
                    path.clone(),
                );
                lockfile.add_package(locked);
            }
        }

        lockfile
            .save(&lockfile_path)
            .into_diagnostic()
            .wrap_err("Failed to save Axon.lock")?;

        if verbose {
            println!("\n✓ Generated Axon.lock");
        }
    }

    println!("\n✓ All dependencies fetched successfully");

    Ok(())
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
/// Parse a source file, load deps/stdlib, and merge into a single program.
/// Shared by cmd_validate and cmd_compile to avoid duplication.
fn load_and_prepare_program(
    file: &Path,
    no_stdlib: bool,
    no_deps: bool,
    verbose: bool,
) -> miette::Result<neuroscript::Program> {
    let source = read_source(file)?;
    let user_program = parse(&source).map_err(|e| {
        let source_named = NamedSource::new(file.to_string_lossy(), source);
        miette::Report::from(e).with_source_code(source_named)
    })?;

    // Try to load dependencies
    let dep_ctx = if no_deps {
        if verbose {
            println!("Skipping dependency loading (--no-deps)");
        }
        None
    } else {
        try_load_dependencies(file, verbose)
    };

    // Load stdlib if not disabled
    let stdlib_program = if no_stdlib {
        if verbose {
            println!("Skipping stdlib loading (--no-stdlib)");
        }
        None
    } else {
        if verbose {
            println!("Loading standard library...");
        }
        match stdlib::load_stdlib() {
            Ok(stdlib_prog) => {
                if verbose {
                    println!("✓ Loaded {} stdlib neurons", stdlib_prog.neurons.len());
                }
                Some(stdlib_prog)
            }
            Err(e) => {
                eprintln!("Warning: Failed to load stdlib: {}", e);
                eprintln!("Continuing without stdlib...");
                None
            }
        }
    };

    // Validate use statements against loaded deps
    if let Some(ref ctx) = dep_ctx {
        let use_errors = package::validate_use_stmts(&user_program, ctx);
        for err in &use_errors {
            eprintln!("Warning: {}", err);
        }
    }

    // Merge: deps -> stdlib -> user
    let empty_stdlib = neuroscript::Program::new();
    let stdlib_prog = stdlib_program.unwrap_or(empty_stdlib);
    let program = package::merge_with_deps(dep_ctx.as_ref(), stdlib_prog, user_program)
        .map_err(|e| miette::miette!("Dependency merge error: {}", e))?;

    if verbose {
        println!(
            "Parsed {} imports and {} neurons total",
            program.uses.len(),
            program.neurons.len()
        );
    }

    Ok(program)
}

fn cmd_validate(file: PathBuf, verbose: bool, no_stdlib: bool, no_deps: bool) -> miette::Result<()> {
    let mut program = load_and_prepare_program(&file, no_stdlib, no_deps, verbose)?;

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
            render_validation_errors(&file, &errors)
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
    no_deps: bool,
    bundle: bool,
) -> miette::Result<()> {
    let mut program = load_and_prepare_program(&file, no_stdlib, no_deps, verbose)?;

    // Validate
    if let Err(errors) = validate(&mut program) {
        return render_validation_errors(&file, &errors);
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
        return Err(miette::miette!(
            "Neuron '{}' not found\n  Available neurons: {}",
            neuron_name,
            available.join(", ")
        ));
    }

    // Optimize
    if !no_optimize {
        let reordered = neuroscript::optimizer::reorder_match_arms(&mut program);
        let pruned = neuroscript::optimizer::optimize_matches(&mut program, !no_dead_elim);
        if verbose {
            if reordered > 0 {
                println!(
                    "  Pattern reordering: optimized {} match expressions",
                    reordered
                );
            }
            if pruned > 0 {
                println!("  Dead branch elimination: pruned {} arms", pruned);
            }
        }
    } else if verbose {
        println!("  Optimizations disabled");
    }

    // Codegen
    let options = neuroscript::CodegenOptions { bundle };
    match neuroscript::generate_pytorch_with_options(&program, &neuron_name, &options) {
        Ok(python_code) => {
            if let Some(output_path) = output {
                fs::write(&output_path, python_code)
                    .into_diagnostic()
                    .wrap_err_with(|| format!("Failed to write to {}", output_path.display()))?;
                if verbose {
                    println!(
                        "✓ Generated PyTorch code for '{}' → {}",
                        neuron_name,
                        output_path.display()
                    );
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
            return Err(miette::miette!("Codegen failed: {}", e));
        }
    }
}

/// List command: Show all neurons and their signatures
fn cmd_list(
    file: Option<PathBuf>,
    verbose: bool,
    list_stdlib: bool,
    list_package: Option<String>,
    available: bool,
) -> miette::Result<()> {
    // Handle --stdlib flag
    if list_stdlib || available {
        list_stdlib_neurons(verbose)?;
    }

    // Handle --package flag
    if let Some(ref pkg_name) = list_package {
        list_package_neurons(pkg_name, verbose)?;
    }

    // Handle --available flag: also list all dep neurons
    if available {
        if let Some(dep_ctx) = try_load_dependencies_from_cwd(verbose) {
            for pkg in &dep_ctx.packages {
                println!(
                    "Package '{}' v{} ({} neurons):\n",
                    pkg.name,
                    pkg.version,
                    pkg.exported_neurons.len()
                );
                print_neuron_map(&pkg.exported_neurons, verbose);
            }
        }
    }

    // If no file was provided, either we're done (flag-only mode) or it's an error
    let file = match file {
        Some(f) => f,
        None => {
            if list_stdlib || list_package.is_some() || available {
                return Ok(());
            }
            return Err(miette::miette!("a file argument is required unless --stdlib, --package, or --available is used"));
        }
    };

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
            .map(|p| format!("{}{}: {}", if p.variadic { "*" } else { "" }, p.name, p.shape))
            .collect();

        let outputs: Vec<String> = neuron
            .outputs
            .iter()
            .map(|p| format!("{}{}: {}", if p.variadic { "*" } else { "" }, p.name, p.shape))
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

/// List all stdlib neurons (primitives from registry + composites from .ns files)
fn list_stdlib_neurons(verbose: bool) -> miette::Result<()> {
    let registry = StdlibRegistry::new();
    let primitive_names = registry.primitives();

    println!("Primitives ({}):\n", primitive_names.len());
    for name in &primitive_names {
        if verbose {
            if let Some(impl_ref) = registry.lookup(name) {
                println!("  {} (impl: {})", name, impl_ref.full_name());
            } else {
                println!("  {}", name);
            }
        } else {
            println!("  {}", name);
        }
    }

    // Load stdlib composites
    println!();
    match stdlib::load_stdlib() {
        Ok(stdlib_program) => {
            let mut composite_names: Vec<&String> = stdlib_program.neurons.keys().collect();
            composite_names.sort();

            println!("Composites ({}):\n", composite_names.len());
            for name in &composite_names {
                let neuron = &stdlib_program.neurons[*name];
                let inputs: Vec<String> = neuron
                    .inputs
                    .iter()
                    .map(|p| format!("{}{}: {}", if p.variadic { "*" } else { "" }, p.name, p.shape))
                    .collect();
                let outputs: Vec<String> = neuron
                    .outputs
                    .iter()
                    .map(|p| format!("{}{}: {}", if p.variadic { "*" } else { "" }, p.name, p.shape))
                    .collect();

                if verbose {
                    println!("  {}", name);
                    println!("    in:  {}", inputs.join(", "));
                    println!("    out: {}", outputs.join(", "));
                } else {
                    println!("  {}", name);
                }
            }
        }
        Err(e) => {
            eprintln!("Warning: Could not load stdlib composites: {}", e);
        }
    }

    println!();
    Ok(())
}

/// List neurons from a specific fetched dependency package
fn list_package_neurons(pkg_name: &str, verbose: bool) -> miette::Result<()> {
    let dep_ctx = try_load_dependencies_from_cwd(verbose)
        .ok_or_else(|| miette::miette!("No Axon.lock found — run `neuroscript fetch` first"))?;

    let pkg = dep_ctx
        .get_package(pkg_name)
        .ok_or_else(|| miette::miette!("Package '{}' not found in dependencies", pkg_name))?;

    println!(
        "Package '{}' v{} ({} neurons):\n",
        pkg.name,
        pkg.version,
        pkg.exported_neurons.len()
    );
    print_neuron_map(&pkg.exported_neurons, verbose);

    Ok(())
}

/// Print a map of neurons in a consistent format
fn print_neuron_map(neurons: &std::collections::HashMap<String, neuroscript::NeuronDef>, verbose: bool) {
    let mut names: Vec<&String> = neurons.keys().collect();
    names.sort();

    for name in names {
        let neuron = &neurons[name];
        let kind = match &neuron.body {
            NeuronBody::Primitive(_) => "primitive",
            NeuronBody::Graph { .. } => "composite",
        };

        let inputs: Vec<String> = neuron
            .inputs
            .iter()
            .map(|p| format!("{}{}: {}", if p.variadic { "*" } else { "" }, p.name, p.shape))
            .collect();

        let outputs: Vec<String> = neuron
            .outputs
            .iter()
            .map(|p| format!("{}{}: {}", if p.variadic { "*" } else { "" }, p.name, p.shape))
            .collect();

        println!("  {} ({})", name, kind);
        if verbose {
            println!("    in:  {}", inputs.join(", "));
            println!("    out: {}", outputs.join(", "));
        }
    }
    println!();
}

/// Keygen command: Generate an Ed25519 keypair for package signing
fn cmd_keygen(name: String) -> miette::Result<()> {
    use neuroscript::package::security;

    let (signing_key, verifying_key) = security::generate_keypair();

    let (key_path, pub_path) = security::save_keypair(&name, &signing_key)
        .map_err(|e| miette::miette!("Failed to save keypair: {}", e))?;

    let pub_key_str = security::format_publisher_key(&verifying_key);

    println!("✓ Generated Ed25519 keypair '{}'", name);
    println!("  Private key: {}", key_path.display());
    println!("  Public key:  {}", pub_path.display());
    println!("\n  Publisher key for Axon.toml:");
    println!("  {}", pub_key_str);

    Ok(())
}

/// Publish command: Sign and prepare a package for distribution
fn cmd_publish(
    path: Option<PathBuf>,
    key: Option<PathBuf>,
    no_sign: bool,
    verbose: bool,
) -> miette::Result<()> {
    use neuroscript::package::{security, Manifest};

    let package_dir = path.unwrap_or_else(|| PathBuf::from("."));
    let manifest_path = package_dir.join("Axon.toml");

    if !manifest_path.exists() {
        return Err(miette::miette!(
            "No Axon.toml found in {}",
            package_dir.display()
        ));
    }

    let manifest = Manifest::from_path(&manifest_path)
        .into_diagnostic()
        .wrap_err("Failed to load Axon.toml")?;

    if verbose {
        println!(
            "Publishing {} v{}",
            manifest.package.name, manifest.package.version
        );
    }

    // Compute file checksums
    let checksums = security::compute_checksums(&package_dir)
        .map_err(|e| miette::miette!("Failed to compute checksums: {}", e))?;

    if checksums.is_empty() {
        return Err(miette::miette!(
            "No .ns files found in package directory"
        ));
    }

    if verbose {
        println!("  Computed checksums for {} files", checksums.len());
        for (path, hash) in &checksums {
            println!("    {} {}", &hash[..15], path);
        }
    }

    // Compute overall checksum
    let overall = security::compute_overall_checksum(&checksums);

    if verbose {
        println!("  Overall checksum: {}", &overall[..22]);
    }

    // Sign if requested
    let (signature, publisher_key) = if no_sign {
        if verbose {
            println!("  Skipping signature (--no-sign)");
        }
        (None, None)
    } else {
        // Find signing key
        let key_path = if let Some(k) = key {
            k
        } else {
            // Auto-discover key for this package
            match security::discover_signing_key(&manifest.package.name)
                .map_err(|e| miette::miette!("Failed to discover signing key: {}", e))?
            {
                Some(p) => p,
                None => {
                    eprintln!(
                        "Warning: No signing key found for '{}'. Use --key or run `neuroscript keygen {}`",
                        manifest.package.name, manifest.package.name
                    );
                    eprintln!("  Publishing without signature (checksums only)");
                    return finish_publish(&manifest_path, checksums, overall, None, None, verbose);
                }
            }
        };

        if verbose {
            println!("  Signing with key: {}", key_path.display());
        }

        let signing_key = security::load_signing_key(&key_path)
            .map_err(|e| miette::miette!("Failed to load signing key: {}", e))?;

        let sig = security::sign_checksum(&overall, &signing_key);
        let pub_key = security::format_publisher_key(&signing_key.verifying_key());

        (Some(sig), Some(pub_key))
    };

    finish_publish(
        &manifest_path,
        checksums,
        overall,
        signature,
        publisher_key,
        verbose,
    )
}

fn finish_publish(
    manifest_path: &Path,
    checksums: std::collections::BTreeMap<String, String>,
    overall: String,
    signature: Option<String>,
    publisher_key: Option<String>,
    verbose: bool,
) -> miette::Result<()> {
    use neuroscript::package::{manifest::Security, security};

    let sec = Security {
        publisher_key,
        signature,
        checksum: Some(overall),
        checksums,
    };

    security::update_manifest_security(manifest_path, sec)
        .map_err(|e| miette::miette!("Failed to update Axon.toml: {}", e))?;

    if verbose {
        println!("  Updated: {}", manifest_path.display());
    }

    println!("✓ Package prepared for distribution");
    println!("  Axon.toml updated with checksums and signature");

    Ok(())
}

/// Verify command: Verify package checksums and signature
fn cmd_verify(path: Option<PathBuf>, verbose: bool) -> miette::Result<()> {
    use neuroscript::package::{security, Manifest};

    let package_dir = path.unwrap_or_else(|| PathBuf::from("."));
    let manifest_path = package_dir.join("Axon.toml");

    if !manifest_path.exists() {
        return Err(miette::miette!(
            "No Axon.toml found in {}",
            package_dir.display()
        ));
    }

    let manifest = Manifest::from_path(&manifest_path)
        .into_diagnostic()
        .wrap_err("Failed to load Axon.toml")?;

    let security = manifest.security.as_ref().ok_or_else(|| {
        miette::miette!("No [security] section in Axon.toml — run `neuroscript publish` first")
    })?;

    if verbose {
        println!(
            "Verifying {} v{}",
            manifest.package.name, manifest.package.version
        );
    }

    let report = security::verify_package(&package_dir, security)
        .map_err(|e| miette::miette!("Verification failed: {}", e))?;

    // Print results
    if report.checksums_valid {
        println!("✓ File checksums: all {} files verified", security.checksums.len());
    } else {
        println!("✗ File checksums: FAILED");
        for f in &report.failed_files {
            println!("    Modified: {}", f);
        }
    }

    for f in &report.missing_files {
        println!("    Missing: {}", f);
    }
    for f in &report.extra_files {
        if verbose {
            println!("    New file: {}", f);
        }
    }

    if report.overall_checksum_valid {
        println!("✓ Overall checksum: valid");
    } else {
        println!("✗ Overall checksum: MISMATCH");
    }

    match report.signature_valid {
        Some(true) => println!("✓ Signature: valid"),
        Some(false) => println!("✗ Signature: INVALID"),
        None => {
            if security.signature.is_some() {
                println!("? Signature: could not verify (missing publisher key)");
            } else {
                println!("- Signature: not present");
            }
        }
    }

    if report.is_valid() {
        println!("\n✓ Package verification passed");
        Ok(())
    } else {
        return Err(miette::miette!("Package verification FAILED"));
    }
}

// ============================================================================
// Dependency Loading Helpers
// ============================================================================

/// Walk up from the given file's directory to find an Axon.lock file
fn find_lockfile(start: &Path) -> Option<PathBuf> {
    let start_dir = if start.is_file() {
        start.parent().unwrap_or(Path::new("."))
    } else {
        start
    };

    let mut current = start_dir.to_path_buf();
    loop {
        let lockfile = current.join("Axon.lock");
        if lockfile.exists() {
            return Some(lockfile);
        }
        if !current.pop() {
            return None;
        }
    }
}

/// Try to load dependencies from an Axon.lock found near the given file.
/// Returns None if no lockfile is found or loading fails (non-fatal).
fn try_load_dependencies(file: &Path, verbose: bool) -> Option<DependencyContext> {
    let lockfile_path = find_lockfile(file)?;

    if verbose {
        println!("Found lockfile: {}", lockfile_path.display());
    }

    match package::load_dependencies(&lockfile_path) {
        Ok(ctx) => {
            if verbose {
                let total_neurons: usize = ctx.packages.iter().map(|p| p.exported_neurons.len()).sum();
                println!(
                    "✓ Loaded {} packages ({} neurons) from dependencies",
                    ctx.packages.len(),
                    total_neurons
                );
            }
            Some(ctx)
        }
        Err(e) => {
            if verbose {
                eprintln!("Warning: Failed to load dependencies: {}", e);
            }
            None
        }
    }
}

/// Try to load dependencies from the current working directory
fn try_load_dependencies_from_cwd(verbose: bool) -> Option<DependencyContext> {
    let cwd = std::env::current_dir().ok()?;
    try_load_dependencies(&cwd, verbose)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Read source file with error handling
fn read_source(file: &Path) -> miette::Result<String> {
    fs::read_to_string(file)
        .into_diagnostic()
        .wrap_err_with(|| format!("Failed to read {}", file.display()))
}

/// Render validation errors as miette diagnostics with source code context.
///
/// Errors that carry a source span get full source-code underlining via miette.
/// Errors without spans are printed as plain text. The first spanned error is
/// returned as the primary miette::Result error; all others are printed to stderr.
fn render_validation_errors(
    file: &Path,
    errors: &[neuroscript::ValidationError],
) -> miette::Result<()> {
    // Try to read the source for span-bearing errors
    let source = fs::read_to_string(file).ok();

    let mut first_report: Option<miette::Report> = None;
    let mut plain_errors: Vec<String> = Vec::new();

    for error in errors {
        if error.span().is_some() {
            if let Some(ref src) = source {
                let report = miette::Report::from(error.clone())
                    .with_source_code(NamedSource::new(
                        file.to_string_lossy(),
                        src.clone(),
                    ));
                if first_report.is_none() {
                    first_report = Some(report);
                } else {
                    eprintln!("{:?}", report);
                }
                continue;
            }
        }
        // No span or no source — always collect for display
        plain_errors.push(format!("  {}", error));
    }

    // Print all non-spanned errors regardless of ordering
    for msg in &plain_errors {
        eprintln!("{}", msg);
    }

    if let Some(report) = first_report {
        return Err(report);
    }

    // Fallback: no spanned errors, use the old-style summary
    let detail = plain_errors.join("\n");
    Err(miette::miette!(
        "Validation failed with {} error(s):\n{}",
        errors.len(),
        detail
    ))
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
            .map(|p| format!("{}{}: {}", if p.variadic { "*" } else { "" }, p.name, p.shape))
            .collect();

        let outputs: Vec<String> = neuron
            .outputs
            .iter()
            .map(|p| format!("{}{}: {}", if p.variadic { "*" } else { "" }, p.name, p.shape))
            .collect();

        println!("  {} ({})", name, kind);
        println!("    in:  {}", inputs.join(", "));
        println!("    out: {}", outputs.join(", "));

        if verbose {
            if let NeuronBody::Graph {
                connections: conns, ..
            } = &neuron.body
            {
                println!("    connections: {:?}", conns);
            }
        }

        println!();
    }
}

/// Infer neuron name from filename, or fail with helpful message
fn infer_neuron_name(file: &Path, program: &neuroscript::Program) -> miette::Result<String> {
    // Extract filename without extension
    let filename = file
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| miette::miette!("Invalid file path: {}", file.display()))?;

    // Convert snake_case or kebab-case to PascalCase
    let neuron_name = filename
        .split(['_', '-'])
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
        let mut msg = format!(
            "No neuron matching filename '{}' found (tried: '{}')",
            file.display(),
            neuron_name
        );
        if !available.is_empty() {
            msg.push_str(&format!("\n  Available neurons: {}", available.join(", ")));
        }
        msg.push_str("\n  Use --neuron <NAME> to specify explicitly");
        return Err(miette::miette!("{}", msg));
    }
}
