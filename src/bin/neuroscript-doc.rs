///! Documentation generator for NeuroScript
///!
///! Converts .ns files with triple-slash doc comments into markdown pages for Docusaurus.
use clap::{Parser, ValueEnum};
use neuroscript::doc_parser;
use neuroscript::grammar::NeuroScriptParser;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "neuroscript-doc")]
#[command(about = "Generate documentation from NeuroScript files", long_about = None)]
struct Args {
    /// Input .ns file to process
    #[arg(short, long)]
    input: PathBuf,

    /// Output directory for generated markdown
    #[arg(short, long)]
    output: PathBuf,

    /// Category for the documentation (primitives or stdlib)
    #[arg(short, long)]
    category: Category,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, Clone, ValueEnum)]
enum Category {
    Primitives,
    Stdlib,
}

fn main() {
    let args = Args::parse();

    if args.verbose {
        println!("Processing: {}", args.input.display());
    }

    // Read the source file
    let source = fs::read_to_string(&args.input).unwrap_or_else(|e| {
        eprintln!("Error reading file {}: {}", args.input.display(), e);
        std::process::exit(1);
    });

    // Parse the file
    let program = NeuroScriptParser::parse_program(&source).unwrap_or_else(|e| {
        eprintln!("Parse error in {}: {:?}", args.input.display(), e);
        std::process::exit(1);
    });

    // Generate markdown for each neuron with documentation
    let mut generated_count = 0;
    for (name, neuron) in &program.neurons {
        if let Some(doc) = &neuron.doc {
            let markdown = generate_markdown(name, neuron, doc, &args.category);

            // Create output filename (e.g., "Linear" -> "linear.md")
            let filename = format!("{}.md", name.to_lowercase());
            let output_path = args.output.join(&filename);

            // Write markdown file
            fs::write(&output_path, markdown).unwrap_or_else(|e| {
                eprintln!("Error writing {}: {}", output_path.display(), e);
                std::process::exit(1);
            });

            if args.verbose {
                println!("  Generated: {}", output_path.display());
            }

            generated_count += 1;
        }
    }

    if generated_count == 0 {
        eprintln!(
            "Warning: No neurons with documentation found in {}",
            args.input.display()
        );
    } else if args.verbose {
        println!(
            "Successfully generated {} documentation page(s)",
            generated_count
        );
    }
}

fn generate_markdown(
    name: &str,
    neuron: &neuroscript::NeuronDef,
    doc: &neuroscript::Documentation,
    category: &Category,
) -> String {
    let mut md = String::new();

    // Frontmatter
    md.push_str("---\n");
    md.push_str(&format!("sidebar_label: {}\n", name));
    md.push_str("---\n\n");

    // Title
    md.push_str(&format!("# {}\n\n", name));

    // Brief description (first paragraph)
    let brief = doc_parser::extract_brief(doc);
    if !brief.is_empty() {
        md.push_str(&brief);
        md.push_str("\n\n");
    }

    // Detailed description
    let description = doc_parser::extract_description(doc);
    if !description.is_empty() {
        md.push_str(&description);
        md.push_str("\n\n");
    }

    // Signature
    md.push_str("## Signature\n\n");
    md.push_str("```neuroscript\n");
    md.push_str(&format!("neuron {}(", name));

    // Parameters
    let params: Vec<String> = neuron
        .params
        .iter()
        .map(|p| {
            if let Some(default) = &p.default {
                format!("{}={:?}", p.name, default)
            } else {
                p.name.clone()
            }
        })
        .collect();
    md.push_str(&params.join(", "));
    md.push_str(")\n```\n\n");

    // Parameters section
    if let Some(params_doc) = doc.sections.get("Parameters") {
        md.push_str("## Parameters\n\n");
        md.push_str(params_doc);
        md.push_str("\n\n");
    }

    // Shape Contract section
    if let Some(shape_contract) = doc.sections.get("Shape Contract") {
        md.push_str("## Shape Contract\n\n");
        md.push_str(shape_contract);
        md.push_str("\n\n");
    }

    // Ports (inputs/outputs)
    if !neuron.inputs.is_empty() || !neuron.outputs.is_empty() {
        md.push_str("## Ports\n\n");

        if !neuron.inputs.is_empty() {
            md.push_str("**Inputs:**\n");
            for port in &neuron.inputs {
                md.push_str(&format!(
                    "- `{}`: `{}`\n",
                    port.name,
                    format_shape(&port.shape)
                ));
            }
            md.push_str("\n");
        }

        if !neuron.outputs.is_empty() {
            md.push_str("**Outputs:**\n");
            for port in &neuron.outputs {
                md.push_str(&format!(
                    "- `{}`: `{}`\n",
                    port.name,
                    format_shape(&port.shape)
                ));
            }
            md.push_str("\n");
        }
    }

    // Example section
    if let Some(example) = doc.sections.get("Example") {
        md.push_str("## Example\n\n");
        md.push_str(example);
        md.push_str("\n\n");
    }

    // Notes section
    if let Some(notes) = doc.sections.get("Notes") {
        md.push_str("## Notes\n\n");
        md.push_str(notes);
        md.push_str("\n\n");
    }

    // See Also section
    if let Some(see_also) = doc.sections.get("See Also") {
        md.push_str("## See Also\n\n");
        md.push_str(see_also);
        md.push_str("\n\n");
    }

    // Implementation (for primitives)
    if matches!(category, Category::Primitives) {
        if let neuroscript::NeuronBody::Primitive(impl_ref) = &neuron.body {
            md.push_str("## Implementation\n\n");
            md.push_str(&format!("```\n{:?}\n```\n\n", impl_ref));
        }
    }

    md
}

fn format_shape(shape: &neuroscript::Shape) -> String {
    let dims: Vec<String> = shape.dims.iter().map(|d| format!("{}", d)).collect();
    format!("[{}]", dims.join(", "))
}
