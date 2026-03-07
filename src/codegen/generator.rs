//! Core PyTorch code generation for neurons
//!
//! Orchestrates the generation of Python nn.Module classes from
//! NeuroScript neuron definitions.

use super::utils::sanitize_python_ident;
use super::{forward, instantiation};
use crate::interfaces::*;
use miette::Diagnostic;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use thiserror::Error;

/// Codegen error type
#[derive(Debug, Error, Diagnostic)]
pub enum CodegenError {
    #[error("Neuron '{0}' not found")]
    NeuronNotFound(String),
    #[error("Invalid connection: {0}")]
    InvalidConnection(String),
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),
}

/// Result of shape check generation containing condition and dimension bindings
#[derive(Debug)]
pub struct ShapeCheckResult {
    /// Boolean condition for runtime shape checking (e.g., "x.ndim == 2 and x.shape[1] == 512")
    pub condition: String,
    /// Dimension binding statements (e.g., vec!["d = x.shape[1]"])
    /// These should be emitted before the pipeline code in a match arm
    pub bindings: Vec<String>,
    /// Guard expression (if present and uses captured dimensions, should be checked after bindings)
    pub guard_condition: Option<String>,
}

/// Lazy binding: (call_name, args, kwargs)
pub(crate) type LazyBinding = (String, Vec<Value>, Vec<Kwarg>);

/// Code generator state
pub struct CodeGenerator<'a> {
    pub program: &'a Program,
    pub registry: StdlibRegistry,

    /// Counter for generating unique node IDs
    pub node_counter: usize,

    /// Set of primitive neurons used (for imports)
    pub used_primitives: HashSet<String>,

    /// Mapping from IR endpoints to Python variable names
    pub var_names: HashMap<String, String>,

    /// Mapping from Call endpoint keys to module instance names
    pub call_to_module: HashMap<String, String>,

    /// Parameters of the current neuron being generated
    pub current_neuron_params: HashSet<String>,

    /// Names of parameters with `: Neuron` type annotation (higher-order neuron params)
    pub neuron_typed_params: HashSet<String>,

    /// Dimension bindings from match pattern captures (e.g., "d" -> "x.shape[1]")
    /// Used to resolve dimension references in match arm pipelines
    pub binding_context: HashMap<String, String>,

    /// Lazy bindings from let: blocks (name -> (call_name, args, kwargs))
    pub lazy_bindings: HashMap<String, LazyBinding>,

    /// Shape inference context (resolved dimensions and node output shapes)
    /// Used for emitting shape assertions and documentation
    pub inference_ctx: InferenceContext,

    /// Mapping from binding name to the neuron it calls (e.g., "block_0" -> "TransformerBlock")
    /// Used to look up output shapes for bound module calls
    pub binding_to_call_name: HashMap<String, String>,

    /// Mapping from binding name to its unroll group info
    /// Used to generate nn.ModuleList and for loops
    pub binding_to_unroll_group: HashMap<String, UnrollGroupInfo>,

    /// Mapping from aggregate name to (base_name, count, is_static) for direct aggregate references in graph
    /// is_static=true means weight-sharing: one class-level instance called N times
    pub aggregate_to_group: HashMap<String, (String, Value, bool)>,

    /// Last shape comment emitted, used to suppress duplicates
    pub last_emitted_shape: Option<String>,
}

/// Declare embedded primitive modules: generates constants and a lookup function.
///
/// Every primitive Python file under `neuroscript_runtime/primitives/` is compiled
/// into the binary via `include_str!`.  Adding a new primitive file requires only
/// a single new entry in the `embedded_primitives!` invocation below.
macro_rules! embedded_primitives {
    ( $( $name:ident, $suffix:literal => $file:literal ),+ $(,)? ) => {
        mod embedded_sources {
            $(
                pub const $name: &str =
                    include_str!(concat!("../../neuroscript_runtime/primitives/", $file));
            )+
        }

        /// Look up the embedded Python source for a module path
        /// like `"neuroscript_runtime.primitives.linear"`.
        fn embedded_source_for_module(module_path: &str) -> Option<&'static str> {
            let suffix = module_path.strip_prefix("neuroscript_runtime.primitives.")?;
            match suffix {
                $( $suffix => Some(embedded_sources::$name), )+
                _ => None,
            }
        }
    };
}

embedded_primitives! {
    ACTIVATIONS,    "activations"    => "activations.py",
    ATTENTION,      "attention"      => "attention.py",
    CONNECTIONS,    "connections"    => "connections.py",
    CONVOLUTIONS,   "convolutions"   => "convolutions.py",
    EMBEDDINGS,     "embeddings"     => "embeddings.py",
    LINEAR,         "linear"         => "linear.py",
    LOGGING,        "logging"        => "logging.py",
    NORMALIZATION,  "normalization"  => "normalization.py",
    OPERATIONS,     "operations"     => "operations.py",
    POOLING,        "pooling"        => "pooling.py",
    REGULARIZATION, "regularization" => "regularization.py",
    STRUCTURAL,     "structural"     => "structural.py",
}

/// Strip the module-level docstring and import lines from an embedded Python source,
/// emitting only the class definitions.  The caller provides a unified import block
/// so per-file imports would be duplicates.
fn strip_preamble(source: &str) -> String {
    let mut out = String::new();
    let mut in_module_docstring = false;
    let mut preamble_done = false;

    for line in source.lines() {
        let trimmed = line.trim();

        if !preamble_done {
            // Detect opening of a module-level triple-quoted docstring
            if !in_module_docstring && trimmed.starts_with("\"\"\"") {
                in_module_docstring = true;
                // Check if the docstring opens and closes on the same line
                // (e.g. `"""one-liner"""`)
                if trimmed.len() > 3 && trimmed[3..].contains("\"\"\"") {
                    in_module_docstring = false;
                }
                continue;
            }
            // Inside a module-level docstring — skip until closing triple-quote
            if in_module_docstring {
                if trimmed.contains("\"\"\"") {
                    in_module_docstring = false;
                }
                continue;
            }
            // Skip import / from lines
            if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
                continue;
            }
            // Skip blank lines that are still part of the preamble
            if trimmed.is_empty() {
                continue;
            }
            // First non-preamble line (e.g. `class Foo(nn.Module):`)
            preamble_done = true;
        }

        out.push_str(line);
        out.push('\n');
    }

    out
}

/// Options for code generation.
#[derive(Debug, Clone)]
pub struct CodegenOptions {
    /// When true, inline primitive class definitions instead of importing from neuroscript_runtime.
    pub bundle: bool,
}

impl Default for CodegenOptions {
    fn default() -> Self {
        Self { bundle: false }
    }
}


impl<'a> CodeGenerator<'a> {
    /// Create a new code generator with an optional inference context
    pub(crate) fn new(program: &'a Program, inference_ctx: InferenceContext) -> Self {
        Self {
            program,
            registry: StdlibRegistry::new(),
            node_counter: 0,
            used_primitives: HashSet::new(),
            var_names: std::collections::HashMap::new(),
            call_to_module: std::collections::HashMap::new(),
            current_neuron_params: HashSet::new(),
            neuron_typed_params: HashSet::new(),
            binding_context: std::collections::HashMap::new(),
            lazy_bindings: std::collections::HashMap::new(),
            inference_ctx,
            binding_to_call_name: std::collections::HashMap::new(),
            binding_to_unroll_group: std::collections::HashMap::new(),
            aggregate_to_group: std::collections::HashMap::new(),
            last_emitted_shape: None,
        }
    }

    /// Generate Python code for a neuron
    fn generate_neuron(&mut self, neuron: &NeuronDef) -> Result<String, CodegenError> {
        self.current_neuron_params = neuron.params.iter().map(|p| p.name.clone()).collect();
        self.neuron_typed_params = neuron
            .params
            .iter()
            .filter(|p| p.type_annotation.as_ref() == Some(&ParamType::Neuron))
            .map(|p| p.name.clone())
            .collect();
        let mut output = String::new();

        // Generate class definition
        writeln!(output, "class {}(nn.Module):", sanitize_python_ident(&neuron.name)).unwrap();

        // Generate __init__
        self.generate_init(&mut output, neuron)?;

        writeln!(output).unwrap();

        // Generate forward
        self.generate_forward(&mut output, neuron)?;

        Ok(output)
    }

    /// Generate __init__ method
    fn generate_init(
        &mut self,
        output: &mut String,
        neuron: &NeuronDef,
    ) -> Result<(), CodegenError> {
        // Build parameter list
        let params = if neuron.params.is_empty() {
            "self".to_string()
        } else {
            let param_strs: Vec<String> = neuron
                .params
                .iter()
                .map(|p| {
                    let safe_name = sanitize_python_ident(&p.name);
                    if let Some(default) = &p.default {
                        format!("{}={}", safe_name, self.value_to_python(default))
                    } else {
                        safe_name
                    }
                })
                .collect();
            format!("self, {}", param_strs.join(", "))
        };

        writeln!(output, "    def __init__({}):", params).unwrap();
        writeln!(output, "        super().__init__()").unwrap();

        // Store parameters as instance variables (needed for guards)
        for param in &neuron.params {
            let safe_name = sanitize_python_ident(&param.name);
            writeln!(output, "        self.{} = {}", safe_name, safe_name).unwrap();
        }

        match &neuron.body {
            NeuronBody::Primitive(_) => {
                // Primitives don't instantiate sub-modules
                writeln!(output, "        pass").unwrap();
            }
            NeuronBody::Graph {
                context_bindings,
                connections,
                ..
            } => {
                // Instantiate all called neurons as modules
                instantiation::generate_module_instantiations(
                    self,
                    output,
                    context_bindings,
                    connections,
                )?;
            }
        }

        Ok(())
    }

    /// Generate forward method
    fn generate_forward(
        &mut self,
        output: &mut String,
        neuron: &NeuronDef,
    ) -> Result<(), CodegenError> {
        // Determine input parameter(s)
        let input_params = if neuron.inputs.len() == 1 && neuron.inputs[0].variadic {
            // Variadic port: single tuple parameter (matches Python runtime convention)
            sanitize_python_ident(&neuron.inputs[0].name)
        } else if neuron.inputs.len() == 1 && neuron.inputs[0].name == "default" {
            "x".to_string()
        } else {
            neuron
                .inputs
                .iter()
                .map(|p| sanitize_python_ident(&p.name))
                .collect::<Vec<_>>()
                .join(", ")
        };

        writeln!(output, "    def forward(self, {}):", input_params).unwrap();

        // Register all params in var_names with self. prefix so that lazy
        // instantiation can resolve them (e.g., dim → self.dim, layer → self.layer).
        // Uses or_insert_with intentionally: context bindings registered in __init__
        // may already have a mapping (e.g., self.binding_name) which should take
        // precedence. Params are a fallback for names not yet in the map.
        for param in &neuron.params {
            let safe_name = sanitize_python_ident(&param.name);
            self.var_names
                .entry(param.name.clone())
                .or_insert_with(|| format!("self.{}", safe_name));
        }

        match &neuron.body {
            NeuronBody::Primitive(_impl_ref) => {
                // Primitive neurons just pass through
                writeln!(output, "        return {}", input_params).unwrap();
            }
            NeuronBody::Graph { connections, .. } => {
                forward::generate_forward_body(
                    self,
                    output,
                    connections,
                    &neuron
                        .inputs
                        .iter()
                        .map(|p| p.name.as_str())
                        .collect::<Vec<_>>(),
                )?;
            }
        }

        Ok(())
    }
}

/// Collect all composite neuron dependencies recursively
fn collect_dependencies(
    neuron_name: &str,
    program: &Program,
    visited: &mut HashSet<String>,
    result: &mut Vec<String>,
) -> Result<(), CodegenError> {
    // Skip if already visited
    if visited.contains(neuron_name) {
        return Ok(());
    }
    visited.insert(neuron_name.to_string());

    // Get the neuron definition
    let neuron = program
        .neurons
        .get(neuron_name)
        .ok_or_else(|| CodegenError::NeuronNotFound(neuron_name.to_string()))?;

    // Only process composite neurons (primitives are just imports)
    if let NeuronBody::Graph {
        context_bindings,
        connections,
        ..
    } = &neuron.body
    {
        // Collect all called neuron names from bindings and connections
        let mut called_neurons = HashSet::new();

        // Collect from context bindings
        for binding in context_bindings {
            called_neurons.insert(binding.call_name.clone());
        }

        // Collect from connections
        collect_calls_from_connections(connections, &mut called_neurons);

        // Recursively collect dependencies (depth-first)
        for called_name in called_neurons {
            // Check if it's a composite neuron in the program
            if let Some(neuron_def) = program.neurons.get(&called_name) {
                if !neuron_def.is_primitive() {
                    // Recursively collect dependencies of this composite neuron
                    collect_dependencies(&called_name, program, visited, result)?;
                }
            }
            // If not in program, it's assumed to be a primitive (ignore)
        }

        // Add this neuron to the result after its dependencies
        result.push(neuron_name.to_string());
    }

    Ok(())
}

/// Collect all neuron names called in connections
fn collect_calls_from_connections(connections: &[Connection], result: &mut HashSet<String>) {
    for conn in connections {
        collect_calls_from_endpoint(&conn.source, result);
        collect_calls_from_endpoint(&conn.destination, result);
    }
}

/// Recursively collect neuron names from an endpoint
fn collect_calls_from_endpoint(endpoint: &Endpoint, result: &mut HashSet<String>) {
    match endpoint {
        Endpoint::Call { name, .. } => {
            result.insert(name.clone());
        }
        Endpoint::Tuple(_port_refs) => {
            // Tuples contain PortRef, not Endpoint - they don't call neurons
        }
        Endpoint::Match(match_expr) => {
            for arm in &match_expr.arms {
                for ep in &arm.pipeline {
                    collect_calls_from_endpoint(ep, result);
                }
            }
        }
        Endpoint::If(if_expr) => {
            for branch in &if_expr.branches {
                for ep in &branch.pipeline {
                    collect_calls_from_endpoint(ep, result);
                }
            }
            if let Some(else_branch) = &if_expr.else_branch {
                for ep in else_branch {
                    collect_calls_from_endpoint(ep, result);
                }
            }
        }
        Endpoint::Ref(_) => {
            // Port references don't call neurons
        }
        Endpoint::Reshape(reshape) => {
            // Collect neuron names from annotation strategies
            if let Some(ref annotation) = reshape.annotation {
                let strategy = match annotation {
                    TransformAnnotation::Reduce(s, _) => s,
                    TransformAnnotation::Repeat(s, _) => s,
                };
                if let TransformStrategy::Neuron { name, .. } = strategy {
                    result.insert(name.clone());
                }
            }
        }
        Endpoint::Wrap(_) => {
            // @wrap should be desugared before codegen
        }
        // Endpoint::Unroll removed — expanded before codegen
    }
}

/// Generate PyTorch code for a specific neuron (PUBLIC API)
pub fn generate_pytorch(program: &Program, neuron_name: &str) -> Result<String, CodegenError> {
    generate_pytorch_with_options(program, neuron_name, &CodegenOptions::default())
}

/// Generate PyTorch code for a specific neuron with options.
///
/// When `options.bundle` is true, primitive class definitions are inlined
/// into the output so the generated file is self-contained (only requires `torch`).
pub fn generate_pytorch_with_options(
    program: &Program,
    neuron_name: &str,
    options: &CodegenOptions,
) -> Result<String, CodegenError> {
    // Verify the requested neuron exists
    let _neuron = program
        .neurons
        .get(neuron_name)
        .ok_or_else(|| CodegenError::NeuronNotFound(neuron_name.to_string()))?;

    // Collect all composite dependencies in topological order
    let mut visited = HashSet::new();
    let mut dependencies = Vec::new();
    collect_dependencies(neuron_name, program, &mut visited, &mut dependencies)?;

    // Generate globals
    let mut globals_output = String::new();
    let dummy_gen = CodeGenerator::new(program, InferenceContext::new());
    for global in &program.globals {
        writeln!(
            globals_output,
            "{} = {}",
            sanitize_python_ident(&global.name),
            dummy_gen.value_to_python(&global.value)
        )
        .unwrap();
    }
    if !program.globals.is_empty() {
        writeln!(globals_output).unwrap();
    }

    // Generate code for all dependencies in order
    let mut all_code = String::new();
    let mut all_primitives = HashSet::new();

    for dep_name in &dependencies {
        let neuron = program.neurons.get(dep_name).unwrap(); // Safe because we just collected it

        // Run shape inference for this neuron
        let inference_ctx = run_shape_inference_for_neuron(neuron, program);
        let mut generator = CodeGenerator::new(program, inference_ctx);

        // Generate neuron code
        let neuron_code = generator.generate_neuron(neuron)?;
        all_code.push_str(&neuron_code);
        all_code.push('\n');

        // Collect primitives used
        all_primitives.extend(generator.used_primitives);
    }

    let registry = StdlibRegistry::new();
    let primitives: Vec<String> = all_primitives.into_iter().collect();

    let mut imports_output = String::new();

    if options.bundle {
        // Bundle mode: emit a unified import block (superset of what all primitive
        // files use) then inline only the class definitions that are needed.
        writeln!(imports_output, "import math").unwrap();
        writeln!(imports_output, "from typing import List, Optional, Tuple, Union").unwrap();
        writeln!(imports_output, "import torch").unwrap();
        writeln!(imports_output, "import torch.nn as nn").unwrap();
        writeln!(imports_output, "import torch.nn.functional as F").unwrap();
        writeln!(imports_output).unwrap();

        // Inline only the needed primitive modules
        let modules = registry.modules_for_primitives(&primitives);
        for module_path in &modules {
            if let Some(source) = embedded_source_for_module(module_path) {
                writeln!(imports_output, "# --- inlined from {} ---\n", module_path).unwrap();
                imports_output.push_str(&strip_preamble(source));
                writeln!(imports_output).unwrap();
            }
        }
    } else {
        // Normal mode: generate import statements
        writeln!(imports_output, "import torch").unwrap();
        writeln!(imports_output, "import torch.nn as nn").unwrap();

        let imports = registry.generate_imports(&primitives);
        for import in imports {
            writeln!(imports_output, "{}", import).unwrap();
        }
        writeln!(imports_output).unwrap();
    }

    // Combine imports, globals and all neuron code
    Ok(format!("{}{}{}", imports_output, globals_output, all_code))
}

/// Run shape inference for a single neuron and return the inference context
/// This provides resolved dimensions and node output shapes for code generation
fn run_shape_inference_for_neuron(neuron: &NeuronDef, program: &Program) -> InferenceContext {
    use crate::shape::ShapeInferenceEngine;

    // Create a new inference context
    let mut ctx = InferenceContext::new();

    // Initialize context with neuron parameters (if they have defaults)
    for param in &neuron.params {
        if let Some(Value::Int(val)) = param.default {
            ctx.resolved_dims.insert(param.name.clone(), val as usize);
        }
    }

    // Register input shapes
    let input_shapes: Vec<Shape> = neuron.inputs.iter().map(|p| p.shape.clone()).collect();
    ctx.node_outputs.insert("in".to_string(), input_shapes);

    // Register individual named input ports
    for port in &neuron.inputs {
        if port.name != "default" {
            ctx.node_outputs
                .insert(port.name.clone(), vec![port.shape.clone()]);
        }
    }

    // If this is a composite neuron, run inference on its connections
    if let NeuronBody::Graph { connections, .. } = &neuron.body {
        let engine = ShapeInferenceEngine::new();
        // Process each connection to populate the context
        for conn in connections {
            // Attempt to check connection (ignore errors - this is best-effort for codegen)
            let _ = engine.check_connection(conn, &mut ctx, program);
        }
    }

    ctx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_new() {
        let program = Program::new();
        let ctx = InferenceContext::new();
        let gen = CodeGenerator::new(&program, ctx);
        assert_eq!(gen.node_counter, 0);
        assert!(gen.used_primitives.is_empty());
    }
}
