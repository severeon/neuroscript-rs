//! Core PyTorch code generation for neurons
//!
//! Orchestrates the generation of Python nn.Module classes from
//! NeuroScript neuron definitions.

use std::collections::HashSet;
use std::fmt::Write;
use crate::interfaces::*;
use super::{instantiation, forward};

/// Extract the neuron name from a binding value
/// Returns None if the value is not a neuron-related value
fn extract_neuron_name_from_value(value: &Value) -> Option<String> {
    match value {
        Value::Call { name, .. } => Some(name.clone()),
        Value::NeuronRef(name) => Some(name.clone()),
        Value::PartialCall { neuron, .. } => {
            // Recursively extract from the neuron field
            extract_neuron_name_from_value(neuron)
        }
        _ => None,
    }
}

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodegenError::NeuronNotFound(name) => write!(f, "Neuron '{}' not found", name),
            CodegenError::InvalidConnection(msg) => write!(f, "Invalid connection: {}", msg),
            CodegenError::UnsupportedFeature(msg) => write!(f, "Unsupported feature: {}", msg),
        }
    }
}

impl std::error::Error for CodegenError {}

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
            binding_context: std::collections::HashMap::new(),
            lazy_bindings: std::collections::HashMap::new(),
            inference_ctx,
        }
    }

    /// Generate Python code for a neuron
    fn generate_neuron(&mut self, neuron: &NeuronDef) -> Result<String, CodegenError> {
        self.current_neuron_params = neuron.params.iter().map(|p| p.name.clone()).collect();
        let mut output = String::new();

        // Generate class definition
        writeln!(output, "class {}(nn.Module):", neuron.name).unwrap();

        // Generate __init__
        self.generate_init(&mut output, neuron)?;

        writeln!(output).unwrap();

        // Generate forward
        self.generate_forward(&mut output, neuron)?;

        Ok(output)
    }

    /// Generate __init__ method
    fn generate_init(&mut self, output: &mut String, neuron: &NeuronDef) -> Result<(), CodegenError> {
        // Build parameter list
        let params = if neuron.params.is_empty() {
            "self".to_string()
        } else {
            let param_strs: Vec<String> = neuron.params.iter().map(|p| {
                if let Some(default) = &p.default {
                    format!("{}={}", p.name, self.value_to_python(default))
                } else {
                    p.name.clone()
                }
            }).collect();
            format!("self, {}", param_strs.join(", "))
        };

        writeln!(output, "    def __init__({}):", params).unwrap();
        writeln!(output, "        super().__init__()").unwrap();

        // Store parameters as instance variables (needed for guards)
        for param in &neuron.params {
            writeln!(output, "        self.{} = {}", param.name, param.name).unwrap();
        }

        match &neuron.body {
            NeuronBody::Primitive(_) => {
                // Primitives don't instantiate sub-modules
                writeln!(output, "        pass").unwrap();
            }
            NeuronBody::Graph { let_bindings, set_bindings, connections } => {
                // Instantiate all called neurons as modules
                instantiation::generate_module_instantiations(self, output, let_bindings, set_bindings, connections)?;
            }
        }

        Ok(())
    }

    /// Generate forward method
    fn generate_forward(&mut self, output: &mut String, neuron: &NeuronDef) -> Result<(), CodegenError> {
        // Determine input parameter(s)
        let input_params = if neuron.inputs.len() == 1 && neuron.inputs[0].name == "default" {
            "x".to_string()
        } else {
            neuron.inputs.iter()
                .map(|p| p.name.clone())
                .collect::<Vec<_>>()
                .join(", ")
        };

        writeln!(output, "    def forward(self, {}):", input_params).unwrap();

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
                    &neuron.inputs.iter().map(|p| p.name.as_str()).collect::<Vec<_>>()
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
    let neuron = program.neurons.get(neuron_name)
        .ok_or_else(|| CodegenError::NeuronNotFound(neuron_name.to_string()))?;

    // Only process composite neurons (primitives are just imports)
    if let NeuronBody::Graph { let_bindings, set_bindings, connections } = &neuron.body {
        // Collect all called neuron names from bindings and connections
        let mut called_neurons = HashSet::new();

        // Collect from set bindings
        for binding in set_bindings {
            if let Some(name) = extract_neuron_name_from_value(&binding.value) {
                called_neurons.insert(name);
            }
        }

        // Collect from let bindings
        for binding in let_bindings {
            if let Some(name) = extract_neuron_name_from_value(&binding.value) {
                called_neurons.insert(name);
            }
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
        Endpoint::Ref(_) => {
            // Port references don't call neurons
        }
    }
}

/// Generate PyTorch code for a specific neuron (PUBLIC API)
pub fn generate_pytorch(program: &Program, neuron_name: &str) -> Result<String, CodegenError> {
    // Verify the requested neuron exists
    let _neuron = program.neurons.get(neuron_name)
        .ok_or_else(|| CodegenError::NeuronNotFound(neuron_name.to_string()))?;

    // Collect all composite dependencies in topological order
    let mut visited = HashSet::new();
    let mut dependencies = Vec::new();
    collect_dependencies(neuron_name, program, &mut visited, &mut dependencies)?;

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

    // Generate imports based on all used primitives
    let registry = StdlibRegistry::new();
    let mut imports_output = String::new();
    writeln!(imports_output, "import torch").unwrap();
    writeln!(imports_output, "import torch.nn as nn").unwrap();

    let primitives: Vec<String> = all_primitives.iter().cloned().collect();
    let imports = registry.generate_imports(&primitives);
    for import in imports {
        writeln!(imports_output, "{}", import).unwrap();
    }
    writeln!(imports_output).unwrap();

    // Combine imports and all neuron code
    Ok(format!("{}{}", imports_output, all_code))
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
            ctx.node_outputs.insert(port.name.clone(), vec![port.shape.clone()]);
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
