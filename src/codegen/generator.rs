//! Core PyTorch code generation for neurons
//!
//! Orchestrates the generation of Python nn.Module classes from
//! NeuroScript neuron definitions.

use std::collections::HashSet;
use std::fmt::Write;
use crate::interfaces::*;
use super::{instantiation, forward};

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
    /// Create a new code generator
    pub(crate) fn new(program: &'a Program) -> Self {
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

    /// Generate imports
    fn generate_imports(&self) -> String {
        let mut output = String::new();

        writeln!(output, "import torch").unwrap();
        writeln!(output, "import torch.nn as nn").unwrap();

        // Generate imports for used primitives
        let primitives: Vec<String> = self.used_primitives.iter().cloned().collect();
        let imports = self.registry.generate_imports(&primitives);

        for import in imports {
            writeln!(output, "{}", import).unwrap();
        }

        writeln!(output).unwrap();
        output
    }
}

/// Generate PyTorch code for a specific neuron (PUBLIC API)
pub fn generate_pytorch(program: &Program, neuron_name: &str) -> Result<String, CodegenError> {
    let neuron = program.neurons.get(neuron_name)
        .ok_or_else(|| CodegenError::NeuronNotFound(neuron_name.to_string()))?;

    let mut generator = CodeGenerator::new(program);

    // First pass: generate neuron code to collect dependencies
    let neuron_code = generator.generate_neuron(neuron)?;

    // Generate imports based on used primitives
    let imports = generator.generate_imports();

    // Combine imports and code
    Ok(format!("{}{}", imports, neuron_code))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_new() {
        let program = Program::new();
        let gen = CodeGenerator::new(&program);
        assert_eq!(gen.node_counter, 0);
        assert!(gen.used_primitives.is_empty());
    }

    #[test]
    fn test_generate_imports() {
        let program = Program::new();
        let mut gen = CodeGenerator::new(&program);
        gen.used_primitives.insert("Linear".to_string());
        gen.used_primitives.insert("GELU".to_string());

        let imports = gen.generate_imports();
        assert!(imports.contains("import torch"));
        assert!(imports.contains("import torch.nn as nn"));
    }
}
