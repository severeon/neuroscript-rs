//! PyTorch code generation from NeuroScript IR
//!
//! This module implements Phase 0 codegen: direct lowering from NeuroScript IR
//! to PyTorch Python code without shape inference or optimizations.
//!
//! Translation pipeline:
//! ```
//! NeuroScript IR
//!     ↓ Lowering
//! PyTorch nn.Module skeleton
//!     ↓ Emit Python source
//! Generated Python file
//! ```

use crate::ir::{Connection, Endpoint, NeuronBody, NeuronDef, Program, Value};
use crate::stdlib_registry::StdlibRegistry;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

#[derive(Debug)]
pub enum CodegenError {
    NeuronNotFound(String),
    InvalidConnection(String),
    UnsupportedFeature(String),
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

/// Code generator state
struct CodeGenerator<'a> {
    program: &'a Program,
    registry: StdlibRegistry,
    
    /// Counter for generating unique node IDs
    node_counter: usize,
    
    /// Set of primitive neurons used (for imports)
    used_primitives: HashSet<String>,
    
    /// Mapping from IR endpoints to Python variable names
    var_names: HashMap<String, String>,
    
    /// Mapping from Call endpoint keys to module instance names
    call_to_module: HashMap<String, String>,
}

impl<'a> CodeGenerator<'a> {
    fn new(program: &'a Program) -> Self {
        Self {
            program,
            registry: StdlibRegistry::new(),
            node_counter: 0,
            used_primitives: HashSet::new(),
            var_names: HashMap::new(),
            call_to_module: HashMap::new(),
        }
    }
    
    /// Generate a unique node ID
    fn next_node_id(&mut self) -> usize {
        let id = self.node_counter;
        self.node_counter += 1;
        id
    }
    
    /// Generate Python code for a neuron
    fn generate_neuron(&mut self, neuron: &NeuronDef) -> Result<String, CodegenError> {
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
        
        match &neuron.body {
            NeuronBody::Primitive(_) => {
                // Primitives don't instantiate sub-modules
                writeln!(output, "        pass").unwrap();
            }
            NeuronBody::Graph(connections) => {
                // Instantiate all called neurons as modules
                self.generate_module_instantiations(output, connections)?;
            }
        }
        
        Ok(())
    }
    
    /// Generate module instantiations in __init__
    fn generate_module_instantiations(&mut self, output: &mut String, connections: &[Connection]) -> Result<(), CodegenError> {
        // Collect all unique Call endpoints and assign them IDs
        let mut seen_calls: HashMap<String, (String, String, Vec<Value>, Vec<(String, Value)>)> = HashMap::new();
        
        for conn in connections {
            // Check both source and destination for Call endpoints
            for endpoint in [&conn.source, &conn.destination] {
                if let Endpoint::Call { name, args, kwargs } = endpoint {
                    let key = self.endpoint_key(endpoint);
                    if !seen_calls.contains_key(&key) {
                        let id = self.next_node_id();
                        let module_name = format!("{}_{}", self.snake_case(name), id);
                        // Store the mapping for use in forward generation
                        self.call_to_module.insert(key.clone(), module_name.clone());
                        seen_calls.insert(key, (name.clone(), module_name, args.clone(), kwargs.clone()));
                    }
                }
            }
        }
        
        // Generate instantiations in deterministic order
        let mut calls: Vec<_> = seen_calls.into_iter().collect();
        calls.sort_by(|a, b| a.1.1.cmp(&b.1.1)); // Sort by module_name for determinism
        
        for (_key, (name, module_name, args, kwargs)) in &calls {
            // Check if this is a primitive
            let is_primitive = if let Some(neuron) = self.program.neurons.get(name.as_str()) {
                neuron.is_primitive()
            } else {
                // Assume it's a primitive if not in program
                true
            };
           
            if is_primitive {
                self.used_primitives.insert(name.clone());
            }
            
            // Generate instantiation
            let args_str = args.iter()
                .map(|v| self.value_to_python(v))
                .collect::<Vec<_>>()
                .join(", ");
            
            let kwargs_str = if kwargs.is_empty() {
                String::new()
            } else {
                let kw: Vec<String> = kwargs.iter()
                    .map(|(k, v)| format!("{}={}", k, self.value_to_python(v)))
                    .collect();
                if args.is_empty() {
                    kw.join(", ")
                } else {
                    format!(", {}", kw.join(", "))
                }
            };
            
            writeln!(output, "        self.{} = {}({}{})", module_name, name, args_str, kwargs_str).unwrap();
        }
        
        // If no modules were instantiated, add pass
        if calls.is_empty() {
            writeln!(output, "        pass").unwrap();
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
            NeuronBody::Graph(connections) => {
                self.generate_forward_body(output, connections, &neuron.inputs.iter().map(|p| p.name.as_str()).collect::<Vec<_>>())?;
            }
        }
        
        Ok(())
    }
    
    /// Generate forward method body from connections
    fn generate_forward_body(&mut self, output: &mut String, connections: &[Connection], inputs: &[&str]) -> Result<(), CodegenError> {
        // Clear var names for this forward pass
        self.var_names.clear();
        
        // Map input ports to initial variables
        if inputs.len() == 1 && inputs[0] == "default" {
            self.var_names.insert("in".to_string(), "x".to_string());
        } else {
            for input in inputs {
                self.var_names.insert((*input).to_string(), (*input).to_string());
            }
        }
        
        let mut temp_var_counter = 0;
        
        // Build a map from Call endpoints to their result variable names
        let mut call_to_result: HashMap<String, String> = HashMap::new();
        
        // Process each connection
        for conn in connections {
            // Resolve the source to a variable name
            let source_var = match &conn.source {
                Endpoint::Ref(port_ref) => {
                    self.var_names.get(&port_ref.node)
                        .cloned()
                        .unwrap_or_else(|| port_ref.node.clone())
                }
                Endpoint::Tuple(refs) => {
                    let vars: Vec<String> = refs.iter()
                        .map(|r| self.var_names.get(&r.node).cloned().unwrap_or_else(|| r.node.clone()))
                        .collect();
                    format!("({})", vars.join(", "))
                }
                Endpoint::Call { name, args: _, kwargs: _ } => {
                    // This Call should have been processed in a previous connection
                    // Look it up in our call_to_result map
                    let key = self.endpoint_key(&conn.source);
                    call_to_result.get(&key)
                        .cloned()
                        .ok_or_else(|| CodegenError::InvalidConnection(
                            format!("Call to {} used as source before being defined", name)
                        ))?
                }
                Endpoint::Match(_) => {
                    return Err(CodegenError::UnsupportedFeature("Match expressions".to_string()));
                }
            };
            
            // Process the destination
            match &conn.destination {
                Endpoint::Ref(port_ref) => {
                    // Simple assignment - the source becomes this port's variable
                    self.var_names.insert(port_ref.node.clone(), source_var);
                }
                Endpoint::Tuple(refs) => {
                    // Tuple unpacking
                   let var_names: Vec<String> = refs.iter().map(|r| {
                        let v = format!("x{}", temp_var_counter);
                        temp_var_counter += 1;
                        self.var_names.insert(r.node.clone(), v.clone());
                        v
                    }).collect();
                    
                    writeln!(output, "        {} = {}", var_names.join(", "), source_var).unwrap();
                }
                Endpoint::Call { name, args: _, kwargs: _ } => {
                    // Generate a call to the module
                    let key = self.endpoint_key(&conn.destination);
                    let module_name = self.call_to_module.get(&key)
                        .cloned()
                        .ok_or_else(|| CodegenError::InvalidConnection(
                            format!("Module for call to {} not found", name)
                        ))?;
                    
                    let result_var = format!("x{}", temp_var_counter);
                    temp_var_counter += 1;
                    
                    // Generate the call
                    writeln!(output, "        {} = self.{}({})", result_var, module_name, source_var).unwrap();
                    
                    // Store the result for later reference
                    call_to_result.insert(key, result_var.clone());
                }
                Endpoint::Match(_) => {
                    return Err(CodegenError::UnsupportedFeature("Match expressions".to_string()));
                }
            }
        }
        
        // Return the output variable
        let output_var = self.var_names.get("out")
            .cloned()
            .unwrap_or_else(|| format!("x{}", temp_var_counter - 1));
        writeln!(output, "        return {}", output_var).unwrap();
        
        Ok(())
    }
    
    /// Generate a unique key for an endpoint (for tracking Call results)
    fn endpoint_key(&self, endpoint: &Endpoint) -> String {
        match endpoint {
            Endpoint::Call { name, args, kwargs } => {
                // Create a unique key based on the call signature
                let args_str = args.iter()
                    .map(|v| format!("{:?}", v))
                    .collect::<Vec<_>>()
                    .join(",");
                let kwargs_str = kwargs.iter()
                    .map(|(k, v)| format!("{}={:?}", k, v))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{}({};{})", name, args_str, kwargs_str)
            }
            _ => format!("{:?}", endpoint),
        }
    }
    
    /// Resolve an endpoint to a Python variable name (DEPRECATED - kept for reference)
    fn _resolve_endpoint(&mut self, endpoint: &Endpoint, _temp_var_counter: &mut usize) -> Result<String, CodegenError> {
        match endpoint {
            Endpoint::Ref(port_ref) => {
                Ok(self.var_names.get(&port_ref.node)
                    .cloned()
                    .unwrap_or_else(|| port_ref.node.clone()))
            }
            Endpoint::Tuple(refs) => {
                // For tuple sources, we return a tuple of variable names
                let vars: Vec<String> = refs.iter()
                    .map(|r| self.var_names.get(&r.node).cloned().unwrap_or_else(|| r.node.clone()))
                    .collect();
                Ok(format!("({})", vars.join(", ")))
            }
            Endpoint::Call { .. } => {
                // Calls as sources mean they're inline - not supported yet in this simple version
                Err(CodegenError::UnsupportedFeature("Inline calls as sources".to_string()))
            }
            Endpoint::Match(_) => {
                Err(CodegenError::UnsupportedFeature("Match expressions".to_string()))
            }
        }
    }
    
    /// Resolve an endpoint as a destination (DEPRECATED - kept for reference)
    fn _resolve_endpoint_dest(&mut self, endpoint: &Endpoint, source_var: &str, temp_var_counter: &mut usize, output: &mut String) -> Result<String, CodegenError> {
        match endpoint {
            Endpoint::Ref(_port_ref) => {
                // Simple assignment
                Ok(source_var.to_string())
            }
            Endpoint::Tuple(refs) => {
                // Tuple unpacking - the previous operation must have returned a tuple
                let result_var = format!("x{}", *temp_var_counter);
                *temp_var_counter += 1;
                
                let var_names: Vec<String> = refs.iter().map(|r| {
                    let v = format!("x{}", *temp_var_counter);
                    *temp_var_counter += 1;
                    self.var_names.insert(r.node.clone(), v.clone());
                    v
                }).collect();
                
                writeln!(output, "        {} = {}", var_names.join(", "), source_var).unwrap();
                Ok(result_var)
            }
            Endpoint::Call { name, args: _, kwargs: _ } => {
                // Generate a call to the module
                let id = self.next_node_id();
                let module_name = format!("{}_{}", self.snake_case(name), id);
                
                let result_var = format!("x{}", *temp_var_counter);
                *temp_var_counter += 1;
                
                // For now, assume single input
                writeln!(output, "        {} = self.{}({})", result_var, module_name, source_var).unwrap();
                
                Ok(result_var)
            }
            Endpoint::Match(_) => {
                Err(CodegenError::UnsupportedFeature("Match expressions".to_string()))
            }
        }
    }
    
    /// Convert a Value to Python code
    fn value_to_python(&self, value: &Value) -> String {
        match value {
            Value::Int(n) => n.to_string(),
            Value::Float(f) => f.to_string(),
            Value::String(s) => format!("\"{}\"", s),
            Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
            Value::Name(n) => n.clone(),
            Value::BinOp { op, left, right } => {
                let op_str = match op {
                    crate::ir::BinOp::Add => "+",
                    crate::ir::BinOp::Sub => "-",
                    crate::ir::BinOp::Mul => "*",
                    crate::ir::BinOp::Div => "/",
                    crate::ir::BinOp::Lt => "<",
                    crate::ir::BinOp::Gt => ">",
                    crate::ir::BinOp::Le => "<=",
                    crate::ir::BinOp::Ge => ">=",
                    crate::ir::BinOp::Eq => "==",
                    crate::ir::BinOp::Ne => "!=",
                };
                format!("{} {} {}", self.value_to_python(left), op_str, self.value_to_python(right))
            }
            Value::Call { name, args, kwargs } => {
                let args_str = args.iter()
                    .map(|v| self.value_to_python(v))
                    .collect::<Vec<_>>()
                    .join(", ");
                
                let kwargs_str = if kwargs.is_empty() {
                    String::new()
                } else {
                    let kw: Vec<String> = kwargs.iter()
                        .map(|(k, v)| format!("{}={}", k, self.value_to_python(v)))
                        .collect();
                    if args.is_empty() {
                        kw.join(", ")
                    } else {
                        format!(", {}", kw.join(", "))
                    }
                };
                
                format!("{}({}{})", name, args_str, kwargs_str)
            }
        }
    }
    
    /// Convert CamelCase to snake_case
    fn snake_case(&self, name: &str) -> String {
        let mut result = String::new();
        let mut chars = name.chars().peekable();
        
        while let Some(c) = chars.next() {
            if c.is_uppercase() {
                if !result.is_empty() {
                    result.push('_');
                }
                result.push(c.to_lowercase().next().unwrap());
            } else {
                result.push(c);
            }
        }
        
        result
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

/// Generate PyTorch code for a specific neuron
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
    fn test_snake_case() {
        let gen = CodeGenerator::new(&Program::new());
        assert_eq!(gen.snake_case("Linear"), "linear");
        assert_eq!(gen.snake_case("GELU"), "g_e_l_u");
        assert_eq!(gen.snake_case("LayerNorm"), "layer_norm");
        assert_eq!(gen.snake_case("MultiHeadAttention"), "multi_head_attention");
    }
    
    #[test]
    fn test_value_to_python() {
        let gen = CodeGenerator::new(&Program::new());
        assert_eq!(gen.value_to_python(&Value::Int(42)), "42");
        assert_eq!(gen.value_to_python(&Value::Float(3.14)), "3.14");
        assert_eq!(gen.value_to_python(&Value::String("hello".to_string())), "\"hello\"");
        assert_eq!(gen.value_to_python(&Value::Bool(true)), "True");
        assert_eq!(gen.value_to_python(&Value::Bool(false)), "False");
        assert_eq!(gen.value_to_python(&Value::Name("dim".to_string())), "dim");
    }
}
