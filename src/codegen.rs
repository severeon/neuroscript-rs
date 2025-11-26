//! PyTorch Code Generator
//!
//! Transforms validated NeuroScript IR into PyTorch Python modules.
//! MVP scope: Only primitives and simple composites, no shape inference.

use crate::ir::*;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub enum CodegenError {
    UnknownPrimitive { impl_ref: String, neuron: String },
    UnsupportedFeature { feature: String, context: String },
    InvalidGraph { message: String },
}

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodegenError::UnknownPrimitive { impl_ref, neuron } => {
                write!(f, "Unknown primitive '{}' in neuron '{}'", impl_ref, neuron)
            }
            CodegenError::UnsupportedFeature { feature, context } => {
                write!(f, "Unsupported feature '{}' in {}", feature, context)
            }
            CodegenError::InvalidGraph { message } => {
                write!(f, "Invalid graph: {}", message)
            }
        }
    }
}

impl std::error::Error for CodegenError {}

pub struct PyTorchCodegen {
    primitives: PrimitiveRegistry,
}

impl PyTorchCodegen {
    pub fn new() -> Self {
        Self {
            primitives: PrimitiveRegistry::new(),
        }
    }

    /// Generate complete Python module from a validated program
    pub fn generate(&self, program: &Program) -> Result<String, CodegenError> {
        let mut output = String::new();

        // 1. Imports
        output.push_str(&self.generate_imports());
        output.push_str("\n\n");

        // 2. Generate each neuron as a class
        for (name, neuron) in &program.neurons {
            output.push_str(&self.generate_neuron(name, neuron)?);
            output.push_str("\n\n");
        }

        // 3. Add test code at the end
        output.push_str(&self.generate_test_code(program));

        Ok(output)
    }

    fn generate_imports(&self) -> String {
        "import torch\nimport torch.nn as nn\nfrom typing import Tuple".to_string()
    }

    fn generate_neuron(&self, name: &str, neuron: &NeuronDef) -> Result<String, CodegenError> {
        let mut code = String::new();

        // Class definition
        code.push_str(&format!("class {}(nn.Module):\n", name));

        // __init__ method
        code.push_str(&self.generate_init(name, neuron)?);

        // forward method
        code.push_str(&self.generate_forward(name, neuron)?);

        Ok(code)
    }

    fn generate_init(&self, name: &str, neuron: &NeuronDef) -> Result<String, CodegenError> {
        let mut code = String::new();

        // Method signature
        let params = neuron.params.iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");

        if params.is_empty() {
            code.push_str("    def __init__(self):\n");
        } else {
            code.push_str(&format!("    def __init__(self, {}):\n", params));
        }

        code.push_str("        super().__init__()\n");

        match &neuron.body {
            NeuronBody::Primitive(impl_ref) => {
                // Generate primitive initialization
                let primitive_code = self.primitives.generate_init(impl_ref, neuron)?;
                code.push_str(&format!("        {}\n", primitive_code));
            }
            NeuronBody::Graph(connections) => {
                // Generate composite initialization - instantiate sub-modules
                code.push_str(&self.generate_composite_init(neuron, connections)?);
            }
        }

        Ok(code)
    }

    fn generate_forward(&self, _name: &str, neuron: &NeuronDef) -> Result<String, CodegenError> {
        let mut code = String::new();

        // Determine input signature
        let input_sig = self.generate_input_signature(&neuron.inputs);
        code.push_str(&format!("    def forward(self, {}):\n", input_sig));

        match &neuron.body {
            NeuronBody::Primitive(impl_ref) => {
                let forward_code = self.primitives.generate_forward(impl_ref, neuron)?;
                code.push_str(&format!("        {}\n", forward_code));
            }
            NeuronBody::Graph(connections) => {
                code.push_str(&self.generate_composite_forward(neuron, connections)?);
            }
        }

        Ok(code)
    }

    fn generate_input_signature(&self, inputs: &[Port]) -> String {
        if inputs.is_empty() || (inputs.len() == 1 && inputs[0].name == "default") {
            "x".to_string()
        } else {
            // Multiple named inputs
            inputs.iter()
                .map(|p| if p.name == "default" { "x" } else { &p.name })
                .collect::<Vec<_>>()
                .join(", ")
        }
    }

    fn generate_composite_init(&self, _neuron: &NeuronDef, connections: &[Connection]) -> Result<String, CodegenError> {
        let mut code = String::new();
        let mut submodules = HashSet::new();

        // Extract all Call nodes from connections
        for conn in connections {
            self.collect_calls(&conn.source, &mut submodules);
            self.collect_calls(&conn.destination, &mut submodules);
        }

        // Generate self.name = ClassName(args) for each submodule
        for (idx, (module_name, args)) in submodules.iter().enumerate() {
            let var_name = format!("{}_{}", module_name.to_lowercase(), idx);
            code.push_str(&format!("        self.{} = {}({})\n", var_name, module_name, args));
        }

        Ok(code)
    }

    fn collect_calls(&self, endpoint: &Endpoint, submodules: &mut HashSet<(String, String)>) {
        match endpoint {
            Endpoint::Call { name, args, .. } => {
                let args_str = args.iter()
                    .map(|a| self.value_to_python(a))
                    .collect::<Vec<_>>()
                    .join(", ");
                submodules.insert((name.clone(), args_str));
            }
            Endpoint::Tuple(_port_refs) => {
                // Tuples don't contain calls, just port refs
                // Nothing to collect here
            }
            _ => {}
        }
    }

    fn generate_composite_forward(&self, _neuron: &NeuronDef, connections: &[Connection]) -> Result<String, CodegenError> {
        let mut code = String::new();

        // Track values at each named point
        let mut values: HashMap<String, String> = HashMap::new();
        values.insert("in".to_string(), "x".to_string());

        // Process each connection
        for conn in connections {
            // Get source value
            let src_val = match &conn.source {
                Endpoint::Ref(pr) => values.get(&pr.node).cloned().unwrap_or("x".to_string()),
                Endpoint::Tuple(prs) => {
                    let vals: Vec<String> = prs.iter()
                        .map(|pr| values.get(&pr.node).cloned().unwrap_or(pr.node.clone()))
                        .collect();
                    format!("({})", vals.join(", "))
                }
                Endpoint::Call { name, args, .. } => {
                    let args_str = args.iter()
                        .map(|a| self.value_to_python(a))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let input = values.values().last().cloned().unwrap_or("x".to_string());
                    let var = format!("_{}", name.to_lowercase());
                    code.push_str(&format!("        {} = self.{}({})\n", var, name.to_lowercase(), input));
                    var
                }
                _ => "x".to_string(),
            };

            // Process destination
            match &conn.destination {
                Endpoint::Ref(pr) => {
                    values.insert(pr.node.clone(), src_val);
                }
                Endpoint::Tuple(prs) => {
                    let var_names: Vec<String> = prs.iter().map(|pr| pr.node.clone()).collect();
                    code.push_str(&format!("        {} = {}\n", var_names.join(", "), src_val));
                    for pr in prs {
                        values.insert(pr.node.clone(), pr.node.clone());
                    }
                }
                Endpoint::Call { name, args, .. } => {
                    let args_str = args.iter()
                        .map(|a| self.value_to_python(a))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let var = format!("_{}", name.to_lowercase());
                    code.push_str(&format!("        {} = self.{}({})\n", var, name.to_lowercase(), src_val));
                    values.insert(name.clone(), var);
                }
                _ => {}
            }
        }

        // Return the output value
        let return_val = values.get("out").cloned().unwrap_or("x".to_string());
        code.push_str(&format!("        return {}\n", return_val));

        Ok(code)
    }

    fn value_to_python(&self, value: &Value) -> String {
        match value {
            Value::Int(n) => n.to_string(),
            Value::Float(f) => f.to_string(),
            Value::String(s) => format!("\"{}\"", s),
            Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
            Value::Name(n) => n.clone(),
            Value::BinOp { op, left, right } => {
                format!("{} {} {}", self.value_to_python(left),
                       self.binop_to_python(op), self.value_to_python(right))
            }
            Value::Call { name, args, kwargs } => {
                let args_str = args.iter()
                    .map(|a| self.value_to_python(a))
                    .collect::<Vec<_>>()
                    .join(", ");
                let kwargs_str = kwargs.iter()
                    .map(|(k, v)| format!("{}={}", k, self.value_to_python(v)))
                    .collect::<Vec<_>>()
                    .join(", ");

                if kwargs_str.is_empty() {
                    format!("{}({})", name, args_str)
                } else if args_str.is_empty() {
                    format!("{}({})", name, kwargs_str)
                } else {
                    format!("{}({}, {})", name, args_str, kwargs_str)
                }
            }
        }
    }

    fn binop_to_python(&self, op: &BinOp) -> &str {
        match op {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Le => "<=",
            BinOp::Ge => ">=",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
        }
    }

    fn generate_test_code(&self, program: &Program) -> String {
        let mut code = String::new();

        code.push_str("if __name__ == '__main__':\n");
        code.push_str("    print('Generated PyTorch modules:')\n");

        for name in program.neurons.keys() {
            code.push_str(&format!("    print('  - {}')\n", name));
        }

        code.push_str("    print('\\n✅ Code generation successful!')\n");

        code
    }
}

/// Registry of primitive implementations
struct PrimitiveRegistry {
    mappings: HashMap<String, PrimitiveImpl>,
}

struct PrimitiveImpl {
    pytorch_class: String,
    init_template: fn(&NeuronDef) -> String,
    forward_template: fn(&NeuronDef) -> String,
}

impl PrimitiveRegistry {
    fn new() -> Self {
        let mut mappings = HashMap::new();

        // core,nn/Linear
        mappings.insert(
            "core,nn/Linear".to_string(),
            PrimitiveImpl {
                pytorch_class: "nn.Linear".to_string(),
                init_template: |neuron| {
                    // Linear(in_dim, out_dim)
                    if neuron.params.len() >= 2 {
                        format!("self.layer = nn.Linear({}, {})",
                            neuron.params[0].name, neuron.params[1].name)
                    } else {
                        "self.layer = nn.Linear(1, 1)  # FIXME: missing params".to_string()
                    }
                },
                forward_template: |_| "return self.layer(x)".to_string(),
            },
        );

        // core,activations/GELU
        mappings.insert(
            "core,activations/GELU".to_string(),
            PrimitiveImpl {
                pytorch_class: "nn.GELU".to_string(),
                init_template: |_| "self.activation = nn.GELU()".to_string(),
                forward_template: |_| "return self.activation(x)".to_string(),
            },
        );

        // core,builtin/Fork
        mappings.insert(
            "core,builtin/Fork".to_string(),
            PrimitiveImpl {
                pytorch_class: "Fork".to_string(),
                init_template: |_| "pass  # Fork is stateless".to_string(),
                forward_template: |_| "return (x, x)".to_string(),
            },
        );

        // core,builtin/Add
        mappings.insert(
            "core,builtin/Add".to_string(),
            PrimitiveImpl {
                pytorch_class: "Add".to_string(),
                init_template: |_| "pass  # Add is stateless".to_string(),
                forward_template: |neuron| {
                    // Add has two inputs: left and right
                    if neuron.inputs.len() == 2 {
                        "return torch.add(left, right)".to_string()
                    } else {
                        "return x  # FIXME: Add needs two inputs".to_string()
                    }
                },
            },
        );

        Self { mappings }
    }

    fn generate_init(&self, impl_ref: &ImplRef, neuron: &NeuronDef) -> Result<String, CodegenError> {
        match impl_ref {
            ImplRef::Source { source, path } => {
                let key = format!("{},{}", source, path);
                if let Some(prim) = self.mappings.get(&key) {
                    Ok((prim.init_template)(neuron))
                } else {
                    Err(CodegenError::UnknownPrimitive {
                        impl_ref: key,
                        neuron: neuron.name.clone(),
                    })
                }
            }
            ImplRef::External { .. } => {
                Err(CodegenError::UnsupportedFeature {
                    feature: "external neurons".to_string(),
                    context: "primitive init".to_string(),
                })
            }
        }
    }

    fn generate_forward(&self, impl_ref: &ImplRef, neuron: &NeuronDef) -> Result<String, CodegenError> {
        match impl_ref {
            ImplRef::Source { source, path } => {
                let key = format!("{},{}", source, path);
                if let Some(prim) = self.mappings.get(&key) {
                    Ok((prim.forward_template)(neuron))
                } else {
                    Err(CodegenError::UnknownPrimitive {
                        impl_ref: key,
                        neuron: neuron.name.clone(),
                    })
                }
            }
            ImplRef::External { .. } => {
                Err(CodegenError::UnsupportedFeature {
                    feature: "external neurons".to_string(),
                    context: "primitive forward".to_string(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codegen_basic() {
        let codegen = PyTorchCodegen::new();

        // Create a simple Linear neuron
        let mut neurons = HashMap::new();
        neurons.insert(
            "Linear".to_string(),
            NeuronDef {
                name: "Linear".to_string(),
                params: vec![
                    Param { name: "in_dim".to_string(), default: None },
                    Param { name: "out_dim".to_string(), default: None },
                ],
                inputs: vec![Port {
                    name: "default".to_string(),
                    shape: Shape::new(vec![]),
                }],
                outputs: vec![Port {
                    name: "default".to_string(),
                    shape: Shape::new(vec![]),
                }],
                body: NeuronBody::Primitive(ImplRef::Source {
                    source: "core".to_string(),
                    path: "nn/Linear".to_string(),
                }),
            },
        );

        let program = Program {
            uses: vec![],
            neurons,
        };

        let result = codegen.generate(&program);
        assert!(result.is_ok());

        let code = result.unwrap();
        assert!(code.contains("class Linear(nn.Module)"));
        assert!(code.contains("def __init__(self, in_dim, out_dim)"));
        assert!(code.contains("nn.Linear"));
    }
}
