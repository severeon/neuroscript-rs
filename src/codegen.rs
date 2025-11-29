use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use crate::interfaces::*;

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
    fn new(program: &'a Program) -> Self {
        Self {
            program,
            registry: StdlibRegistry::new(),
            node_counter: 0,
            used_primitives: HashSet::new(),
            var_names: HashMap::new(),
            call_to_module: HashMap::new(),
            current_neuron_params: HashSet::new(),
            binding_context: HashMap::new(),
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
    
    /// Check if a Value contains references to captured dimensions (not parameters)
    fn has_captured_dimensions(&self, value: &Value) -> bool {
        match value {
            Value::Name(n) => !self.current_neuron_params.contains(n),
            Value::BinOp { left, right, .. } => {
                self.has_captured_dimensions(left) || self.has_captured_dimensions(right)
            }
            Value::Call { args, kwargs, .. } => {
                args.iter().any(|v| self.has_captured_dimensions(v)) ||
                kwargs.iter().any(|(_, v)| self.has_captured_dimensions(v))
            }
            _ => false,
        }
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
        let mut all_endpoints = Vec::new();
        self.collect_calls(connections, &mut all_endpoints);

        for endpoint in &all_endpoints {
            if let Endpoint::Call { name, args, kwargs, .. } = endpoint {
                let key = self.endpoint_key(&endpoint);
                if !seen_calls.contains_key(&key) {
                    let id = self.next_node_id();
                    let module_name = format!("{}_{}", self.snake_case(&name), id);
                    // Store the mapping for use in forward generation
                    self.call_to_module.insert(key.clone(), module_name.clone());
                    seen_calls.insert(key, (name.clone(), module_name, args.clone(), kwargs.clone()));
                }
            }
        }

        // Generate instantiations in deterministic order
        let mut calls: Vec<_> = seen_calls.into_iter().collect();
        calls.sort_by(|a, b| a.1.1.cmp(&b.1.1)); // Sort by module_name for determinism

        let mut instantiated_count = 0;
        for (_key, (name, module_name, args, kwargs)) in &calls {
            // Check if any arguments contain captured dimensions
            let has_captured = args.iter().any(|v| self.has_captured_dimensions(v)) ||
                               kwargs.iter().any(|(_, v)| self.has_captured_dimensions(v));

            if has_captured {
                // Skip instantiation in __init__ for modules with captured dimensions
                // They will be instantiated lazily in forward()
                // Initialize cache variable to None
                writeln!(output, "        self._{} = None  # Lazy instantiation (has captured dimensions)", module_name).unwrap();
                instantiated_count += 1;
                continue;
            }

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
            instantiated_count += 1;
        }

        // If no modules were instantiated, add pass
        if instantiated_count == 0 {
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
        let indent = "        ";

        // Build a map from Call endpoints to their result variable names
        let mut call_to_result: HashMap<String, String> = HashMap::new();

        // Track the last result variable (for implicit output)
        let mut last_result = None;

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
                Endpoint::Call { name, .. } => {
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
                    return Err(CodegenError::UnsupportedFeature("Match expressions as source".to_string()));
                }
            };
            
            // Process the destination
            let result_var = self.process_destination(output, &conn.destination, source_var, indent, &mut temp_var_counter, &mut call_to_result)?;

            // Track the last result for implicit output
            last_result = Some(result_var.clone());

            // If destination was a Call, store result in call_to_result
            if let Endpoint::Call { .. } = &conn.destination {
                 let key = self.endpoint_key(&conn.destination);
                 call_to_result.insert(key, result_var);
            }
        }

        // Return the output variable
        // Priority: explicit "out" port > last result > last temp variable
        let output_var = self.var_names.get("out")
            .cloned()
            .or(last_result)
            .unwrap_or_else(|| format!("x{}", temp_var_counter - 1));
        writeln!(output, "        return {}", output_var).unwrap();
        
        Ok(())
    }

    /// Process a destination endpoint, generating code and returning the result variable name
    fn process_destination(
        &mut self, 
        output: &mut String, 
        endpoint: &Endpoint, 
        source_var: String, 
        indent: &str, 
        temp_var_counter: &mut usize,
        call_to_result: &mut HashMap<String, String>
    ) -> Result<String, CodegenError> {
        match endpoint {
            Endpoint::Ref(port_ref) => {
                // Simple assignment - the source becomes this port's variable
                self.var_names.insert(port_ref.node.clone(), source_var.clone());
                Ok(source_var)
            }
            Endpoint::Tuple(refs) => {
                // Tuple unpacking
                let var_names: Vec<String> = refs.iter().map(|r| {
                    let v = format!("x{}", *temp_var_counter);
                    *temp_var_counter += 1;
                    self.var_names.insert(r.node.clone(), v.clone());
                    v
                }).collect();
                
                writeln!(output, "{}{} = {}", indent, var_names.join(", "), source_var).unwrap();
                Ok(source_var) // Return tuple as result
            }
            Endpoint::Call { name, args, kwargs, .. } => {
                // Generate a call to the module
                let key = self.endpoint_key(endpoint);
                let module_name = self.call_to_module.get(&key)
                    .cloned()
                    .ok_or_else(|| CodegenError::InvalidConnection(
                        format!("Module for call to {} not found", name)
                    ))?;

                let result_var = format!("x{}", *temp_var_counter);
                *temp_var_counter += 1;

                // Check if this call has captured dimensions (needs lazy instantiation)
                let has_captured = args.iter().any(|v| self.has_captured_dimensions(v)) ||
                                   kwargs.iter().any(|(_, v)| self.has_captured_dimensions(v));

                if has_captured {
                    // Lazy instantiation: check cache, instantiate if needed
                    writeln!(output, "{}if self._{} is None:", indent, module_name).unwrap();

                    // Generate instantiation with current dimension values
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

                    // Mark as primitive for imports
                    if let Some(neuron) = self.program.neurons.get(name.as_str()) {
                        if neuron.is_primitive() {
                            self.used_primitives.insert(name.clone());
                        }
                    } else {
                        self.used_primitives.insert(name.clone());
                    }

                    writeln!(output, "{}    self._{} = {}({}{})", indent, module_name, name, args_str, kwargs_str).unwrap();

                    // Call the lazily-instantiated module
                    writeln!(output, "{}{} = self._{}({})", indent, result_var, module_name, source_var).unwrap();
                } else {
                    // Normal call to pre-instantiated module
                    writeln!(output, "{}{} = self.{}({})", indent, result_var, module_name, source_var).unwrap();
                }

                Ok(result_var)
            }
            Endpoint::Match(match_expr) => {
                let result_var = format!("x{}", *temp_var_counter);
                *temp_var_counter += 1;

                // Initialize result_var to None for safety (though not strictly needed if all paths return)
                writeln!(output, "{}{} = None", indent, result_var).unwrap();

                let mut first = true;
                let mut prev_condition = String::new();
                for arm in &match_expr.arms {
                    let shape_check = self.generate_shape_check(&arm.pattern, arm.guard.as_ref(), &source_var);
                    
                    // Determine prefix: use "else:" if pattern condition is same as previous
                    let prefix = if first {
                        "if"
                    } else if shape_check.condition == prev_condition {
                        // Same pattern, different guard (or no guard) -> use else
                        "else"
                    } else {
                        "elif"
                    };
                    first = false;

                    // Only output condition if it's not "else"
                    if prefix == "else" {
                        writeln!(output, "{}{}:", indent, prefix).unwrap();
                    } else {
                        writeln!(output, "{}{} {}:", indent, prefix, shape_check.condition).unwrap();
                        prev_condition = shape_check.condition.clone();
                    }

                    // Process pipeline - save var_names to avoid pollution from match arm scope
                    let saved_var_names = self.var_names.clone();
                    let arm_indent = format!("{}    ", indent);

                    // Emit dimension bindings before processing pipeline
                    for binding in &shape_check.bindings {
                        writeln!(output, "{}{}", arm_indent, binding).unwrap();
                    }

                    // If guard uses captured dimensions, check it after binding
                    let pipeline_indent = if let Some(guard_cond) = &shape_check.guard_condition {
                        writeln!(output, "{}if {}:", arm_indent, guard_cond).unwrap();
                        format!("{}    ", arm_indent)
                    } else {
                        arm_indent.clone()
                    };

                    let mut current_var = source_var.clone();

                    for ep in &arm.pipeline {
                         current_var = self.process_destination(output, ep, current_var, &pipeline_indent, temp_var_counter, call_to_result)?;

                         // If endpoint was a Call, store result in call_to_result
                        if let Endpoint::Call { .. } = ep {
                             let key = self.endpoint_key(ep);
                             call_to_result.insert(key, current_var.clone());
                        }
                    }

                    writeln!(output, "{}{} = {}", pipeline_indent, result_var, current_var).unwrap();

                    // Restore var_names to prevent match arm scope from leaking
                    self.var_names = saved_var_names;
                }

                // Else clause
                writeln!(output, "{}else:", indent).unwrap();
                writeln!(output, "{}    raise ValueError(f'No match found for shape {{ {}.shape }}')", indent, source_var).unwrap();

                Ok(result_var)
            }
        }
    }

    /// Generate a runtime shape check condition and dimension bindings
    ///
    /// Returns a ShapeCheckResult containing:
    /// - condition: Boolean expression for runtime check (shape checks only, no guard)
    /// - bindings: Dimension variable assignments (e.g., "d = x.shape[1]")
    /// - guard_condition: Separate guard check if it references captured dimensions
    ///
    /// Named dimensions in patterns are handled as follows:
    /// - If the name is a neuron parameter: Check equality (no binding needed)
    /// - Otherwise: Capture the dimension value for use in the pipeline
    fn generate_shape_check(&self, pattern: &Shape, guard: Option<&Value>, var_name: &str) -> ShapeCheckResult {
        let mut checks = Vec::new();
        let mut bindings = Vec::new();

        // Rank check (unless variadic)
        let has_variadic = pattern.dims.iter().any(|d| matches!(d, Dim::Variadic(_)));
        if !has_variadic {
            checks.push(format!("{}.ndim == {}", var_name, pattern.dims.len()));
        }

        for (i, dim) in pattern.dims.iter().enumerate() {
            match dim {
                Dim::Literal(n) => {
                    checks.push(format!("{}.shape[{}] == {}", var_name, i, n));
                }
                Dim::Named(n) => {
                    // Check if it's a parameter
                    if self.current_neuron_params.contains(n) {
                        // Parameter: check equality with self.param
                        checks.push(format!("{}.shape[{}] == self.{}", var_name, i, n));
                    } else {
                        // Pattern capture: bind dimension for use in pipeline
                        bindings.push(format!("{} = {}.shape[{}]", n, var_name, i));
                    }
                }
                _ => {} // Skip Wildcard, Variadic, Expr for now
            }
        }

        // Handle guard: if it references captured dimensions, defer to after binding
        let guard_condition = if let Some(guard_expr) = guard {
            if !bindings.is_empty() && self.has_captured_dimensions(guard_expr) {
                // Guard uses captured dims - check it separately after binding
                Some(self.value_to_python_with_self(guard_expr))
            } else {
                // Guard doesn't use captured dims - include in main condition
                let guard_str = self.value_to_python_with_self(guard_expr);
                checks.push(format!("({})", guard_str));
                None
            }
        } else {
            None
        };

        let condition = if checks.is_empty() {
            "True".to_string()
        } else {
            checks.join(" and ")
        };

        ShapeCheckResult { condition, bindings, guard_condition }
    }

    /// Convert a Value to Python, replacing parameter names with self.param
    fn value_to_python_with_self(&self, value: &Value) -> String {
        match value {
            Value::Name(n) => {
                // If it's a parameter, reference it as self.param
                if self.current_neuron_params.contains(n) {
                    format!("self.{}", n)
                } else {
                    n.clone()
                }
            }
            Value::BinOp { op, left, right } => {
                let op_str = match op {
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
                };
                format!("{} {} {}", self.value_to_python_with_self(left), op_str, self.value_to_python_with_self(right))
            }
            _ => self.value_to_python(value)
        }
    }

    /// Collect all Call endpoints recursively, including from Match expressions
    fn collect_calls(&self, connections: &[Connection], calls: &mut Vec<Endpoint>) {
        for conn in connections {
            self.collect_calls_from_endpoint(&conn.source, calls);
            self.collect_calls_from_endpoint(&conn.destination, calls);
        }
    }

    fn collect_calls_from_endpoint(&self, endpoint: &Endpoint, calls: &mut Vec<Endpoint>) {
        match endpoint {
            Endpoint::Call { .. } => calls.push(endpoint.clone()),
            Endpoint::Match(match_expr) => {
                for arm in &match_expr.arms {
                    for ep in &arm.pipeline {
                        self.collect_calls_from_endpoint(ep, calls);
                    }
                }
            }
            Endpoint::Tuple(_refs) => {
                // Tuple unpacking doesn't contain calls in current IR
            }
            Endpoint::Ref(_) => {}
        }
    }
    
    /// Generate a unique key for an endpoint (for tracking Call results)
    fn endpoint_key(&self, endpoint: &Endpoint) -> String {
        match endpoint {
            Endpoint::Call { name, args, kwargs, .. } => {
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
            Endpoint::Call { name, args: _, kwargs: _, .. } => {
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
        let program = Program::new();
        let gen = CodeGenerator::new(&program);
        assert_eq!(gen.snake_case("Linear"), "linear");
        assert_eq!(gen.snake_case("GELU"), "g_e_l_u");
        assert_eq!(gen.snake_case("LayerNorm"), "layer_norm");
        assert_eq!(gen.snake_case("MultiHeadAttention"), "multi_head_attention");
    }
    
    #[test]
    fn test_value_to_python() {
        let program = Program::new();
        let gen = CodeGenerator::new(&program);
        assert_eq!(gen.value_to_python(&Value::Int(42)), "42");
        assert_eq!(gen.value_to_python(&Value::Float(3.14)), "3.14");
        assert_eq!(gen.value_to_python(&Value::String("hello".to_string())), "\"hello\"");
        assert_eq!(gen.value_to_python(&Value::Bool(true)), "True");
        assert_eq!(gen.value_to_python(&Value::Bool(false)), "False");
        assert_eq!(gen.value_to_python(&Value::Name("dim".to_string())), "dim");
    }

    #[test]
    fn test_codegen_match() {
        use crate::ir::*;
        
        // Construct a simple program with a match expression
        let mut program = Program::new();
        let neuron = NeuronDef {
            name: "MatchTest".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Named("dim".to_string())]) }],
            outputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Named("dim".to_string())]) }],
            body: NeuronBody::Graph(vec![
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Match(MatchExpr {
                        arms: vec![
                            MatchArm {
                                pattern: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]),
                                guard: None,
                                pipeline: vec![
                                    Endpoint::Call { name: "Identity".to_string(), args: vec![], kwargs: vec![], id: 0 },
                                    Endpoint::Ref(PortRef::new("out"))
                                ]
                            },
                            MatchArm {
                                pattern: Shape::new(vec![Dim::Wildcard, Dim::Literal(256)]),
                                guard: None,
                                pipeline: vec![
                                    Endpoint::Call { name: "Linear".to_string(), args: vec![Value::Int(256), Value::Int(512)], kwargs: vec![], id: 1 },
                                    Endpoint::Ref(PortRef::new("out"))
                                ]
                            }
                        ]
                    })
                }
            ])
        };
        
        program.neurons.insert("MatchTest".to_string(), neuron);
        
        let code = generate_pytorch(&program, "MatchTest").unwrap();
        println!("{}", code);
        
        assert!(code.contains("if x.ndim == 2 and x.shape[1] == 512:"));
        assert!(code.contains("elif x.ndim == 2 and x.shape[1] == 256:"));
        // Note: IDs might vary depending on counter, but names should be consistent
        assert!(code.contains("self.identity_"));
        assert!(code.contains("self.linear_"));
    }

    #[test]
    fn test_codegen_match_with_captured_dims() {
        use crate::ir::*;

        // Test match expression with captured dimensions
        let mut program = Program::new();
        let neuron = NeuronDef {
            name: "DynamicMatch".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Wildcard]) }],
            outputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]) }],
            body: NeuronBody::Graph(vec![
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Match(MatchExpr {
                        arms: vec![
                            MatchArm {
                                pattern: Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]),
                                guard: Some(Value::BinOp {
                                    op: BinOp::Gt,
                                    left: Box::new(Value::Name("d".to_string())),
                                    right: Box::new(Value::Int(512))
                                }),
                                pipeline: vec![
                                    Endpoint::Call {
                                        name: "Linear".to_string(),
                                        args: vec![Value::Name("d".to_string()), Value::Int(512)],
                                        kwargs: vec![],
                                        id: 0
                                    },
                                    Endpoint::Ref(PortRef::new("out"))
                                ]
                            },
                            MatchArm {
                                pattern: Shape::new(vec![Dim::Wildcard, Dim::Named("d".to_string())]),
                                guard: None,
                                pipeline: vec![
                                    Endpoint::Call {
                                        name: "Linear".to_string(),
                                        args: vec![Value::Name("d".to_string()), Value::Int(256)],
                                        kwargs: vec![],
                                        id: 1
                                    },
                                    Endpoint::Call {
                                        name: "Linear".to_string(),
                                        args: vec![Value::Int(256), Value::Int(512)],
                                        kwargs: vec![],
                                        id: 2
                                    },
                                    Endpoint::Ref(PortRef::new("out"))
                                ]
                            }
                        ]
                    })
                }
            ])
        };

        program.neurons.insert("DynamicMatch".to_string(), neuron);

        let code = generate_pytorch(&program, "DynamicMatch").unwrap();
        println!("{}", code);

        // Verify dimension binding is generated
        assert!(code.contains("d = x.shape[1]"), "Dimension binding should be generated");

        // Verify guard condition includes the bound dimension (on separate line after binding)
        assert!(code.contains("if d > 512:"), "Guard should reference bound dimension");

        // Verify lazy instantiation for modules with captured dimensions
        assert!(code.contains("self._linear_") && code.contains("= None"), "Should have lazy instantiation");
        assert!(code.contains("if self._linear_") && code.contains("is None:"), "Should check for lazy instantiation");
        assert!(code.contains("Linear(d,"), "Should instantiate Linear with captured dimension");
    }

    #[test]
    fn test_codegen_match_guards_with_bindings() {
        use crate::ir::*;

        // Test guard expression that references captured dimension
        let mut program = Program::new();
        let neuron = NeuronDef {
            name: "GuardTest".to_string(),
            params: vec![],
            inputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Wildcard]) }],
            outputs: vec![Port { name: "default".to_string(), shape: Shape::new(vec![Dim::Wildcard, Dim::Literal(512)]) }],
            body: NeuronBody::Graph(vec![
                Connection {
                    source: Endpoint::Ref(PortRef::new("in")),
                    destination: Endpoint::Match(MatchExpr {
                        arms: vec![
                            MatchArm {
                                pattern: Shape::new(vec![Dim::Wildcard, Dim::Named("dim".to_string())]),
                                guard: Some(Value::BinOp {
                                    op: BinOp::Le,
                                    left: Box::new(Value::Name("dim".to_string())),
                                    right: Box::new(Value::Int(512))
                                }),
                                pipeline: vec![
                                    Endpoint::Call { name: "Identity".to_string(), args: vec![], kwargs: vec![], id: 0 },
                                    Endpoint::Ref(PortRef::new("out"))
                                ]
                            }
                        ]
                    })
                }
            ])
        };

        program.neurons.insert("GuardTest".to_string(), neuron);

        let code = generate_pytorch(&program, "GuardTest").unwrap();
        println!("{}", code);

        // Verify dimension is bound before being used in guard
        assert!(code.contains("dim = x.shape[1]"), "Dimension must be bound");
        assert!(code.contains("if dim <= 512:"), "Guard should use bound dimension");
    }
}
