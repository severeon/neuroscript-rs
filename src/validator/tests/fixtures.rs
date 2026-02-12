use crate::interfaces::*;
use crate::validator::Validator;

/// Builder for creating Program instances with a fluent API
pub struct ProgramBuilder {
    program: Program,
}

impl ProgramBuilder {
    /// Create a new empty program
    pub fn new() -> Self {
        Self {
            program: Program::new(),
        }
    }

    /// Add a neuron definition to the program
    pub fn with_neuron(mut self, name: &str, def: NeuronDef) -> Self {
        self.program.neurons.insert(name.to_string(), def);
        self
    }

    /// Add a simple primitive neuron with single input and output
    pub fn with_simple_neuron(self, name: &str, in_shape: Shape, out_shape: Shape) -> Self {
        let def = NeuronDef {
            name: name.to_string(),
            params: vec![],
            inputs: vec![default_port(in_shape)],
            outputs: vec![default_port(out_shape)],
            body: NeuronBody::Primitive(ImplRef::Source {
                source: "test".to_string(),
                path: "test".to_string(),
            }),
            max_cycle_depth: None,
            doc: None,
        };
        self.with_neuron(name, def)
    }

    /// Add a multi-port primitive neuron
    pub fn with_multi_port_neuron(
        self,
        name: &str,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Self {
        let def = NeuronDef {
            name: name.to_string(),
            params: vec![],
            inputs,
            outputs,
            body: NeuronBody::Primitive(ImplRef::Source {
                source: "test".to_string(),
                path: "test".to_string(),
            }),
            max_cycle_depth: None,
            doc: None,
        };
        self.with_neuron(name, def)
    }

    /// Add a composite neuron with graph body
    pub fn with_composite(
        self,
        name: &str,
        connections: Vec<Connection>,
        max_cycle_depth: Option<usize>,
    ) -> Self {
        let def = NeuronDef {
            name: name.to_string(),
            params: vec![],
            inputs: vec![default_port(wildcard())],
            outputs: vec![default_port(wildcard())],
            body: NeuronBody::Graph {
                context_bindings: vec![],
                connections,
            },
            max_cycle_depth,
            doc: None,
        };
        self.with_neuron(name, def)
    }

    /// Add a composite neuron with custom ports
    pub fn with_composite_ports(
        self,
        name: &str,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
        connections: Vec<Connection>,
        max_cycle_depth: Option<usize>,
    ) -> Self {
        let def = NeuronDef {
            name: name.to_string(),
            params: vec![],
            inputs,
            outputs,
            body: NeuronBody::Graph {
                context_bindings: vec![],
                connections,
            },
            max_cycle_depth,
            doc: None,
        };
        self.with_neuron(name, def)
    }

    /// Build and return the final Program
    pub fn build(self) -> Program {
        self.program
    }
}

impl Default for ProgramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Assert that validation produces an error matching the predicate
pub fn assert_validation_error<F>(program: &mut Program, check: F)
where
    F: Fn(&ValidationError) -> bool,
{
    let result = Validator::validate(program);
    assert!(result.is_err(), "Expected validation error, but got Ok");
    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(check),
        "Expected error not found. Errors: {:?}",
        errors
    );
}

/// Assert that validation succeeds
pub fn assert_validation_ok(program: &mut Program) {
    let result = Validator::validate(program);
    assert!(result.is_ok(), "Expected validation to succeed, got: {:?}", result);
}

/// Create a port with the given name and shape
pub fn port(name: &str, shape: Shape) -> Port {
    Port {
        name: name.to_string(),
        shape,
        variadic: false,
    }
}

/// Create a variadic port with the given name and shape
pub fn variadic_port(name: &str, shape: Shape) -> Port {
    Port {
        name: name.to_string(),
        shape,
        variadic: true,
    }
}

/// Create a default port (named "default") with the given shape
pub fn default_port(shape: Shape) -> Port {
    Port {
        name: "default".to_string(),
        shape,
        variadic: false,
    }
}

// ========== Shape Helpers ==========

pub fn wildcard() -> Shape {
    Shape::new(vec![Dim::Wildcard])
}

pub fn shape_512() -> Shape {
    Shape::new(vec![Dim::Literal(512)])
}

pub fn shape_256() -> Shape {
    Shape::new(vec![Dim::Literal(256)])
}

pub fn shape_batch_512() -> Shape {
    Shape::new(vec![Dim::Wildcard, Dim::Literal(512)])
}

pub fn shape_batch_256() -> Shape {
    Shape::new(vec![Dim::Wildcard, Dim::Literal(256)])
}

pub fn shape_two_wildcard() -> Shape {
    Shape::new(vec![Dim::Wildcard, Dim::Wildcard])
}

pub fn named_dim(name: &str) -> Dim {
    Dim::Named(name.to_string())
}

pub fn variadic_dim(name: &str) -> Dim {
    Dim::Variadic(name.to_string())
}

// ========== Endpoint Helpers ==========

pub fn ref_endpoint(name: &str) -> Endpoint {
    Endpoint::Ref(PortRef::new(name))
}

pub fn call_endpoint(name: &str) -> Endpoint {
    Endpoint::Call {
        name: name.to_string(),
        args: vec![],
        kwargs: vec![],
        id: 0,
        frozen: false,
    }
}

pub fn tuple_endpoint(names: Vec<&str>) -> Endpoint {
    Endpoint::Tuple(names.iter().map(|n| PortRef::new(*n)).collect())
}

// ========== Connection Helpers ==========

pub fn connection(source: Endpoint, destination: Endpoint) -> Connection {
    Connection {
        source,
        destination,
    }
}
