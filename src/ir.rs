//! NeuroScript Intermediate Representation
//!
//! The IR is a graph of neuron instantiations with typed edges.
//! Every connection carries a shape contract.

use std::collections::HashMap;

/// A dimension in a tensor shape
#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    /// Literal value: 512
    Literal(i64),
    /// Named dimension: batch, seq, dim
    Named(String),
    /// Wildcard: * (matches any single dimension)
    Wildcard,
    /// Variadic: *batch (captures zero or more dimensions)
    Variadic(String),
    /// Computed: dim * 4
    Expr(Box<DimExpr>),
}

/// Binary operation on dimensions
#[derive(Debug, Clone, PartialEq)]
pub struct DimExpr {
    pub op: BinOp,
    pub left: Dim,
    pub right: Dim,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add, Sub, Mul, Div,
    Lt, Gt, Le, Ge, Eq, Ne,
}

/// A tensor shape: [batch, seq, dim] or [*, 512] or []
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    pub dims: Vec<Dim>,
}

impl Shape {
    pub fn scalar() -> Self {
        Shape { dims: vec![] }
    }

    pub fn new(dims: Vec<Dim>) -> Self {
        Shape { dims }
    }
}

/// An input or output port of a neuron
#[derive(Debug, Clone, PartialEq)]
pub struct Port {
    pub name: String, // "default" if unnamed
    pub shape: Shape,
}

/// A value in the language
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Name(String),
    BinOp {
        op: BinOp,
        left: Box<Value>,
        right: Box<Value>,
    },
    Call {
        name: String,
        args: Vec<Value>,
        kwargs: Vec<(String, Value)>,
    },
}

/// Reference to a port: in, out, fork.left
#[derive(Debug, Clone, PartialEq)]
pub struct PortRef {
    pub node: String,
    pub port: String, // "default" if not specified
}

impl PortRef {
    pub fn new(node: impl Into<String>) -> Self {
        PortRef {
            node: node.into(),
            port: "default".into(),
        }
    }

    pub fn with_port(node: impl Into<String>, port: impl Into<String>) -> Self {
        PortRef {
            node: node.into(),
            port: port.into(),
        }
    }
}

/// An endpoint in a connection
#[derive(Debug, Clone, PartialEq)]
pub enum Endpoint {
    /// Simple reference: in, out, my_neuron
    Ref(PortRef),
    /// Tuple of references: (a, b)
    Tuple(Vec<PortRef>),
    /// Neuron instantiation: Linear(512, 256)
    Call {
        name: String,
        args: Vec<Value>,
        kwargs: Vec<(String, Value)>,
        id: usize,
    },
    /// Pattern match expression
    Match(MatchExpr),
}

/// A connection: source -> destination
#[derive(Debug, Clone, PartialEq)]
pub struct Connection {
    pub source: Endpoint,
    pub destination: Endpoint,
}

/// One arm of a match expression
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Shape,
    pub guard: Option<Value>, // where clause
    pub pipeline: Vec<Endpoint>,
}

/// Pattern matching on shapes
#[derive(Debug, Clone, PartialEq)]
pub struct MatchExpr {
    pub arms: Vec<MatchArm>,
}

/// Reference to an implementation
#[derive(Debug, Clone, PartialEq)]
pub enum ImplRef {
    /// External source: core,nn/Linear
    Source { source: String, path: String },
    /// External API: external(`lmstudio`, model=`qwen`)
    External { kwargs: Vec<(String, Value)> },
}

/// The body of a neuron definition
#[derive(Debug, Clone, PartialEq)]
pub enum NeuronBody {
    /// Primitive: has implementation reference
    Primitive(ImplRef),
    /// Composite: defined by internal graph
    Graph(Vec<Connection>),
}

/// A parameter in a neuron definition
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub default: Option<Value>,
}

/// A complete neuron definition
#[derive(Debug, Clone, PartialEq)]
pub struct NeuronDef {
    pub name: String,
    pub params: Vec<Param>,
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
    pub body: NeuronBody,
}

impl NeuronDef {
    pub fn is_primitive(&self) -> bool {
        matches!(self.body, NeuronBody::Primitive(_))
    }

    pub fn is_composite(&self) -> bool {
        matches!(self.body, NeuronBody::Graph(_))
    }
}

/// An import statement: use core,nn/*
#[derive(Debug, Clone, PartialEq)]
pub struct UseStmt {
    pub source: String,
    pub path: Vec<String>,
}

/// A complete NeuroScript program
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub uses: Vec<UseStmt>,
    pub neurons: HashMap<String, NeuronDef>,
}

impl Program {
    pub fn new() -> Self {
        Program {
            uses: vec![],
            neurons: HashMap::new(),
        }
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

// === Display implementations for debugging ===

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dim::Literal(n) => write!(f, "{}", n),
            Dim::Named(s) => write!(f, "{}", s),
            Dim::Wildcard => write!(f, "*"),
            Dim::Variadic(s) => write!(f, "*{}", s),
            Dim::Expr(e) => write!(f, "({} {} {})", e.left, e.op, e.right),
        }
    }
}

impl std::fmt::Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
            BinOp::Le => write!(f, "<="),
            BinOp::Ge => write!(f, ">="),
            BinOp::Eq => write!(f, "=="),
            BinOp::Ne => write!(f, "!="),
        }
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}
