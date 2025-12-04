//! NeuroScript Intermediate Representation
//!
//! The IR is a graph of neuron instantiations with typed edges.
//! Every connection carries a shape contract.

use std::collections::HashMap;
use crate::interfaces::*;

impl Shape {
    pub fn scalar() -> Self {
        Shape { dims: vec![] }
    }

    pub fn new(dims: Vec<Dim>) -> Self {
        Shape { dims }
    }
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

impl NeuronDef {
    pub fn is_primitive(&self) -> bool {
        matches!(self.body, NeuronBody::Primitive(_))
    }

    pub fn is_composite(&self) -> bool {
        matches!(self.body, NeuronBody::Graph { .. })
    }
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
