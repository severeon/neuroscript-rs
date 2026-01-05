//! NeuroScript Intermediate Representation
//!
//! The IR is a graph of neuron instantiations with typed edges.
//! Every connection carries a shape contract.

use crate::interfaces::*;
use std::collections::HashMap;

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
            globals: vec![],
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
            Dim::Global(s) => write!(f, "@global {}", s),
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

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{:?}", n),
            Value::String(s) => write!(f, "`{}`", s),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Name(s) => write!(f, "{}", s),
            Value::Global(s) => write!(f, "@global {}", s),
            Value::BinOp { op, left, right } => write!(f, "({} {} {})", left, op, right),
            Value::Call { name, args, kwargs } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                for (i, (k, v)) in kwargs.iter().enumerate() {
                    if !args.is_empty() || i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}={}", k, v)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl std::fmt::Display for PortRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.port == "default" {
            write!(f, "{}", self.node)
        } else {
            write!(f, "{}.{}", self.node, self.port)
        }
    }
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Endpoint::Ref(p) => write!(f, "{}", p),
            Endpoint::Tuple(ps) => {
                write!(f, "(")?;
                for (i, p) in ps.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ")")
            }
            Endpoint::Call {
                name,
                args,
                kwargs,
                id: _,
                frozen: _,
            } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                for (i, (k, v)) in kwargs.iter().enumerate() {
                    if !args.is_empty() || i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}={}", k, v)?;
                }
                write!(f, ")")
            }
            Endpoint::Match(m) => write!(f, "{}", m),
        }
    }
}

impl std::fmt::Display for MatchExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "match:")?;
        for arm in &self.arms {
            write!(f, "    {} ", arm.pattern)?;
            if let Some(g) = &arm.guard {
                write!(f, "where {} ", g)?;
            }
            write!(f, ": ")?;
            for (i, e) in arm.pipeline.iter().enumerate() {
                if i > 0 {
                    write!(f, " -> ")?;
                }
                write!(f, "{}", e)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> {}", self.source, self.destination)
    }
}

impl std::fmt::Display for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scope::Instance { lazy: false } => Ok(()),
            Scope::Instance { lazy: true } => write!(f, "@lazy "),
            Scope::Static => write!(f, "@static "),
            Scope::Global => write!(f, "@global "),
        }
    }
}

impl std::fmt::Display for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{} = {}(", self.scope, self.name, self.call_name)?;
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", arg)?;
        }
        for (i, (k, v)) in self.kwargs.iter().enumerate() {
            if !self.args.is_empty() || i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}={}", k, v)?;
        }
        write!(f, ")")
    }
}

impl std::fmt::Display for GlobalBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "@global {} = {}", self.name, self.value)
    }
}

impl std::fmt::Display for NeuronDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "neuron {}(", self.name)?;
        for (i, p) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", p.name)?;
            if let Some(d) = &p.default {
                write!(f, "={}", d)?;
            }
        }
        writeln!(f, "):")?;

        match &self.body {
            NeuronBody::Primitive(imp) => {
                writeln!(f, "  impl: {:?}", imp)?;
            }
            NeuronBody::Graph {
                context_bindings,
                connections,
                ..
            } => {
                if !context_bindings.is_empty() {
                    writeln!(f, "  context:")?;
                    for b in context_bindings {
                        writeln!(f, "    {}", b)?;
                    }
                }
                writeln!(f, "  graph:")?;
                for c in connections {
                    writeln!(f, "    {}", c)?;
                }
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for u in &self.uses {
            writeln!(f, "use {}, {}/*", u.source, u.path.join("/"))?;
        }
        if !self.uses.is_empty() {
            writeln!(f)?;
        }

        for g in &self.globals {
            writeln!(f, "{}", g)?;
        }
        if !self.globals.is_empty() {
            writeln!(f)?;
        }

        for n in self.neurons.values() {
            writeln!(f, "{}", n)?;
        }
        Ok(())
    }
}
