//! NeuroScript Intermediate Representation
//!
//! The IR is a graph of neuron instantiations with typed edges.
//! Every connection carries a shape contract.

use miette::Diagnostic;
use miette::SourceSpan;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// A dimension in a tensor shape
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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

impl Dim {
    /// Try to get the literal value if this is a Literal dimension
    pub fn as_literal(&self) -> Option<i64> {
        match self {
            Dim::Literal(n) => Some(*n),
            _ => None,
        }
    }

    /// Check if this dimension is compatible with another for broadcasting
    pub fn broadcastable_with(&self, other: &Dim) -> bool {
        match (self, other) {
            (Dim::Literal(a), Dim::Literal(b)) => *a == *b || *a == 1 || *b == 1,
            (Dim::Wildcard, _) | (_, Dim::Wildcard) => true,
            _ => false, // Named, Variadic, Expr can't be broadcast at runtime
        }
    }
}

/// Binary operation on dimensions
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DimExpr {
    pub op: BinOp,
    pub left: Dim,
    pub right: Dim,
}

/// Binary operation on dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
}

/// A tensor shape: [batch, seq, dim] or [*, 512] or []
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    pub dims: Vec<Dim>,
}

#[derive(Debug)]
pub enum CodegenError {
    NeuronNotFound(String),
    InvalidConnection(String),
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

    /// Dimension bindings from match pattern captures (e.g., "d" -> "x.shape[1]")
    /// Used to resolve dimension references in match arm pipelines
    pub binding_context: HashMap<String, String>,

    /// Lazy bindings from let: blocks (name -> (call_name, args, kwargs))
    pub lazy_bindings: HashMap<String, LazyBinding>,

    /// Shape inference context (resolved dimensions and node output shapes)
    /// Used for emitting shape assertions and documentation
    pub inference_ctx: InferenceContext,
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
        kwargs: Vec<Kwarg>,
    },
}

/// Reference to a port: in, out, fork.left
#[derive(Debug, Clone, PartialEq)]
pub struct PortRef {
    pub node: String,
    pub port: String, // "default" if not specified
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
        kwargs: Vec<Kwarg>,
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
    /// Whether this arm is reachable (not shadowed by earlier arms)
    /// Set to false by validator for dead code elimination
    pub is_reachable: bool,
}

/// Pattern matching on shapes
#[derive(Debug, Clone, PartialEq)]
pub struct MatchExpr {
    pub arms: Vec<MatchArm>,
}

// ImplRef is already defined above as a struct

/// Scope of a binding
#[derive(Debug, Clone, PartialEq)]
pub enum Scope {
    /// Instance-level (default), unique to each neuron instance
    Instance { lazy: bool },
    /// Static (class-level), shared across all instances of a neuron type
    Static,
    /// Global (module-level), shared across the entire module
    Global,
}

pub(crate) type Kwarg = (String, Value);
pub(crate) type CallArgs = (Vec<Value>, Vec<Kwarg>);
pub(crate) type CallExpr = (String, Vec<Value>, Vec<Kwarg>);
type LazyBinding = (String, Vec<Value>, Vec<Kwarg>);

/// A binding in a let: or set: block, or context: block
#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub name: String,
    pub call_name: String,
    pub args: Vec<Value>,
    pub kwargs: Vec<Kwarg>,
    pub scope: Scope, // Added scope
}

/// A module-level global definition: @global name = Value
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalBinding {
    pub name: String,
    pub value: Value,
}

/// The body of a neuron definition
#[derive(Debug, Clone, PartialEq)]
pub enum NeuronBody {
    /// Primitive: has implementation reference
    Primitive(ImplRef),
    /// Composite: defined by internal graph
    Graph {
        let_bindings: Vec<Binding>, // Deprecated, kept for parsing let/set blocks
        set_bindings: Vec<Binding>, // Deprecated, kept for parsing let/set blocks
        context_bindings: Vec<Binding>, // New unified bindings
        connections: Vec<Connection>,
    },
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
    /// Allow cycles up to this depth (for unrolled loops/recursion)
    /// None means no cycles allowed, Some(n) allows cycles up to depth n
    pub max_cycle_depth: Option<usize>,
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
    pub globals: Vec<GlobalBinding>, // Module-level globals
    pub neurons: HashMap<String, NeuronDef>,
}

// lexer

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    Neuron,
    Use,
    In,
    Out,
    Impl,
    Graph,
    Match,
    Where,
    External,
    And,
    Or,
    Let, // Deprecated
    Set, // Deprecated
    Context,

    // Annotations
    AtStatic, // @static
    AtGlobal, // @global
    AtLazy,   // @lazy

    // Literals
    Int(i64),
    Float(f64),
    String(String),
    True,
    False,

    // Identifiers
    Ident(String),

    // Operators
    Arrow,  // ->
    Colon,  // :
    Comma,  // ,
    Dot,    // .
    Slash,  // /
    Star,   // *
    Plus,   // +
    Minus,  // -
    Eq,     // ==
    Ne,     // !=
    Lt,     // <
    Gt,     // >
    Le,     // <=
    Ge,     // >=
    Assign, // =

    // Delimiters
    LParen,   // (
    RParen,   // )
    LBracket, // [
    RBracket, // ]

    // Structure
    Newline,
    Indent,
    Dedent,
    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub text: String,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Error, Diagnostic)]
pub enum LexError {
    #[error("Unexpected character '{ch}'")]
    UnexpectedChar {
        ch: char,
        #[label("here")]
        span: SourceSpan,
    },

    #[error("Unterminated string")]
    UnterminatedString {
        #[label("string starts here")]
        span: SourceSpan,
    },

    #[error("Invalid number")]
    InvalidNumber {
        #[label("here")]
        span: SourceSpan,
    },

    #[error("Inconsistent indentation")]
    InconsistentIndent {
        #[label("expected {expected} spaces, found {found}")]
        span: SourceSpan,
        expected: usize,
        found: usize,
    },
}

pub struct Lexer<'a> {
    pub source: &'a str,
    pub chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    pub pos: usize,
    pub line: usize,
    pub col: usize,
    pub indent_stack: Vec<usize>,
    pub pending_dedents: usize,
    pub at_line_start: bool,
}

#[derive(Debug, Error, Diagnostic)]
pub enum ParseError {
    #[error("Expected {expected}, found {found}")]
    Expected {
        expected: String,
        found: String,
        #[label("here")]
        span: SourceSpan,
    },

    #[error("Unexpected token '{found}'")]
    Unexpected {
        found: String,
        #[label("unexpected")]
        span: SourceSpan,
    },

    #[error("Duplicate neuron definition '{name}'")]
    DuplicateNeuron {
        name: String,
        #[label("redefined here")]
        span: SourceSpan,
    },

    #[error("{0}")]
    Lex(#[from] crate::LexError),
}

pub struct Parser {
    pub tokens: Vec<Token>,
    pub pos: usize,
    pub next_node_id: usize,
}

// Shape is already defined above

/// Tracks the state of dimension variables during inference
#[derive(Debug, Clone, Default)]
pub struct InferenceContext {
    /// Map from dimension name to its resolved value (if known)
    pub resolved_dims: HashMap<String, usize>,

    /// Map from dimension name to other equivalent dimension names
    pub equivalences: HashMap<String, String>,

    /// Map from named nodes to their output shapes
    /// e.g. "in" -> [Shape], "x" -> [Shape]
    pub node_outputs: HashMap<String, Vec<Shape>>,

    /// Map from anonymous call IDs to their output shapes
    /// e.g. Linear(512, 256) (id=1) -> [[*, 256]]
    pub call_outputs: HashMap<usize, Vec<Shape>>,

    /// Map from variadic name to the sequence of dimensions it captured
    /// e.g. "shape" -> [batch, seq]
    pub resolved_variadics: HashMap<String, Vec<Dim>>,

    /// Pending expression constraints to be solved
    /// Format: (result_dim, expression, context)
    pub pending_constraints: Vec<(Dim, DimExpr, String)>,
}

#[derive(Debug, Error)]
pub enum ShapeError {
    #[error("Shape mismatch: expected {expected}, got {got}")]
    Mismatch {
        expected: Shape,
        got: Shape,
        context: String,
    },

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimMismatch {
        expected: Dim,
        got: Dim,
        context: String,
    },

    #[error("Unknown dimension variable: {name}")]
    UnknownDim { name: String, context: String },

    #[error("Constraint violation: {message}")]
    ConstraintViolation { message: String, context: String },

    #[error("Inference failed for node {node}: {message}")]
    NodeInferenceFailed { node: String, message: String },

    #[error("Unknown node or port: {0}")]
    UnknownNode(String),

    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),
}

/// Implementation reference for a primitive neuron.
#[derive(Clone, Debug, PartialEq)]
pub enum ImplRef {
    /// External implementation with keyword arguments
    External {
        /// Keyword arguments for external implementation
        kwargs: Vec<Kwarg>,
    },
    /// Source-based implementation
    Source {
        /// Source identifier
        source: String,
        /// Path within the source
        path: String,
    },
}

/// Standard library registry - maps neuron names to implementations.
pub struct StdlibRegistry {
    pub primitives: HashMap<String, ImplRef>,
}

/// Validation errors
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    MissingNeuron {
        name: String,
        context: String,
    },
    PortMismatch {
        source_node: String,
        source_port: String,
        source_shape: Shape,
        dest_node: String,
        dest_port: String,
        dest_shape: Shape,
        context: String,
    },
    CycleDetected {
        cycle: Vec<String>,
        context: String,
    },
    ArityMismatch {
        expected: usize,
        got: usize,
        context: String,
    },
    UnknownNode {
        node: String,
        context: String,
    },
    NonExhaustiveMatch {
        context: String,
        suggestion: String,
    },
    UnreachableMatchArm {
        arm_index: usize,
        shadowed_by: usize,
        context: String,
    },
    DuplicateBinding {
        name: String,
        neuron: String,
    },
    InvalidRecursion {
        binding: String,
        neuron: String,
        reason: String,
    },
    Custom(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::MissingNeuron { name, context } => {
                write!(f, "Neuron '{}' not found (in {})", name, context)
            }
            ValidationError::PortMismatch {
                source_node,
                source_port,
                source_shape,
                dest_node,
                dest_port,
                dest_shape,
                context,
            } => {
                let source = if source_port == "default" {
                    source_node.clone()
                } else {
                    format!("{}.{}", source_node, source_port)
                };
                let dest = if dest_port == "default" {
                    dest_node.clone()
                } else {
                    format!("{}.{}", dest_node, dest_port)
                };

                write!(
                    f,
                    "Port mismatch: {} {} -> {} {} (in {})\n  Suggestion: check if dimensions match or if a transpose/reshape is needed.",
                    source, source_shape, dest, dest_shape, context
                )
            }
            ValidationError::CycleDetected { cycle, context } => {
                write!(f, "Cycle detected in {}: {}", context, cycle.join(" -> "))
            }
            ValidationError::ArityMismatch {
                expected,
                got,
                context,
            } => {
                write!(
                    f,
                    "Arity mismatch: expected {} ports, got {} (in {})",
                    expected, got, context
                )
            }
            ValidationError::UnknownNode { node, context } => {
                write!(f, "Unknown node '{}' (in {})", node, context)
            }
            ValidationError::NonExhaustiveMatch {
                context,
                suggestion,
            } => {
                write!(
                    f,
                    "Non-exhaustive match expression (in {}): {}",
                    context, suggestion
                )
            }
            ValidationError::UnreachableMatchArm {
                arm_index,
                shadowed_by,
                context,
            } => {
                write!(
                    f,
                    "Unreachable match arm {} shadowed by arm {} (in {})",
                    arm_index, shadowed_by, context
                )
            }
            ValidationError::DuplicateBinding { name, neuron } => {
                write!(f, "Duplicate binding '{}' in neuron '{}'", name, neuron)
            }
            ValidationError::InvalidRecursion {
                binding,
                neuron,
                reason,
            } => {
                write!(
                    f,
                    "Invalid recursion in binding '{}' of neuron '{}': {}",
                    binding, neuron, reason
                )
            }
            ValidationError::Custom(msg) => {
                write!(f, "{}", msg)
            }
        }
    }
}
