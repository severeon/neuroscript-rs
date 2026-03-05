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
    /// Global: @global hidden_dim
    Global(String),
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

    /// Names of parameters with `: Neuron` type annotation (higher-order neuron params)
    pub neuron_typed_params: HashSet<String>,

    /// Dimension bindings from match pattern captures (e.g., "d" -> "x.shape[1]")
    /// Used to resolve dimension references in match arm pipelines
    pub binding_context: HashMap<String, String>,

    /// Lazy bindings from let: blocks (name -> (call_name, args, kwargs))
    pub lazy_bindings: HashMap<String, LazyBinding>,

    /// Shape inference context (resolved dimensions and node output shapes)
    /// Used for emitting shape assertions and documentation
    pub inference_ctx: InferenceContext,

    /// Mapping from binding name to the neuron it calls (e.g., "block_0" -> "TransformerBlock")
    /// Used to look up output shapes for bound module calls
    pub binding_to_call_name: HashMap<String, String>,

    /// Mapping from binding name to its unroll group info
    /// Used to generate nn.ModuleList and for loops
    pub binding_to_unroll_group: HashMap<String, UnrollGroupInfo>,

    /// Mapping from aggregate name to (base_name, count, is_static) for direct aggregate references in graph
    /// is_static=true means weight-sharing: one class-level instance called N times
    pub aggregate_to_group: HashMap<String, (String, Value, bool)>,

    /// Last shape comment emitted, used to suppress duplicates
    pub last_emitted_shape: Option<String>,
}

/// An input or output port of a neuron
#[derive(Debug, Clone, PartialEq)]
pub struct Port {
    pub name: String, // "default" if unnamed
    pub shape: Shape,
    pub variadic: bool, // true for `in *inputs: [shape]`
}

/// A value in the language
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Name(String),
    Global(String),
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
        frozen: bool,
    },
    /// Pattern match expression
    Match(MatchExpr),
    /// Conditional expression
    If(IfExpr),
    /// Shape transformation: => [shape] or => @annotation [shape]
    Reshape(ReshapeExpr),
    /// Wrap annotation: @wrap(Wrapper, args): content
    Wrap(WrapExpr),
}

/// A connection: source -> destination
#[derive(Debug, Clone, PartialEq)]
pub struct Connection {
    pub source: Endpoint,
    pub destination: Endpoint,
}

/// A port shape contract for a neuron parameter, used in match(block) dispatch
#[derive(Debug, Clone, PartialEq)]
pub struct NeuronPortContract {
    pub input_ports: Vec<(String, Shape)>,   // (port_name, shape_pattern)
    pub output_ports: Vec<(String, Shape)>,
}

/// Pattern in a match arm: either a shape pattern or a neuron port contract
#[derive(Debug, Clone, PartialEq)]
pub enum MatchPattern {
    /// Shape pattern for data-threading match: `[*, dim]`
    Shape(Shape),
    /// Neuron contract pattern for parameter dispatch: `in [*shape] -> out [*shape]`
    NeuronContract(NeuronPortContract),
}

impl MatchPattern {
    /// Get the shape if this is a Shape pattern
    pub fn as_shape(&self) -> Option<&Shape> {
        match self {
            MatchPattern::Shape(s) => Some(s),
            MatchPattern::NeuronContract(_) => None,
        }
    }
}

/// What the match expression dispatches on
#[derive(Debug, Clone, PartialEq)]
pub enum MatchSubject {
    /// Threading: data flows through (traditional match on tensor shapes)
    Implicit,
    /// Evaluation: match on a named parameter (e.g., `match(block)`)
    Named(String),
}

/// One arm of a match expression
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: MatchPattern,
    pub guard: Option<Value>, // where clause
    pub pipeline: Vec<Endpoint>,
    /// Whether this arm is reachable (not shadowed by earlier arms)
    /// Set to false by validator for dead code elimination
    pub is_reachable: bool,
}

/// Pattern matching on shapes or neuron contracts
#[derive(Debug, Clone, PartialEq)]
pub struct MatchExpr {
    pub subject: MatchSubject,
    pub arms: Vec<MatchArm>,
    pub id: usize,
}

/// A branch in an if expression
#[derive(Debug, Clone, PartialEq)]
pub struct IfBranch {
    pub condition: Value,
    pub pipeline: Vec<Endpoint>,
}

/// If/Else conditional expression
#[derive(Debug, Clone, PartialEq)]
pub struct IfExpr {
    pub branches: Vec<IfBranch>,            // if and elifs
    pub else_branch: Option<Vec<Endpoint>>, // optional else
    pub id: usize,
}

/// Pseudo-neuron name used by @wrap pipeline desugaring to emit nn.Sequential.
/// This is a reserved internal name — user-defined neurons must not use it.
pub const SEQUENTIAL_PSEUDO_NEURON: &str = "__sequential__";

/// A @wrap annotation expression
#[derive(Debug, Clone, PartialEq)]
pub struct WrapExpr {
    /// The higher-order neuron to wrap with
    pub wrapper_name: String,
    /// Arguments to the wrapper (excluding the first Neuron-typed param)
    pub wrapper_args: Vec<Value>,
    /// Keyword arguments to the wrapper
    pub wrapper_kwargs: Vec<Kwarg>,
    /// The wrapped content: either a reference to an existing binding or an anonymous pipeline
    pub content: WrapContent,
    /// Unique ID for deduplication
    pub id: usize,
}

/// What @wrap wraps
#[derive(Debug, Clone, PartialEq)]
pub enum WrapContent {
    /// Reference form: @wrap(Wrapper, args): existing_binding
    Ref(String),
    /// Pipeline form: @wrap(Wrapper, args): -> X -> Y -> Z
    Pipeline(Vec<Endpoint>),
}

/// A reshape expression: [dim_spec, dim_spec, ...]
#[derive(Debug, Clone, PartialEq)]
pub struct ReshapeExpr {
    pub dims: Vec<ReshapeDim>,
    pub annotation: Option<TransformAnnotation>,
    pub id: usize,
}

impl ReshapeExpr {
    /// Convert reshape dims to a Shape for validation/inference purposes.
    ///
    /// Known limitations:
    /// - `Binding { name, expr }` (e.g., `dh=dim/heads`) maps to `Dim::Named(name)`,
    ///   discarding the expression constraint. Shape inference sees `dh` as unconstrained.
    ///   TODO: propagate binding constraints for tighter validation.
    /// - `Others` maps to `Dim::Wildcard`, but semantically `others` means "collapse all
    ///   remaining dims" (like PyTorch's -1), not "one unknown dim". This means rank
    ///   validation doesn't catch mismatches through `others`.
    ///   TODO: add a `Dim::Inferred` variant or track rank separately.
    pub fn to_shape(&self) -> Shape {
        Shape {
            dims: self
                .dims
                .iter()
                .map(|d| match d {
                    ReshapeDim::Named(name) => Dim::Named(name.clone()),
                    ReshapeDim::Literal(n) => Dim::Literal(*n),
                    ReshapeDim::Binding { name, .. } => Dim::Named(name.clone()),
                    ReshapeDim::Others => Dim::Wildcard,
                    ReshapeDim::Expr(expr) => Dim::Expr(expr.clone()),
                })
                .collect(),
        }
    }
}

/// A dimension spec in a reshape expression
#[derive(Debug, Clone, PartialEq)]
pub enum ReshapeDim {
    /// Named dimension reference: b, seq, dim
    Named(String),
    /// Literal value: 1, 5, 512
    Literal(i64),
    /// Decomposition binding: h=dim/heads
    Binding { name: String, expr: Box<Value> },
    /// Others keyword: flattens remaining dims
    Others,
    /// Dimension expression: h*w, dim/heads (uses existing DimExpr)
    Expr(Box<DimExpr>),
}

/// Transform annotation: @reduce(mean), @repeat(copy)
#[derive(Debug, Clone, PartialEq)]
pub enum TransformAnnotation {
    Reduce(TransformStrategy),
    Repeat(TransformStrategy),
}

/// Strategy for a transform: intrinsic name or neuron call
#[derive(Debug, Clone, PartialEq)]
pub enum TransformStrategy {
    /// Built-in: mean, sum, min, max, prod, logsumexp, copy
    Intrinsic(String),
    /// Neuron call: AttentionPool(dim)
    Neuron {
        name: String,
        args: Vec<Value>,
        kwargs: Vec<Kwarg>,
    },
}

/// Compile-time unroll block within a context section
#[derive(Debug, Clone, PartialEq)]
pub struct ContextUnroll {
    /// Aggregate name for the unroll group (e.g., "layers")
    pub aggregate_name: String,
    /// Number of iterations (must resolve to positive integer)
    pub count: Value,
    /// Bindings to replicate
    pub bindings: Vec<Binding>,
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

/// Metadata for bindings that were created by unroll expansion
#[derive(Debug, Clone, PartialEq)]
pub struct UnrollGroupInfo {
    /// Base name before suffixing (e.g., "block")
    pub base_name: String,
    /// The unroll count expression (e.g., Value::Name("num_layers") or Value::Int(12))
    pub count: Value,
    /// Index within the unroll group (0, 1, 2, ...)
    pub index: usize,
    /// Aggregate name for nn.ModuleList (e.g., "layers")
    pub aggregate_name: String,
}

/// A binding in a let: or set: block, or context: block
#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub name: String,
    pub call_name: String,
    pub args: Vec<Value>,
    pub kwargs: Vec<Kwarg>,
    pub scope: Scope,
    pub frozen: bool,
    /// If this binding was created by unroll expansion, metadata about its group
    pub unroll_group: Option<UnrollGroupInfo>,
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
        context_bindings: Vec<Binding>,
        context_unrolls: Vec<ContextUnroll>,
        connections: Vec<Connection>,
    },
}

/// Type annotation for a neuron parameter
#[derive(Debug, Clone, PartialEq)]
pub enum ParamType {
    /// Default: a numeric/value parameter
    Value,
    /// A neuron type parameter (higher-order neuron)
    Neuron,
}

/// A parameter in a neuron definition
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub default: Option<Value>,
    pub type_annotation: Option<ParamType>,
}

/// Documentation extracted from triple-slash comments
#[derive(Debug, Clone, PartialEq)]
pub struct Documentation {
    /// Raw markdown content (all doc comment lines joined)
    pub content: String,
    /// Parsed sections (e.g., "Parameters", "Shape Contract", "Example")
    pub sections: HashMap<String, String>,
    /// Source span for the documentation block
    pub span: Option<SourceSpan>,
}

impl Documentation {
    /// Create empty documentation
    pub fn empty() -> Self {
        Self {
            content: String::new(),
            sections: HashMap::new(),
            span: None,
        }
    }
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
    /// Optional documentation from triple-slash comments
    pub doc: Option<Documentation>,
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
    InvalidUnrollCount {
        neuron: String,
        reason: String,
    },
    InvalidReshape {
        message: String,
        context: String,
    },
    InvalidAnnotation {
        annotation: String,
        reason: String,
        context: String,
    },
    /// Match/if arms produce different port signatures.
    InconsistentArmPorts {
        expr_kind: String,
        arm_index: Option<usize>, // None = else branch, Some(n) = 1-based arm number
        expected_count: usize,
        got_count: usize,
        expected_names: Vec<String>,
        got_names: Vec<String>,
        context: String,
    },
    Custom(String),
    UseError {
        message: String,
    },
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
            ValidationError::InvalidUnrollCount { neuron, reason } => {
                write!(
                    f,
                    "Invalid unroll count in neuron '{}': {}",
                    neuron, reason
                )
            }
            ValidationError::InvalidReshape { message, context } => {
                write!(f, "Invalid reshape: {} (in {})", message, context)
            }
            ValidationError::InvalidAnnotation {
                annotation,
                reason,
                context,
            } => {
                write!(
                    f,
                    "Invalid annotation {}: {} ({})",
                    annotation, reason, context
                )
            }
            ValidationError::InconsistentArmPorts {
                expr_kind,
                arm_index,
                expected_count,
                got_count,
                expected_names,
                got_names,
                context,
            } => {
                let arm_label = match arm_index {
                    None => "else branch".to_string(),
                    Some(n) => format!("arm {}", n),
                };
                write!(
                    f,
                    "Inconsistent ports in {} expression: arm 1 has {} port(s) [{}] but {} has {} port(s) [{}] (in {})",
                    expr_kind,
                    expected_count, expected_names.join(", "),
                    arm_label,
                    got_count, got_names.join(", "),
                    context
                )
            }
            ValidationError::Custom(msg) => {
                write!(f, "{}", msg)
            }
            ValidationError::UseError { message } => {
                write!(f, "Import error: {}", message)
            }
        }
    }
}
