//! NeuroScript Intermediate Representation
//!
//! The IR is a graph of neuron instantiations with typed edges.
//! Every connection carries a shape contract.

use miette::Diagnostic;
use miette::SourceSpan;
use std::collections::HashMap;
use std::cell::Cell;
use thiserror::Error;

/// Global ID generator for unique endpoint IDs across parsing and IR passes.
///
/// Counter for generating unique endpoint IDs within a compilation.
/// Uses interior mutability via Cell since the compiler is single-threaded.
pub(crate) struct IdGenerator {
    next: Cell<usize>,
}

impl IdGenerator {
    pub fn new() -> Self {
        Self {
            next: Cell::new(0),
        }
    }

    pub fn next_id(&self) -> usize {
        let id = self.next.get();
        self.next.set(id + 1);
        id
    }

    /// Return the current counter value (for passing to later passes).
    pub fn current(&self) -> usize {
        self.next.get()
    }
}

impl Default for IdGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// A dimension in a tensor shape
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dim {
    /// Literal value: 512
    Literal(i64),
    /// Named dimension: batch, seq, dim
    Named(String),
    /// Wildcard: * (matches any single dimension)
    Wildcard,
    /// Inferred dimension: collapses remaining dims (like PyTorch's -1 in reshape).
    /// Semantically different from Wildcard: Wildcard matches exactly one unknown
    /// dimension in shape patterns, while Inferred means "compute this dimension
    /// from the total element count and other specified dimensions" (i.e., may absorb
    /// multiple dimensions). Only produced by `ReshapeExpr::to_shape()` from
    /// `ReshapeDim::Others`.
    Inferred,
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

/// An endpoint in a connection — either a source or destination of data flow.
///
/// Endpoints form the nodes of the dataflow graph inside a composite neuron.
/// They range from simple port references (`in`, `out`) to inline neuron calls
/// (`Linear(512, 256)`), pattern matching (`match:`), conditionals (`if/elif/else`),
/// shape transforms (`=>`), and wrapper annotations (`@wrap`).
#[derive(Debug, Clone, PartialEq)]
pub enum Endpoint {
    /// Simple reference: `in`, `out`, `my_binding`
    Ref(PortRef),
    /// Tuple of references: `(a, b)` — for implicit fork
    Tuple(Vec<PortRef>),
    /// Neuron instantiation: `Linear(512, 256)`
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

/// A connection in the dataflow graph: `source -> destination`.
///
/// Represents a single edge in the neuron's internal graph. Data flows from
/// the source endpoint to the destination endpoint. Multiple connections form
/// a pipeline (e.g., `in -> Linear(512, 256) -> ReLU() -> out`).
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
    /// # Known limitations
    ///
    /// ## Binding expressions are dropped (deliberate)
    ///
    /// `Binding { name, expr }` (e.g., `dh=dim/heads`) maps to `Dim::Named(name)`,
    /// intentionally discarding the expression constraint. The shape solver currently
    /// only handles simple single-unknown linear equations (e.g., `dim = 512`). It
    /// cannot verify divisibility constraints like `dh = dim / heads` because that
    /// requires solving a two-variable equation, which is beyond the solver's scope.
    ///
    /// Adding a `Dim::Constrained(name, expr)` variant was considered but rejected
    /// because `Dim` is pattern-matched in 20+ files; adding a variant would cause
    /// cascading changes with no immediate benefit until the solver is extended to
    /// handle multi-variable constraints. The expression *is* preserved in the IR
    /// (`ReshapeDim::Binding`) and is available to codegen, which correctly emits the
    /// computed reshape size — so runtime behavior is correct even though the
    /// compile-time shape checker cannot verify the constraint.
    ///
    /// To fix this properly, the shape solver would need to be extended to support
    /// constraint propagation (e.g., `dh * heads == dim`), which is a larger effort
    /// tracked under "full dimension variable type inference" in the roadmap.
    ///
    /// ## `Others` loses rank information
    ///
    /// `Others` maps to `Dim::Wildcard`, but semantically `others` means "collapse
    /// all remaining dims" (like PyTorch's `-1`), not "one unknown dim". This means
    /// rank validation doesn't catch mismatches through `others`. A `Dim::Inferred`
    /// variant or separate rank tracking would be needed to fix this.
    pub fn to_shape(&self) -> Shape {
        Shape {
            dims: self
                .dims
                .iter()
                .map(|d| match d {
                    ReshapeDim::Named(name) => Dim::Named(name.clone()),
                    ReshapeDim::Literal(n) => Dim::Literal(*n),
                    // Expression is intentionally dropped here — see doc comment above.
                    // The binding expr is still available in ReshapeDim::Binding for codegen.
                    ReshapeDim::Binding { name, .. } => Dim::Named(name.clone()),
                    ReshapeDim::Others => Dim::Inferred,
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

/// A binding in a `context:` block — instantiates a sub-module.
///
/// Bindings define named sub-modules within a composite neuron. They are
/// instantiated in `__init__` and referenced in `forward()`. Bindings can
/// be `@lazy` (deferred instantiation), `@static` (class-level shared),
/// or default eager instance-level.
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

/// A complete neuron definition — the fundamental building block of NeuroScript.
///
/// A neuron is either **primitive** (backed by an external implementation like
/// `nn.Linear`) or **composite** (defined by an internal dataflow graph of
/// connections between other neurons). Composite neurons may have `context:`
/// bindings that instantiate sub-modules and `graph:` connections that wire
/// data flow.
#[derive(Debug, Clone, PartialEq)]
pub struct NeuronDef {
    /// Neuron name (e.g., `TransformerBlock`)
    pub name: String,
    /// Constructor parameters (e.g., `d_model`, `num_heads`)
    pub params: Vec<Param>,
    /// Input port declarations with shapes
    pub inputs: Vec<Port>,
    /// Output port declarations with shapes
    pub outputs: Vec<Port>,
    /// Body: either a primitive `impl:` reference or a composite `graph:`
    pub body: NeuronBody,
    /// Allow cycles up to this depth (for unrolled loops/recursion).
    /// `None` means no cycles allowed, `Some(n)` allows cycles up to depth `n`.
    pub max_cycle_depth: Option<usize>,
    /// Optional documentation from `///` doc comments
    pub doc: Option<Documentation>,
}

/// An import statement: use core,nn/*
#[derive(Debug, Clone, PartialEq)]
pub struct UseStmt {
    pub source: String,
    pub path: Vec<String>,
}

/// A complete NeuroScript program — the top-level compilation unit.
///
/// Contains use-imports, module-level `@global` declarations, and a map of
/// neuron definitions keyed by name. This is the output of the parser and the
/// input to validation, optimization, and code generation.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// Module-level `use` imports (e.g., `use core,nn/*`)
    pub uses: Vec<UseStmt>,
    /// Module-level `@global` declarations (e.g., `@global vocab_size = 50257`)
    pub globals: Vec<GlobalBinding>,
    /// All neuron definitions, keyed by name
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

    /// Pending expression constraints to be solved later when more dims are known.
    /// Format: (result_dim, expression, context)
    /// TODO(SHAPE-4): these are pushed but never drained — add a flush pass that
    /// retries resolution after all direct unifications complete.
    pub pending_constraints: Vec<(Dim, DimExpr, String)>,
}

#[derive(Debug, Error, Diagnostic)]
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

/// Validation errors collected during the validation pass.
///
/// The validator collects all errors rather than failing fast, so users see
/// every problem in a single run. Variants cover missing neurons, shape
/// mismatches, cycles, arity errors, binding issues, and more.
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
        /// Source span for diagnostic reporting
        span: Option<SourceSpan>,
    },
    InvalidAnnotation {
        annotation: String,
        reason: String,
        context: String,
        /// Source span for diagnostic reporting
        span: Option<SourceSpan>,
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
    MutualLazyRecursion {
        /// Binding names forming the cycle, e.g. ["a", "b", "a"]
        cycle: Vec<String>,
        neuron: String,
        /// Source span for diagnostic reporting (None until Binding carries spans)
        span: Option<SourceSpan>,
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
            ValidationError::InvalidReshape { message, context, .. } => {
                write!(f, "Invalid reshape: {} (in {})", message, context)
            }
            ValidationError::InvalidAnnotation {
                annotation,
                reason,
                context,
                ..
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
            ValidationError::MutualLazyRecursion { cycle, neuron, .. } => {
                write!(
                    f,
                    "Mutual @lazy recursion detected between bindings: {} (in {})",
                    cycle.join(" -> "), neuron
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

// === Convenience constructors ===

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
            Dim::Inferred => write!(f, "..."),
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
            Endpoint::If(expr) => write!(f, "{}", expr),
            Endpoint::Reshape(r) => write!(f, "{}", r),
            Endpoint::Wrap(w) => write!(f, "{}", w),
        }
    }
}

impl std::fmt::Display for IfExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, branch) in self.branches.iter().enumerate() {
            if i == 0 {
                write!(f, "if {}: ", branch.condition)?;
            } else {
                write!(f, "elif {}: ", branch.condition)?;
            }
            for (j, e) in branch.pipeline.iter().enumerate() {
                if j > 0 {
                    write!(f, " -> ")?;
                }
                write!(f, "{}", e)?;
            }
        }
        if let Some(else_branch) = &self.else_branch {
            write!(f, " else: ")?;
            for (j, e) in else_branch.iter().enumerate() {
                if j > 0 {
                    write!(f, " -> ")?;
                }
                write!(f, "{}", e)?;
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for WrapExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "@wrap({}", self.wrapper_name)?;
        for arg in &self.wrapper_args {
            write!(f, ", {}", arg)?;
        }
        for (k, v) in &self.wrapper_kwargs {
            write!(f, ", {}={}", k, v)?;
        }
        write!(f, "): ")?;
        match &self.content {
            WrapContent::Ref(name) => write!(f, "{}", name),
            WrapContent::Pipeline(pipeline) => {
                write!(f, "-> ")?;
                for (i, ep) in pipeline.iter().enumerate() {
                    if i > 0 {
                        write!(f, " -> ")?;
                    }
                    write!(f, "{}", ep)?;
                }
                Ok(())
            }
        }
    }
}

impl std::fmt::Display for MatchPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatchPattern::Shape(s) => write!(f, "{}", s),
            MatchPattern::NeuronContract(c) => write!(f, "{}", c),
        }
    }
}

impl std::fmt::Display for NeuronPortContract {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, (name, shape)) in self.input_ports.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if name == "default" {
                write!(f, "in {}", shape)?;
            } else {
                write!(f, "in {} {}", name, shape)?;
            }
        }
        write!(f, " -> ")?;
        for (i, (name, shape)) in self.output_ports.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if name == "default" {
                write!(f, "out {}", shape)?;
            } else {
                write!(f, "out {} {}", name, shape)?;
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for MatchSubject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatchSubject::Implicit => write!(f, ""),
            MatchSubject::Named(name) => write!(f, "({})", name),
        }
    }
}

impl std::fmt::Display for MatchExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.subject {
            MatchSubject::Implicit => writeln!(f, "match: ->")?,
            MatchSubject::Named(name) => writeln!(f, "match({}):", name)?,
        }
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

impl std::fmt::Display for ContextUnroll {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} = unroll({}):", self.aggregate_name, self.count)?;
        for b in &self.bindings {
            writeln!(f, "      {}", b)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let arrow = match &self.destination {
            Endpoint::Reshape(_) => "=>",
            _ => "->",
        };
        write!(f, "{} {} {}", self.source, arrow, self.destination)
    }
}

impl std::fmt::Display for ReshapeDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReshapeDim::Named(name) => write!(f, "{}", name),
            ReshapeDim::Literal(n) => write!(f, "{}", n),
            ReshapeDim::Binding { name, expr } => write!(f, "{}={}", name, expr),
            ReshapeDim::Others => write!(f, "others"),
            ReshapeDim::Expr(expr) => write!(f, "({} {} {})", expr.left, expr.op, expr.right),
        }
    }
}

impl std::fmt::Display for ReshapeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref ann) = self.annotation {
            write!(f, "{} ", ann)?;
        }
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

impl std::fmt::Display for TransformAnnotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformAnnotation::Reduce(s) => write!(f, "@reduce({})", s),
            TransformAnnotation::Repeat(s) => write!(f, "@repeat({})", s),
        }
    }
}

impl std::fmt::Display for TransformStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformStrategy::Intrinsic(name) => write!(f, "{}", name),
            TransformStrategy::Neuron { name, args, kwargs } => {
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
            if let Some(ParamType::Neuron) = &p.type_annotation {
                write!(f, ": Neuron")?;
            }
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
                context_unrolls,
                connections,
            } => {
                if !context_bindings.is_empty() || !context_unrolls.is_empty() {
                    writeln!(f, "  context:")?;
                    for b in context_bindings {
                        writeln!(f, "    {}", b)?;
                    }
                    for u in context_unrolls {
                        write!(f, "    {}", u)?;
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
