//! NeuroScript - Neural Architecture Composition Language
//!
//! A language for defining composable neural architectures.
//!
//! # Example
//!
//! ```neuroscript
//! neuron MLP(dim):
//!   in: [*, dim]
//!   out: [*, dim]
//!   graph:
//!     in ->
//!       Linear(dim, dim * 4)
//!       GELU()
//!       Linear(dim * 4, dim)
//!       out
//! ```

pub mod codegen;
pub mod doc_parser;
pub mod grammar;
pub mod interfaces;
pub mod optimizer;
#[cfg(feature = "cli")]
pub mod package;
pub mod passes;
pub mod shape;
pub mod stdlib;
pub mod stdlib_registry;
pub mod validator;
pub mod visitor;
#[cfg(feature = "wasm")]
pub mod wasm;

// Re-export main IR types explicitly (no glob re-exports)
pub use codegen::{generate_pytorch, generate_pytorch_with_options, CodegenOptions};
pub use interfaces::{
    BinOp, Binding, Connection, ContextUnroll, Dim, DimExpr, Endpoint, GlobalBinding,
    IfBranch, IfExpr, ImplRef, InferenceContext, MatchArm, MatchExpr, MatchPattern, MatchSubject,
    Documentation, NeuronBody, NeuronDef, NeuronPortContract, Param, ParamType, ParseError, Port, PortRef,
    Program, ReshapeDim, ReshapeExpr, Scope, Shape, ShapeError, StdlibRegistry,
    TransformAnnotation, TransformStrategy, UnrollGroupInfo, UseStmt, ValidationError, Value,
    WrapContent, WrapExpr, SEQUENTIAL_PSEUDO_NEURON,
};
pub use validator::Validator;

/// Parse a NeuroScript source string into a Program using the pest grammar.
pub fn parse(source: &str) -> Result<Program, ParseError> {
    grammar::NeuroScriptParser::parse_program(source)
}

/// Prepare a parsed program for validation by applying IR transformations.
///
/// This function mutates the program in place:
/// - Expands `unroll` constructs into repeated bindings
/// - Desugars `@wrap` annotations into standard `Call` endpoints
///
/// After calling this, `Endpoint::Wrap` nodes will no longer appear in the IR
/// and all `unroll` blocks will be fully expanded.
///
/// This is automatically called by [`validate`], but is exposed as a separate
/// step for callers who need explicit control over the pipeline (e.g., for
/// inspection or custom tooling between preparation and validation).
pub fn prepare(program: &mut Program) -> Result<(), Vec<ValidationError>> {
    // Expand unroll constructs first so any @wrap nodes inside
    // unroll templates are flattened into connections
    passes::unroll::expand_unrolls(program)?;

    // Desugar @wrap annotations into standard Call endpoints
    // Must run after unroll expansion and before validation
    passes::desugar::desugar_wraps(program)?;

    Ok(())
}

/// Validate a NeuroScript program for correctness.
///
/// This function first calls [`prepare`] to expand unrolls and desugar `@wrap`
/// annotations, then runs all validation checks. If you need to inspect or
/// modify the IR between preparation and validation, call [`prepare`] yourself
/// and then pass the prepared program to this function (it is safe to call
/// `prepare` multiple times — the transformations are idempotent).
///
/// Validation checks:
/// 1. All referenced neurons exist
/// 2. Connection endpoints match (tuple arity, port names, shapes)
/// 3. No cycles in the dependency graph
/// 4. Shape compatibility for all connections (shape inference)
/// 5. Resolve neuron contract match expressions (`match(param): ...`)
///    for higher-order neurons with `: Neuron` typed parameters
pub fn validate(program: &mut Program) -> Result<(), Vec<ValidationError>> {
    // Prepare the IR (expand unrolls, desugar @wrap)
    prepare(program)?;

    // First run basic validation (read-only)
    validator::Validator::validate(program)?;

    // Compute reachability for match arms (marks unreachable/shadowed arms).
    // Must run after validation (which checks exhaustiveness) and before
    // contract resolution and codegen (which consume is_reachable flags).
    optimizer::compute_reachability(program);

    // Then run shape inference validation
    let mut shape_engine = shape::ShapeInferenceEngine::new();
    match shape_engine.infer(program) {
        Ok(()) => {}
        Err(shape_errors) => {
            // Convert shape errors to validation errors
            let validation_errors = shape_errors
                .into_iter()
                .map(|e| ValidationError::Custom(format!("Shape error: {}", e)))
                .collect();
            return Err(validation_errors);
        }
    }

    // Resolve neuron contract match expressions (match(param): ...)
    passes::contract_resolver::resolve_neuron_contracts(program)?;

    Ok(())
}
