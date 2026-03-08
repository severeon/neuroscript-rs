//! Compile-time neuron contract resolution.
//!
//! Resolves `match(param): in [...] -> out [...]: ...` expressions by inspecting
//! the concrete neuron passed as the parameter at each call site. The matching
//! arm's pipeline replaces the entire match expression.
//!
//! Called after validation/shape inference, before codegen.
//!
//! # Example
//!
//! Given a higher-order neuron:
//! ```text
//! neuron SmartStack(block: Neuron, d_model, count=6):
//!     ...
//!     graph:
//!         in -> match(block):
//!             in [*, seq, d] -> out [*, seq, d]: blocks -> out
//!             in [*, d] -> out [*, d]: blocks -> out
//! ```
//!
//! When `SmartStack` is called with `SmartStack(TransformerBlock, 512)`, the
//! resolver looks up `TransformerBlock`'s port declarations and matches them
//! against each arm's contract. The first matching arm's pipeline replaces the
//! match expression.
//!
//! # Multi-endpoint pipelines
//!
//! When a matching arm contains a multi-step pipeline (e.g., `blocks -> out`),
//! the resolver splices the pipeline into the parent connection graph:
//! - The first endpoint replaces the match expression inline
//! - Additional connections are inserted for the remaining pipeline steps

mod call_sites;
mod detection;
mod matching;
mod resolution;
#[cfg(test)]
mod tests;

use crate::interfaces::*;

/// Maximum recursion depth for contract resolution to prevent infinite loops
/// in pathological cases (e.g., mutually recursive contract definitions).
pub(crate) const MAX_CONTRACT_RESOLUTION_DEPTH: usize = 32;

/// Resolve all neuron contract match expressions in the program.
///
/// For each composite neuron containing a `MatchExpr` with `MatchSubject::Named(param)`:
/// 1. Find all call sites where this neuron is instantiated with a concrete neuron argument
/// 2. Look up the concrete neuron's port declarations
/// 3. Match ports against each arm's `NeuronPortContract`
/// 4. Select the first matching arm's pipeline
///
/// Returns `Err` with collected errors if any contracts cannot be resolved.
#[must_use]
pub fn resolve_neuron_contracts(program: &mut Program) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    // Collect neurons that have Named match subjects
    let neurons_with_contracts: Vec<String> = program
        .neurons
        .iter()
        .filter(|(_, neuron)| detection::has_named_match(neuron))
        .map(|(name, _)| name.clone())
        .collect();

    if neurons_with_contracts.is_empty() {
        return Ok(());
    }

    // For each neuron with contract matches, find call sites and resolve
    for neuron_name in &neurons_with_contracts {
        errors.extend(resolution::resolve_contracts_for_neuron(program, neuron_name));
    }

    // Post-resolution check: detect any remaining MatchSubject::Named patterns
    // that weren't resolved (e.g., because the argument was a complex expression
    // rather than a simple neuron name). These will cause codegen failures, so
    // report them here with a clear message.
    //
    // Only run this check when resolution itself didn't produce errors — if
    // resolution already failed (no matching arm, etc.), the Named match is
    // still present and would be redundantly flagged here.
    if errors.is_empty() {
        for (neuron_name, neuron) in &program.neurons {
            if let NeuronBody::Graph { connections, .. } = &neuron.body {
                for conn in connections {
                    detection::collect_unresolved_contracts(&conn.source, neuron_name, &mut errors, 0);
                    detection::collect_unresolved_contracts(&conn.destination, neuron_name, &mut errors, 0);
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
