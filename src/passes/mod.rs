//! Compiler passes that transform the IR before validation and codegen.
//!
//! - `unroll`: Expands `context: unroll(N)` blocks into repeated bindings
//! - `desugar`: Rewrites `@wrap` annotations into standard `Call` endpoints
//! - `contract_resolver`: Resolves `match(param):` expressions at compile time

pub mod contract_resolver;
pub mod desugar;
pub mod unroll;
