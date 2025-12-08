//! Visitor pattern for traversing IR trees
//!
//! This module provides generic visitor traits and walking functions for traversing
//! Endpoint and Value trees. This eliminates ~300 lines of duplicated traversal logic
//! across the codebase.
//!
//! # Example
//!
//! ```ignore
//! struct CallCollector {
//!     calls: Vec<Endpoint>,
//! }
//!
//! impl EndpointVisitor for CallCollector {
//!     fn visit_endpoint(&mut self, endpoint: &Endpoint) {
//!         if let Endpoint::Call { .. } = endpoint {
//!             self.calls.push(endpoint.clone());
//!         }
//!         // Continue walking
//!         walk_endpoint(self, endpoint);
//!     }
//! }
//! ```

use crate::interfaces::*;

/// Visitor trait for traversing Endpoint trees
///
/// Implement this trait to perform custom operations while walking an Endpoint tree.
/// The default implementation calls `walk_endpoint` to continue traversal.
pub trait EndpointVisitor {
    /// Visit an endpoint node
    ///
    /// Override this to perform custom logic. Call `walk_endpoint(self, endpoint)`
    /// to continue traversing children.
    fn visit_endpoint(&mut self, endpoint: &Endpoint) {
        walk_endpoint(self, endpoint);
    }

    /// Visit a match arm (called for each arm in a Match endpoint)
    fn visit_match_arm(&mut self, arm: &MatchArm) {
        walk_match_arm(self, arm);
    }
}

/// Walk an Endpoint tree, calling visitor methods for each node
///
/// This function implements the default traversal logic for Endpoints.
/// It visits all children recursively.
pub fn walk_endpoint<V: EndpointVisitor + ?Sized>(visitor: &mut V, endpoint: &Endpoint) {
    match endpoint {
        Endpoint::Call { .. } => {
            // Call endpoints have no children in the IR (args/kwargs are Values, not Endpoints)
        }
        Endpoint::Tuple(_refs) => {
            // Tuple unpacking contains PortRefs, not Endpoints
        }
        Endpoint::Ref(_) => {
            // Port references have no children
        }
        Endpoint::Match(match_expr) => {
            for arm in &match_expr.arms {
                visitor.visit_match_arm(arm);
            }
        }
    }
}

/// Walk a match arm, visiting all endpoints in its pipeline
pub fn walk_match_arm<V: EndpointVisitor + ?Sized>(visitor: &mut V, arm: &MatchArm) {
    for endpoint in &arm.pipeline {
        visitor.visit_endpoint(endpoint);
    }
}

/// Walk all connections, visiting source and destination endpoints
pub fn walk_connections<V: EndpointVisitor + ?Sized>(visitor: &mut V, connections: &[Connection]) {
    for conn in connections {
        visitor.visit_endpoint(&conn.source);
        visitor.visit_endpoint(&conn.destination);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CallCollector {
        calls: Vec<String>,
    }

    impl EndpointVisitor for CallCollector {
        fn visit_endpoint(&mut self, endpoint: &Endpoint) {
            if let Endpoint::Call { name, .. } = endpoint {
                self.calls.push(name.clone());
            }
            // Continue walking
            walk_endpoint(self, endpoint);
        }
    }

    #[test]
    fn test_call_collector() {
        let mut collector = CallCollector { calls: Vec::new() };

        // Simple call
        let call = Endpoint::Call {
            name: "Linear".to_string(),
            args: vec![],
            kwargs: vec![],
            id: 0,
        };

        collector.visit_endpoint(&call);
        assert_eq!(collector.calls, vec!["Linear"]);
    }

    #[test]
    fn test_call_collector_match() {
        let mut collector = CallCollector { calls: Vec::new() };

        // Match expression with calls in pipeline
        let match_expr = MatchExpr {
            arms: vec![
                MatchArm {
                    pattern: Shape::new(vec![Dim::Wildcard]),
                    guard: None,
                    pipeline: vec![
                        Endpoint::Call {
                            name: "Linear".to_string(),
                            args: vec![],
                            kwargs: vec![],
                            id: 0,
                        },
                        Endpoint::Call {
                            name: "ReLU".to_string(),
                            args: vec![],
                            kwargs: vec![],
                            id: 1,
                        },
                    ],
                    is_reachable: true,
                },
            ],
        };

        collector.visit_endpoint(&Endpoint::Match(match_expr));
        assert_eq!(collector.calls, vec!["Linear", "ReLU"]);
    }
}
