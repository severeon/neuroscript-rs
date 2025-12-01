use crate::interfaces::*;

/// Optimize match expressions by removing unreachable arms.
///
/// This pass traverses the program graph and finds all `Endpoint::Match` nodes.
/// For each match expression, it removes arms where `is_reachable` is false.
/// It returns the total number of pruned arms.
pub fn optimize_matches(program: &mut Program) -> usize {
    let mut pruned_count = 0;
    for neuron in program.neurons.values_mut() {
        if let NeuronBody::Graph(connections) = &mut neuron.body {
            for connection in connections {
                pruned_count += optimize_endpoint(&mut connection.source);
                pruned_count += optimize_endpoint(&mut connection.destination);
            }
        }
    }
    pruned_count
}

fn optimize_endpoint(endpoint: &mut Endpoint) -> usize {
    let mut count = 0;
    match endpoint {
        Endpoint::Match(match_expr) => {
            // Prune arms
            let initial_len = match_expr.arms.len();
            match_expr.arms.retain(|arm| arm.is_reachable);
            count += initial_len - match_expr.arms.len();

            // Recurse into remaining arms
            for arm in &mut match_expr.arms {
                for pipe_endpoint in &mut arm.pipeline {
                    count += optimize_endpoint(pipe_endpoint);
                }
            }
        }
        _ => {}
    }
    count
}

/// Count the total number of match expressions in the program.
/// This is useful for logging optimizer statistics.
pub fn count_matches(program: &Program) -> usize {
    let mut count = 0;
    for neuron in program.neurons.values() {
        if let NeuronBody::Graph(connections) = &neuron.body {
            for connection in connections {
                count += count_matches_in_endpoint(&connection.source);
                count += count_matches_in_endpoint(&connection.destination);
            }
        }
    }
    count
}

fn count_matches_in_endpoint(endpoint: &Endpoint) -> usize {
    let mut count = 0;
    match endpoint {
        Endpoint::Match(match_expr) => {
            count += 1;
            // Recurse into arms
            for arm in &match_expr.arms {
                for pipe_endpoint in &arm.pipeline {
                    count += count_matches_in_endpoint(pipe_endpoint);
                }
            }
        }
        _ => {}
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_optimize_matches() {
        // Construct a program with a match expression having an unreachable arm
        let mut program = Program {
            uses: vec![],
            neurons: HashMap::new(),
        };

        let match_expr = MatchExpr {
            arms: vec![
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Literal(1)],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: true,
                },
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Literal(1)],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: false, // This one should be pruned
                },
                MatchArm {
                    pattern: Shape {
                        dims: vec![Dim::Literal(2)],
                    },
                    guard: None,
                    pipeline: vec![],
                    is_reachable: true,
                },
            ],
        };

        let connection = Connection {
            source: Endpoint::Ref(PortRef {
                node: "in".to_string(),
                port: "default".to_string(),
            }),
            destination: Endpoint::Match(match_expr),
        };

        let neuron = NeuronDef {
            name: "TestNeuron".to_string(),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
            body: NeuronBody::Graph(vec![connection]),
        };

        program.neurons.insert("TestNeuron".to_string(), neuron);

        let pruned = optimize_matches(&mut program);
        assert_eq!(pruned, 1);

        // Verify the arm was removed
        let neuron = program.neurons.get("TestNeuron").unwrap();
        if let NeuronBody::Graph(connections) = &neuron.body {
            if let Endpoint::Match(match_expr) = &connections[0].destination {
                assert_eq!(match_expr.arms.len(), 2);
                assert_eq!(match_expr.arms[0].is_reachable, true);
                assert_eq!(match_expr.arms[1].is_reachable, true);
            } else {
                panic!("Expected Match endpoint");
            }
        } else {
            panic!("Expected Graph body");
        }
    }
}
