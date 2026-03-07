use super::*;
use pest::Parser;

#[test]
fn test_parse_simple_literal() {
    let input = "42";
    let result = NeuroScriptParser::parse(Rule::integer, input);
    assert!(
        result.is_ok(),
        "Failed to parse integer: {:?}",
        result.err()
    );
}

#[test]
fn test_parse_identifier() {
    let input = "my_neuron";
    let result = NeuroScriptParser::parse(Rule::ident, input);
    assert!(
        result.is_ok(),
        "Failed to parse identifier: {:?}",
        result.err()
    );
}

#[test]
fn test_parse_shape() {
    let input = "[batch, seq, dim]";
    let result = NeuroScriptParser::parse(Rule::shape, input);
    assert!(result.is_ok(), "Failed to parse shape: {:?}", result.err());
}

#[test]
fn test_parse_shape_with_wildcard() {
    let input = "[*, 512]";
    let result = NeuroScriptParser::parse(Rule::shape, input);
    assert!(
        result.is_ok(),
        "Failed to parse shape with wildcard: {:?}",
        result.err()
    );
}

#[test]
fn test_parse_shape_with_expr() {
    let input = "[dim * 4]";
    let result = NeuroScriptParser::parse(Rule::shape, input);
    assert!(
        result.is_ok(),
        "Failed to parse shape with expression: {:?}",
        result.err()
    );
}

#[test]
fn test_parse_use_stmt() {
    let input = "use core,nn/*";
    let result = NeuroScriptParser::parse(Rule::use_stmt, input);
    assert!(
        result.is_ok(),
        "Failed to parse use statement: {:?}",
        result.err()
    );
}

#[test]
fn test_parse_simple_neuron() {
    let input = r#"neuron Linear(in_dim, out_dim):
in:
default: [*, in_dim]
out:
default: [*, out_dim]
impl:
core,nn/Linear
"#;
    let result = NeuroScriptParser::parse(Rule::neuron_def, input);
    if let Err(e) = &result {
        eprintln!("Parse error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse simple neuron");
}

#[test]
fn test_parse_program() {
    let input = r#"use core,nn/*

neuron Linear(in_dim, out_dim):
in:
default: [*, in_dim]
out:
default: [*, out_dim]
impl:
core,nn/Linear
"#;
    let result = NeuroScriptParser::parse(Rule::program, input);
    if let Err(e) = &result {
        eprintln!("Parse error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse program");
}

#[test]
fn test_parse_residual_example() {
    let input = include_str!("../../examples/stdlib/residual.ns");
    let result = NeuroScriptParser::parse(Rule::program, input);
    if let Err(e) = &result {
        eprintln!("Parse error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse stdlib/residual.ns example");
}

// Test suite for example files
macro_rules! test_example_file {
    ($name:ident, $file:expr) => {
        #[test]
        fn $name() {
            let input = include_str!(concat!("../../examples/", $file));
            let result = NeuroScriptParser::parse(Rule::program, input);
            if let Err(e) = &result {
                eprintln!("Parse error in {}: {}", $file, e);
            }
            assert!(result.is_ok(), "Failed to parse {}", $file);
        }
    };
}

test_example_file!(test_primitives_basics, "primitives/basics.ns");
test_example_file!(test_primitives_activations, "primitives/activations.ns");
test_example_file!(test_primitives_structural, "primitives/structural.ns");
test_example_file!(test_primitives_operations, "primitives/operations.ns");
test_example_file!(test_primitives_attention, "primitives/attention.ns");
test_example_file!(test_stdlib_feedforward, "stdlib/feedforward.ns");
test_example_file!(test_stdlib_attention, "stdlib/attention.ns");
test_example_file!(test_stdlib_transformer_block, "stdlib/transformer_block.ns");
test_example_file!(test_tutorials_fork_join, "tutorials/02_fork_join.ns");
test_example_file!(test_match_multiline, "match_multiline.ns");
test_example_file!(test_real_world_resnet, "real_world/resnet.ns");
test_example_file!(test_dropout, "dropout.ns");
test_example_file!(test_unroll_threaded, "unroll_threaded.ns");
test_example_file!(test_unroll_context, "unroll_context.ns");
test_example_file!(test_unroll_static, "unroll_static.ns");
test_example_file!(test_unroll_gpt2, "unroll_gpt2.ns");

#[test]
fn test_multiple_connections_in_graph() {
    let input = r#"neuron Test:
in: [*shape]
out: [*shape]
graph:
in ->
    A()
    x

x ->
    B()
    out
"#;
    let result = NeuroScriptParser::parse(Rule::program, input);
    if let Err(e) = &result {
        eprintln!("Parse error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse multiple connections");
}

#[test]
fn test_parse_if_expr() {
    let input = "if has_pool: pool else: Identity() -> out";
    let result = NeuroScriptParser::parse(Rule::if_expr, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse if expr");
}

#[test]
fn test_if_is_not_ident() {
    let input = "if";
    let result = NeuroScriptParser::parse(Rule::ident, input);
    assert!(result.is_err(), "if should not parse as identifier");
}

#[test]
fn test_if_as_endpoint() {
    let input = "if true: pool";
    let result = NeuroScriptParser::parse(Rule::endpoint, input);
    assert!(result.is_ok(), "Failed to parse if as endpoint");
}

#[test]
fn test_branch_pipeline_with_arrow() {
    let input = "Identity() -> out";
    let result = NeuroScriptParser::parse(Rule::branch_pipeline, input);
    assert!(result.is_ok());
}

#[test]
fn test_parse_connection_with_if() {
    let input = "act -> if has_pool: pool else: Identity() -> out\n";
    let result = NeuroScriptParser::parse(Rule::connection, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok());
}

#[test]
fn test_parse_simple_neuron_with_if() {
    let input = r#"neuron Test:
    graph:
        act -> if has_pool: pool else: Identity() -> out
"#;
    let result = NeuroScriptParser::parse(Rule::neuron_def, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok());
}

#[test]
fn test_doc_comment_blank_line_before_neuron() {
    // Single blank line between doc comment and neuron keyword
    let input = r#"/// A documented neuron

neuron Test:
  in: [*, dim]
  out: [*, dim]
  impl: core,nn/Linear
"#;
    let result = NeuroScriptParser::parse(Rule::neuron_def, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse neuron_def with single blank line after doc comment");

    // Multiple blank lines between doc comment and neuron keyword
    let input_multi = r#"/// A documented neuron


neuron Test:
  in: [*, dim]
  out: [*, dim]
  impl: core,nn/Linear
"#;
    let result_multi = NeuroScriptParser::parse(Rule::neuron_def, input_multi);
    if let Err(e) = &result_multi {
        eprintln!("Error: {}", e);
    }
    assert!(result_multi.is_ok(), "Failed to parse neuron_def with multiple blank lines after doc comment");
}

#[test]
fn test_parse_variadic_input_port() {
    // Variadic input port syntax: in *inputs: [*shape]
    let input = r#"neuron Concat(dim):
    in *inputs: [*shape]
    out: [*shape_out]
    impl: core,structural/Concat
"#;
    let result = NeuroScriptParser::parse(Rule::neuron_def, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse neuron with variadic input port");
}

// ============================================================================
// Unroll grammar tests
// ============================================================================

#[test]
fn test_parse_named_unroll_context_block() {
    // Named context-level unroll block
    let input = "blocks = unroll(3):\n    block = TransformerBlock(d_model)\n";
    let result = NeuroScriptParser::parse(Rule::named_unroll_context_block, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse named unroll context block");
}

#[test]
fn test_parse_named_unroll_context_with_annotation() {
    // Named context-level unroll with @static annotation
    let input = "layers = unroll(num_layers):\n    @static block = TransformerBlock(d_model)\n";
    let result = NeuroScriptParser::parse(Rule::named_unroll_context_block, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse named unroll context block with @static");
}

#[test]
fn test_parse_named_unroll_context_with_param() {
    // Named context-level unroll with parameter reference
    let input = "blocks = unroll(num_layers):\n    block = Block(dim)\n";
    let result = NeuroScriptParser::parse(Rule::named_unroll_context_block, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse named unroll context block with param");
}

#[test]
fn test_unroll_is_not_ident() {
    // "unroll" should not parse as an identifier (it's a keyword)
    let input = "unroll";
    let result = NeuroScriptParser::parse(Rule::ident, input);
    assert!(result.is_err(), "unroll should not parse as identifier");
}

// ============================================================================
// Fat Arrow (=>) reshape tests
// ============================================================================

#[test]
fn test_parse_fat_arrow_grammar_rule() {
    // Test the fat_arrow_step grammar rule directly
    let input = "=> [batch, seq, heads, dh]";
    let result = NeuroScriptParser::parse(Rule::fat_arrow_step, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse fat_arrow_step");
}

#[test]
fn test_parse_fat_arrow_with_binding_grammar() {
    // Test fat_arrow_step with a binding dim
    let input = "=> [batch, seq, heads, dh=dim/heads]";
    let result = NeuroScriptParser::parse(Rule::fat_arrow_step, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse fat_arrow_step with binding");
}

#[test]
fn test_parse_fat_arrow_with_annotation_grammar() {
    // Test fat_arrow_step with @reduce annotation
    let input = "=> @reduce(mean) [b, c]";
    let result = NeuroScriptParser::parse(Rule::fat_arrow_step, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse fat_arrow_step with annotation");
}

#[test]
fn test_parse_fat_arrow_with_others_grammar() {
    // Test fat_arrow_step with 'others' keyword
    let input = "=> [b, others]";
    let result = NeuroScriptParser::parse(Rule::fat_arrow_step, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse fat_arrow_step with others");
}

#[test]
fn test_parse_reshape_dim_binding() {
    // Test reshape_dim with binding syntax
    let input = "hw=h*w";
    let result = NeuroScriptParser::parse(Rule::reshape_dim, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse reshape_dim binding");
}

#[test]
fn test_parse_reshape_expr_with_binding() {
    let input = "[b, c, hw=h*w]";
    let result = NeuroScriptParser::parse(Rule::reshape_expr, input);
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    assert!(result.is_ok(), "Failed to parse reshape_expr with binding");
}

#[test]
fn test_parse_fat_arrow_basic() {
    // Inline chained fat arrows with binding expression
    let source = r#"
neuron Reshape(dim, heads):
  in: [batch, seq, dim]
  out: [batch, heads, seq, dim / heads]
  graph:
    in => [batch, seq, heads, dh=dim/heads] => [batch, heads, seq, dh] -> out
"#;
    let program = crate::parse(source).expect("should parse");
    let neuron = program.neurons.get("Reshape").unwrap();
    if let crate::interfaces::NeuronBody::Graph { connections, .. } = &neuron.body {
        assert_eq!(connections.len(), 3, "expected 3 connections: in=>reshape, reshape=>reshape, reshape->out");
    } else {
        panic!("expected graph body");
    }
}

#[test]
fn test_parse_fat_arrow_with_annotation() {
    // Fat arrow with @reduce(mean) annotation
    let source = r#"
neuron Pool:
  in: [b, c, h, w]
  out: [b, c]
  graph:
    in => @reduce(mean) [b, c] -> out
"#;
    let program = crate::parse(source).expect("should parse");
    let neuron = program.neurons.get("Pool").unwrap();
    if let crate::interfaces::NeuronBody::Graph { connections, .. } = &neuron.body {
        assert_eq!(connections.len(), 2, "expected 2 connections: in=>reshape, reshape->out");
        // Verify the first connection's destination is a Reshape with annotation
        match &connections[0].destination {
            crate::interfaces::Endpoint::Reshape(expr) => {
                assert!(expr.annotation.is_some(), "expected annotation on reshape");
                match expr.annotation.as_ref().unwrap() {
                    crate::interfaces::TransformAnnotation::Reduce(strategy, _) => {
                        match strategy {
                            crate::interfaces::TransformStrategy::Intrinsic(name) => {
                                assert_eq!(name, "mean");
                            }
                            _ => panic!("expected intrinsic strategy"),
                        }
                    }
                    _ => panic!("expected Reduce annotation"),
                }
            }
            _ => panic!("expected Reshape endpoint"),
        }
    } else {
        panic!("expected graph body");
    }
}

#[test]
fn test_parse_fat_arrow_indented() {
    // Fat arrows in indented pipeline with binding expression
    let source = r#"
neuron VitFlatten:
  in: [b, c, h, w]
  out: [b, seq, c]
  graph:
    in ->
      Linear(512, 256)
      => [b, c, hw=h*w]
      => [b, hw, c]
      out
"#;
    let program = crate::parse(source).expect("should parse");
    let neuron = program.neurons.get("VitFlatten").unwrap();
    if let crate::interfaces::NeuronBody::Graph { connections, .. } = &neuron.body {
        assert!(connections.len() >= 4, "expected at least 4 connections, got {}", connections.len());
    } else {
        panic!("expected graph body");
    }
}

#[test]
fn test_parse_fat_arrow_others() {
    // Fat arrow with 'others' keyword for flattening
    let source = r#"
neuron Flatten:
  in: [b, c, h, w]
  out: [b, flat]
  graph:
    in => [b, others] -> out
"#;
    let program = crate::parse(source).expect("should parse");
    let neuron = program.neurons.get("Flatten").unwrap();
    if let crate::interfaces::NeuronBody::Graph { connections, .. } = &neuron.body {
        assert_eq!(connections.len(), 2, "expected 2 connections: in=>reshape, reshape->out");
        // Verify the reshape has 'others' dim
        match &connections[0].destination {
            crate::interfaces::Endpoint::Reshape(expr) => {
                assert_eq!(expr.dims.len(), 2, "expected 2 dims in reshape");
                assert_eq!(expr.dims[1], crate::interfaces::ReshapeDim::Others);
            }
            _ => panic!("expected Reshape endpoint"),
        }
    } else {
        panic!("expected graph body");
    }
}

#[test]
fn test_parse_fat_arrow_with_neuron_call_annotation() {
    // Fat arrow with @reduce(NeuronCall(args)) annotation
    let source = r#"
neuron CustomPool(dim):
  in: [b, c, h, w]
  out: [b, c]
  graph:
    in => @reduce(AttentionPool(dim)) [b, c] -> out
"#;
    let program = crate::parse(source).expect("should parse");
    let neuron = program.neurons.get("CustomPool").unwrap();
    if let crate::interfaces::NeuronBody::Graph { connections, .. } = &neuron.body {
        match &connections[0].destination {
            crate::interfaces::Endpoint::Reshape(expr) => {
                match expr.annotation.as_ref().unwrap() {
                    crate::interfaces::TransformAnnotation::Reduce(strategy, _) => {
                        match strategy {
                            crate::interfaces::TransformStrategy::Neuron { name, args, .. } => {
                                assert_eq!(name, "AttentionPool");
                                assert_eq!(args.len(), 1);
                            }
                            _ => panic!("expected Neuron strategy"),
                        }
                    }
                    _ => panic!("expected Reduce annotation"),
                }
            }
            _ => panic!("expected Reshape endpoint"),
        }
    } else {
        panic!("expected graph body");
    }
}

#[test]
fn test_parse_wrap_ref() {
    let source = r#"
neuron Test(dim):
    in: [*, dim]
    out: [*, dim]
    context:
        attn = MultiHeadSelfAttention(dim, 8)
    graph:
        in -> @wrap(HyperConnect, 4, dim, 0): attn -> out
"#;
    let program = crate::parse(source).expect("should parse @wrap ref form");
    let neuron = program.neurons.get("Test").unwrap();
    if let crate::interfaces::NeuronBody::Graph { connections, .. } = &neuron.body {
        assert!(!connections.is_empty(), "Should have connections");
        // The @wrap endpoint should appear as an Endpoint::Wrap in the connections
        let has_wrap = connections.iter().any(|c| {
            matches!(&c.source, crate::interfaces::Endpoint::Wrap(_))
                || matches!(&c.destination, crate::interfaces::Endpoint::Wrap(_))
        });
        assert!(has_wrap, "Should contain a Wrap endpoint");
        // Check wrapper name and args
        for conn in connections {
            if let crate::interfaces::Endpoint::Wrap(wrap_expr) = &conn.source {
                assert_eq!(wrap_expr.wrapper_name, "HyperConnect");
                assert_eq!(wrap_expr.wrapper_args.len(), 3); // 4, dim, 0
                assert!(matches!(&wrap_expr.content, crate::interfaces::WrapContent::Ref(name) if name == "attn"));
                break;
            }
            if let crate::interfaces::Endpoint::Wrap(wrap_expr) = &conn.destination {
                assert_eq!(wrap_expr.wrapper_name, "HyperConnect");
                assert_eq!(wrap_expr.wrapper_args.len(), 3); // 4, dim, 0
                assert!(matches!(&wrap_expr.content, crate::interfaces::WrapContent::Ref(name) if name == "attn"));
                break;
            }
        }
    } else {
        panic!("Expected Graph body");
    }
}

#[test]
fn test_parse_wrap_pipeline() {
    let source = r#"
neuron Test(dim):
    in: [*, dim]
    out: [*, dim]
    graph:
        in ->
            @wrap(HyperConnect, 4, dim, 0): ->
                LayerNorm(dim)
                Linear(dim, dim)
            out
"#;
    let program = crate::parse(source).expect("should parse @wrap pipeline form");
    let neuron = program.neurons.get("Test").unwrap();
    if let crate::interfaces::NeuronBody::Graph { connections, .. } = &neuron.body {
        assert!(!connections.is_empty(), "Should have connections");
        // Find the wrap endpoint
        let has_wrap = connections.iter().any(|c| {
            matches!(&c.source, crate::interfaces::Endpoint::Wrap(_))
                || matches!(&c.destination, crate::interfaces::Endpoint::Wrap(_))
        });
        assert!(has_wrap, "Should contain a Wrap endpoint");
        // Check wrapper details
        for conn in connections {
            if let crate::interfaces::Endpoint::Wrap(wrap_expr) = &conn.source {
                assert_eq!(wrap_expr.wrapper_name, "HyperConnect");
                assert!(matches!(&wrap_expr.content, crate::interfaces::WrapContent::Pipeline(p) if !p.is_empty()));
                break;
            }
            if let crate::interfaces::Endpoint::Wrap(wrap_expr) = &conn.destination {
                assert_eq!(wrap_expr.wrapper_name, "HyperConnect");
                assert!(matches!(&wrap_expr.content, crate::interfaces::WrapContent::Pipeline(p) if !p.is_empty()));
                break;
            }
        }
    } else {
        panic!("Expected Graph body");
    }
}

#[test]
fn test_parse_wrap_inline_pipeline() {
    let source = r#"
neuron Test(dim):
    in: [*, dim]
    out: [*, dim]
    graph:
        in -> @wrap(HyperConnect, 4, dim, 0): -> LayerNorm(dim) -> out
"#;
    let program = crate::parse(source).expect("should parse @wrap inline pipeline form");
    let neuron = program.neurons.get("Test").unwrap();
    if let crate::interfaces::NeuronBody::Graph { connections, .. } = &neuron.body {
        assert!(!connections.is_empty(), "Should have connections");
    } else {
        panic!("Expected Graph body");
    }
}

#[test]
fn test_parse_wrap_grammar_rule() {
    // Test the wrap_endpoint grammar rule directly
    let input = "@wrap(HyperConnect, 4, dim): foo";
    let result = NeuroScriptParser::parse(Rule::wrap_endpoint, input);
    assert!(
        result.is_ok(),
        "Failed to parse wrap_endpoint: {:?}",
        result.err()
    );
}
