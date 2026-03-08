//! AST Builder for pest-based parser
//!
//! Converts pest parse trees (Pair<Rule>) into NeuroScript IR types.
//! Handles indentation validation during the conversion.
//!
//! # Safety of `.unwrap()` calls
//!
//! This module contains many `.unwrap()` calls on `inner.next()` (pest `Pairs` iterators).
//! These are safe because the PEG grammar (`neuroscript.pest`) guarantees the required
//! children exist — pest will not produce a successful parse tree missing mandatory
//! sub-rules. Each `unwrap()` corresponds to a mandatory child in the grammar rule
//! being destructured.

use miette::SourceSpan;
use pest::iterators::Pair;

use crate::doc_parser;
use crate::grammar::error;
use crate::grammar::Rule;
use crate::interfaces::{
    BinOp, Binding, Connection, ContextUnroll, Dim, DimExpr, Documentation, Endpoint,
    GlobalBinding, IdGenerator, ImplRef, MatchArm, MatchExpr, MatchPattern, MatchSubject,
    NeuronBody, NeuronDef, NeuronPortContract, Param, ParseError, Port, PortRef, Program,
    ReshapeDim, ReshapeExpr, Shape, TransformAnnotation, TransformStrategy, UseStmt, Value,
    WrapContent, WrapExpr,
};
use crate::interfaces::{CallArgs, CallExpr, Kwarg};

/// AST builder state
pub struct AstBuilder {
    /// Shared ID generator for globally unique endpoint IDs
    id_gen: IdGenerator,
}

/// Temporary state used during neuron construction
#[derive(Default)]
struct NeuronBuilderState {
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    context_bindings: Vec<Binding>,
    context_unrolls: Vec<ContextUnroll>,
    connections: Vec<Connection>,
    impl_ref: Option<ImplRef>,
}

impl AstBuilder {
    pub fn new() -> Self {
        AstBuilder {
            id_gen: IdGenerator::new(),
        }
    }

    fn next_id(&mut self) -> usize {
        self.id_gen.next_id()
    }

    /// Return the current ID counter value so later passes can continue from it.
    pub fn id_counter(&self) -> usize {
        self.id_gen.current()
    }

    /// Build a Program from a pest parse tree
    pub fn build_program(&mut self, pair: Pair<Rule>) -> Result<Program, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::program);

        let mut program = Program::new();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::use_stmt => {
                    let use_stmt = self.build_use_stmt(inner)?;
                    program.uses.push(use_stmt);
                }
                Rule::global_decl => {
                    let global = self.build_global_decl(inner)?;
                    program.globals.push(global);
                }
                Rule::neuron_def => {
                    let neuron = self.build_neuron_def(inner)?;
                    let offset = 0; // TODO: track proper offset
                    if program.neurons.contains_key(&neuron.name) {
                        return Err(error::duplicate_neuron(&neuron.name, offset));
                    }
                    program.neurons.insert(neuron.name.clone(), neuron);
                }
                Rule::EOI => {}
                Rule::NEWLINE => {}
                _ => {}
            }
        }

        Ok(program)
    }

    /// Build a GlobalBinding from a global_decl pair
    fn build_global_decl(&mut self, pair: Pair<Rule>) -> Result<GlobalBinding, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::global_decl);

        let mut inner = pair.into_inner();
        inner.next(); // Skip at
        inner.next(); // Skip keyword_global

        let content = inner.next().unwrap();
        match content.as_rule() {
            Rule::binding => {
                let binding = self.build_binding(content)?;
                // Convert Binding to GlobalBinding (if it's a call)
                Ok(GlobalBinding {
                    name: binding.name,
                    value: Value::Call {
                        name: binding.call_name,
                        args: binding.args,
                        kwargs: binding.kwargs,
                    },
                })
            }
            Rule::global_value_binding => {
                let mut v_inner = content.into_inner();
                let name = self.extract_ident(v_inner.next().unwrap())?;
                v_inner.next(); // Skip assign
                let value = self.build_value(v_inner.next().unwrap())?;
                Ok(GlobalBinding { name, value })
            }
            _ => unreachable!(),
        }
    }

    /// Build a UseStmt from a use_stmt pair
    fn build_use_stmt(&mut self, pair: Pair<Rule>) -> Result<UseStmt, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::use_stmt);

        let mut inner = pair.into_inner();

        // Skip keyword_use
        inner.next();

        // Get source identifier
        let source = self.extract_ident(inner.next().unwrap())?;

        // Skip comma
        inner.next();

        // Get path
        let path_pair = inner.next().unwrap();
        let path = self.build_impl_path(path_pair)?;

        Ok(UseStmt { source, path })
    }

    /// Build an impl path (e.g., "nn/Linear" -> ["nn", "Linear"])
    fn build_impl_path(&mut self, pair: Pair<Rule>) -> Result<Vec<String>, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::impl_path);

        let mut path = vec![];
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::ident => path.push(inner.as_str().to_string()),
                Rule::star => path.push("*".to_string()),
                Rule::slash => {}
                _ => {}
            }
        }
        Ok(path)
    }

    /// Build a NeuronDef from a neuron_def pair
    fn build_neuron_def(&mut self, pair: Pair<Rule>) -> Result<NeuronDef, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::neuron_def);

        let mut inner = pair.into_inner();

        // Check for optional doc_block
        let mut doc: Option<Documentation> = None;
        let mut next = inner.next();

        if let Some(ref p) = next {
            if p.as_rule() == Rule::doc_block {
                doc = Some(self.build_doc_block(p.clone())?);
                next = inner.next();
            }
        }

        // Skip any blank lines (NEWLINEs) between doc_block and keyword_neuron
        while let Some(ref p) = next {
            if p.as_rule() == Rule::NEWLINE {
                next = inner.next();
            } else {
                break;
            }
        }

        // Skip keyword_neuron (it's in 'next' if no doc_block, or we already moved past it)
        if let Some(ref p) = next {
            if p.as_rule() == Rule::keyword_neuron {
                next = inner.next(); // Move to ident
            }
        }

        // Get name
        let name = self.extract_ident(next.unwrap())?;

        // Check for params (optional)
        let mut params = vec![];
        let mut next = inner.next();

        if let Some(ref p) = next {
            if p.as_rule() == Rule::params {
                params = self.build_params(p.clone())?;
                next = inner.next();
            }
        }

        // Skip colon and newlines, process sections
        let mut state = NeuronBuilderState::default();

        // Process remaining elements (sections)
        while let Some(p) = next {
            if p.as_rule() == Rule::neuron_section {
                self.process_neuron_section(p, &mut state)?;
            }
            next = inner.next();
        }

        // Construct body
        let body = if let Some(impl_ref_val) = state.impl_ref {
            NeuronBody::Primitive(impl_ref_val)
        } else {
            NeuronBody::Graph {
                context_bindings: state.context_bindings,
                context_unrolls: state.context_unrolls,
                connections: state.connections,
            }
        };

        // Set max_cycle_depth based on body type
        let max_cycle_depth = match &body {
            NeuronBody::Graph { .. } => Some(10),
            NeuronBody::Primitive(_) => None,
        };

        Ok(NeuronDef {
            name,
            params,
            inputs: state.inputs,
            outputs: state.outputs,
            body,
            max_cycle_depth,
            doc,
        })
    }

    /// Build Documentation from a doc_block pair
    fn build_doc_block(&mut self, pair: Pair<Rule>) -> Result<Documentation, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::doc_block);

        let mut doc_lines = Vec::new();
        let span_start = pair.as_span().start();
        let span_end = pair.as_span().end();

        // Extract all DOC_COMMENT lines
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::DOC_COMMENT {
                doc_lines.push(inner.as_str().to_string());
            }
        }

        // Parse the doc comments using our doc_parser
        let span = Some((span_start, span_end - span_start).into());
        Ok(doc_parser::parse_doc_comments(doc_lines, span))
    }

    /// Process a neuron_section and update the appropriate vectors
    fn process_neuron_section(
        &mut self,
        pair: Pair<Rule>,
        state: &mut NeuronBuilderState,
    ) -> Result<(), ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::neuron_section);

        let section = pair.into_inner().next().unwrap();

        match section.as_rule() {
            Rule::in_section => {
                let ports = self.build_in_section(section)?;
                state.inputs.extend(ports);
            }
            Rule::out_section => {
                let ports = self.build_out_section(section)?;
                state.outputs.extend(ports);
            }
            Rule::context_section => {
                let (bindings, unrolls) = self.build_context_section(section)?;
                state.context_bindings.extend(bindings);
                state.context_unrolls.extend(unrolls);
            }
            Rule::graph_section => {
                let conns = self.build_graph_section(section)?;
                state.connections.extend(conns);
            }
            Rule::impl_section => {
                state.impl_ref = Some(self.build_impl_section(section)?);
            }
            _ => {}
        }

        Ok(())
    }

    /// Build params from a params pair
    fn build_params(&mut self, pair: Pair<Rule>) -> Result<Vec<Param>, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::params);

        let mut params = vec![];

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::param {
                params.push(self.build_param(inner)?);
            }
        }

        Ok(params)
    }

    /// Build a single param
    fn build_param(&mut self, pair: Pair<Rule>) -> Result<Param, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::param);

        let mut inner = pair.into_inner();
        let name = self.extract_ident(inner.next().unwrap())?;

        let mut default = None;
        let mut type_annotation = None;
        for p in inner {
            match p.as_rule() {
                Rule::type_annotation => {
                    let type_name = p.as_str();
                    type_annotation = Some(match type_name {
                        "Neuron" | "NeuronType" => crate::interfaces::ParamType::Neuron,
                        _ => crate::interfaces::ParamType::Value,
                    });
                }
                Rule::value => {
                    default = Some(self.build_value(p)?);
                }
                _ => {}
            }
        }

        Ok(Param { name, default, type_annotation })
    }

    /// Build ports from in_section
    fn build_in_section(&mut self, pair: Pair<Rule>) -> Result<Vec<Port>, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::in_section);
        self.build_port_section(pair)
    }

    /// Build ports from out_section
    fn build_out_section(&mut self, pair: Pair<Rule>) -> Result<Vec<Port>, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::out_section);
        self.build_port_section(pair)
    }

    /// Build ports from a port section (in or out)
    fn build_port_section(&mut self, pair: Pair<Rule>) -> Result<Vec<Port>, ParseError> {
        let mut ports = vec![];
        let mut name: Option<String> = None;
        let mut is_variadic = false;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::star => {
                    is_variadic = true;
                }
                Rule::port_def => {
                    ports.push(self.build_port_def(inner)?);
                }
                Rule::shape => {
                    // Inline shape without name (e.g., "in: [shape]")
                    // Or shape following an ident
                    let s = self.build_shape(inner)?;
                    if let Some(n) = name.take() {
                        // We had an ident before, this is "in name: [shape]"
                        ports.push(Port { name: n, shape: s, variadic: is_variadic });
                        is_variadic = false;
                    } else {
                        ports.push(Port {
                            name: "default".to_string(),
                            shape: s,
                            variadic: is_variadic,
                        });
                        is_variadic = false;
                    }
                }
                Rule::ident => {
                    // Named inline port - collect name, shape comes next
                    name = Some(inner.as_str().to_string());
                }
                _ => {}
            }
        }

        Ok(ports)
    }

    /// Build a port_def (name: [shape])
    fn build_port_def(&mut self, pair: Pair<Rule>) -> Result<Port, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::port_def);

        let mut inner = pair.into_inner();
        let name = self.extract_ident(inner.next().unwrap())?;

        // Skip colon
        inner.next();

        let shape = self.build_shape(inner.next().unwrap())?;

        Ok(Port { name, shape, variadic: false })
    }

    /// Build a Shape from a shape pair
    fn build_shape(&mut self, pair: Pair<Rule>) -> Result<Shape, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::shape);

        let mut dims = vec![];

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::dim {
                dims.push(self.build_dim(inner)?);
            }
        }

        Ok(Shape { dims })
    }

    /// Build a Dim from a dim pair
    fn build_dim(&mut self, pair: Pair<Rule>) -> Result<Dim, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::dim);

        let inner: Vec<_> = pair.into_inner().collect();

        if inner.is_empty() {
            return Ok(Dim::Wildcard);
        }

        // Check for binary expression (primary op primary)
        if inner.len() >= 3 {
            let left = self.build_dim_primary(inner[0].clone())?;
            let op = self.build_dim_op(inner[1].clone())?;
            let right = self.build_dim_primary(inner[2].clone())?;

            return Ok(Dim::Expr(Box::new(DimExpr { op, left, right })));
        }

        // Single primary
        self.build_dim_primary(inner[0].clone())
    }

    /// Build a dimension primary
    fn build_dim_primary(&mut self, pair: Pair<Rule>) -> Result<Dim, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::dim_primary);

        let inner: Vec<_> = pair.into_inner().collect();

        if inner.is_empty() {
            return Ok(Dim::Wildcard);
        }

        let first = &inner[0];

        match first.as_rule() {
            // Handle parenthesized expressions: lparen ~ dim ~ rparen
            Rule::lparen => {
                let dim_pair = &inner[1];
                self.build_dim(dim_pair.clone())
            }
            Rule::star => {
                // Check for variadic: *name
                if inner.len() > 1 && inner[1].as_rule() == Rule::ident {
                    Ok(Dim::Variadic(inner[1].as_str().to_string()))
                } else {
                    Ok(Dim::Wildcard)
                }
            }
            Rule::integer => {
                let n: i64 = first.as_str().parse().unwrap_or(0);
                Ok(Dim::Literal(n))
            }
            Rule::ident => Ok(Dim::Named(first.as_str().to_string())),
            Rule::global_ref => {
                let mut inner = first.clone().into_inner();
                // Skip '@', 'global'
                inner.next();
                inner.next();
                let ident = inner.next().unwrap().as_str().to_string();
                Ok(Dim::Global(ident))
            }
            Rule::dim => self.build_dim(first.clone()),
            _ => Ok(Dim::Wildcard),
        }
    }

    /// Build a dimension operator
    fn build_dim_op(&mut self, pair: Pair<Rule>) -> Result<BinOp, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::dim_op);

        let inner = pair.into_inner().next().unwrap();

        match inner.as_rule() {
            Rule::plus => Ok(BinOp::Add),
            Rule::minus => Ok(BinOp::Sub),
            Rule::star => Ok(BinOp::Mul),
            Rule::slash => Ok(BinOp::Div),
            _ => Ok(BinOp::Add),
        }
    }

    /// Build bindings and unroll blocks from context_section
    fn build_context_section(
        &mut self,
        pair: Pair<Rule>,
    ) -> Result<(Vec<Binding>, Vec<ContextUnroll>), ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::context_section);

        let mut bindings = vec![];
        let mut unrolls = vec![];

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::context_binding => {
                    bindings.push(self.build_context_binding(inner)?);
                }
                Rule::named_unroll_context_block => {
                    // The PEG grammar greedily matches all subsequent context_bindings
                    // into the unroll block. Use indentation to separate bindings that
                    // belong inside the unroll from those that follow it at the same level.
                    let (unroll, overflow) =
                        self.build_named_unroll_context_block(inner)?;
                    unrolls.push(unroll);
                    bindings.extend(overflow);
                }
                _ => {}
            }
        }

        Ok((bindings, unrolls))
    }

    /// Build a named unroll context block: `name = unroll(count):\n bindings...`
    /// Returns overflow bindings that belong to the parent context section
    /// (at shallower indentation).
    fn build_named_unroll_context_block(
        &mut self,
        pair: Pair<Rule>,
    ) -> Result<(ContextUnroll, Vec<Binding>), ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::named_unroll_context_block);

        // Get the column of the name to establish the baseline
        let unroll_col = pair.as_span().start_pos().line_col().1;

        let mut inner = pair.into_inner();

        // Parse: ident = unroll(count):
        let aggregate_name = inner.next().unwrap().as_str().to_string(); // ident
        inner.next(); // Skip assign (=)
        inner.next(); // Skip keyword_unroll
        inner.next(); // Skip lparen

        let count = self.build_value(inner.next().unwrap())?;

        // Skip rparen, colon
        inner.next();
        inner.next();

        let mut unroll_bindings = vec![];
        let mut overflow_bindings = vec![];

        for p in inner {
            match p.as_rule() {
                Rule::context_binding => {
                    let col = p.as_span().start_pos().line_col().1;

                    if col > unroll_col {
                        // More indented than unroll keyword → belongs inside
                        unroll_bindings.push(self.build_context_binding(p)?);
                    } else {
                        // Same or less indentation → overflow to parent
                        overflow_bindings.push(self.build_context_binding(p)?);
                    }
                }
                Rule::NEWLINE => {}
                _ => {}
            }
        }

        Ok((
            ContextUnroll {
                aggregate_name,
                count,
                bindings: unroll_bindings,
            },
            overflow_bindings,
        ))
    }

    /// Build a context binding with optional annotation
    fn build_context_binding(&mut self, pair: Pair<Rule>) -> Result<Binding, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::context_binding);

        let mut inner = pair.into_inner();
        let mut first = inner.next().unwrap();
        let mut scope = crate::interfaces::Scope::Instance { lazy: false };

        // Check for annotation
        if first.as_rule() == Rule::binding_annotation {
            scope = self.build_binding_annotation(first)?;
            first = inner.next().unwrap(); // Move to binding
        }

        // Build the binding
        let mut binding = self.build_binding(first)?;
        binding.scope = scope;

        Ok(binding)
    }

    /// Build a binding annotation
    fn build_binding_annotation(
        &mut self,
        pair: Pair<Rule>,
    ) -> Result<crate::interfaces::Scope, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::binding_annotation);

        let mut inner = pair.into_inner();
        inner.next(); // Skip 'at'

        let keyword = inner.next().unwrap();
        match keyword.as_rule() {
            Rule::keyword_static => Ok(crate::interfaces::Scope::Static),
            Rule::keyword_global => Ok(crate::interfaces::Scope::Global),
            Rule::keyword_lazy => Ok(crate::interfaces::Scope::Instance { lazy: true }),
            _ => Ok(crate::interfaces::Scope::Instance { lazy: false }),
        }
    }

    /// Build a binding (name = neuron_expr)
    fn build_binding(&mut self, pair: Pair<Rule>) -> Result<Binding, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::binding);

        let mut inner = pair.into_inner();
        let name = self.extract_ident(inner.next().unwrap())?;

        // Skip assign
        inner.next();

        let neuron_expr = inner.next().unwrap();
        let (call_name, args, kwargs, frozen) = self.build_neuron_expr(neuron_expr)?;

        Ok(Binding {
            name,
            call_name,
            args,
            kwargs,
            scope: crate::interfaces::Scope::Instance { lazy: false },
            frozen,
            unroll_group: None,
        })
    }

    /// Build a neuron expression (Call or Freeze or Ref)
    /// Returns (name, args, kwargs, frozen)
    fn build_neuron_expr(
        &mut self,
        pair: Pair<Rule>,
    ) -> Result<(String, Vec<Value>, Vec<Kwarg>, bool), ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::neuron_expr);

        let inner = pair.into_inner().next().unwrap();
        match inner.as_rule() {
            Rule::call_expr => {
                let (name, args, kwargs) = self.build_call_expr(inner)?;
                Ok((name, args, kwargs, false))
            }
            Rule::ident => {
                let name = inner.as_str().to_string();
                Ok((name, vec![], vec![], false))
            }
            Rule::neuron_expr => {
                // This is the recursive Freeze(...) case
                // Rule: keyword_freeze ~ lparen ~ neuron_expr ~ rparen
                // Note: pair.into_inner() above already gave us the First alternative or Second.
                // If it peaked neuron_expr again, it's the Freeze case.

                // Wait, if it's keyword_freeze, it's the first alternative.
                // Let's re-examine neuroscript.pest:
                // neuron_expr = { (keyword_freeze ~ lparen ~ neuron_expr ~ rparen) | call_expr }

                let mut f_inner = inner.into_inner();
                f_inner.next(); // Skip keyword_freeze
                f_inner.next(); // Skip lparen
                let sub_expr = f_inner.next().unwrap();
                let (sub_name, sub_args, sub_kwargs, _) = self.build_neuron_expr(sub_expr)?;

                // Wrap in Freeze call
                Ok((
                    "Freeze".to_string(),
                    vec![Value::Call {
                        name: sub_name,
                        args: sub_args,
                        kwargs: sub_kwargs,
                    }],
                    vec![],
                    true, // Mark as frozen
                ))
            }
            _ => unreachable!("Unexpected rule in neuron_expr: {:?}", inner.as_rule()),
        }
    }

    /// Build a call expression, returning (name, args, kwargs)
    fn build_call_expr(&mut self, pair: Pair<Rule>) -> Result<CallExpr, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::call_expr);

        let mut inner = pair.into_inner();
        let name = self.extract_ident(inner.next().unwrap())?;

        let mut args = vec![];
        let mut kwargs = vec![];

        // Skip lparen
        inner.next();

        // Parse call_args if present
        if let Some(args_pair) = inner.next() {
            if args_pair.as_rule() == Rule::call_args {
                (args, kwargs) = self.build_call_args(args_pair)?;
            }
        }

        Ok((name, args, kwargs))
    }

    fn build_call_args(&mut self, pair: Pair<Rule>) -> Result<CallArgs, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::call_args);

        let mut args = vec![];
        let mut kwargs = vec![];

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::kwarg => {
                    let (name, value) = self.build_kwarg(inner)?;
                    kwargs.push((name, value));
                }
                Rule::value => {
                    args.push(self.build_value(inner)?);
                }
                _ => {}
            }
        }

        Ok((args, kwargs))
    }

    /// Build a keyword argument
    fn build_kwarg(&mut self, pair: Pair<Rule>) -> Result<Kwarg, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::kwarg);

        let mut inner = pair.into_inner();
        let name = self.extract_ident(inner.next().unwrap())?;

        // Skip assign
        inner.next();

        let value = self.build_value(inner.next().unwrap())?;

        Ok((name, value))
    }

    /// Build impl_section
    fn build_impl_section(&mut self, pair: Pair<Rule>) -> Result<ImplRef, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::impl_section);

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::impl_ref {
                return self.build_impl_ref(inner);
            }
        }

        Err(error::expected("impl reference", "nothing", 0))
    }

    /// Build an impl reference
    fn build_impl_ref(&mut self, pair: Pair<Rule>) -> Result<ImplRef, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::impl_ref);

        let mut inner = pair.into_inner();
        let first = inner.next().unwrap();

        match first.as_rule() {
            Rule::keyword_external => {
                // external(kwargs)
                let mut kwargs = vec![];

                // Skip lparen
                inner.next();

                if let Some(args_pair) = inner.next() {
                    if args_pair.as_rule() == Rule::call_args {
                        let (_, kw) = self.build_call_args(args_pair)?;
                        kwargs = kw;
                    }
                }

                Ok(ImplRef::External { kwargs })
            }
            Rule::ident => {
                // source,path
                let source = first.as_str().to_string();

                // Skip comma
                inner.next();

                let path_pair = inner.next().unwrap();
                let path_parts = self.build_impl_path(path_pair)?;

                Ok(ImplRef::Source {
                    source,
                    path: path_parts.join("/"),
                })
            }
            _ => Err(error::expected("impl reference", first.as_str(), 0)),
        }
    }

    /// Build graph_section
    fn build_graph_section(&mut self, pair: Pair<Rule>) -> Result<Vec<Connection>, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::graph_section);

        let mut connections = vec![];

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::connection {
                let conns = self.build_connection(inner)?;
                connections.extend(conns);
            }
        }

        Ok(connections)
    }

    /// Build a connection (possibly a pipeline with multiple connections)
    fn build_connection(&mut self, pair: Pair<Rule>) -> Result<Vec<Connection>, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::connection);

        let mut inner = pair.into_inner();

        // First endpoint
        let first_endpoint = self.build_endpoint(inner.next().unwrap())?;

        // Second pair is either arrow (for connection_tail) or fat_arrow_step
        let second = inner.next().unwrap();
        match second.as_rule() {
            Rule::arrow => {
                // connection_tail
                let tail = inner.next().unwrap();
                self.build_connection_tail(first_endpoint, tail)
            }
            Rule::fat_arrow_step => {
                // fat_arrow_step ~ connection_tail_after_reshape
                let reshape = self.build_fat_arrow_step(second)?;
                let mut connections = vec![Connection {
                    source: first_endpoint,
                    destination: reshape.clone(),
                }];

                // connection_tail_after_reshape
                if let Some(tail) = inner.next() {
                    let tail_conns =
                        self.build_connection_tail_after_reshape(reshape, tail)?;
                    connections.extend(tail_conns);
                }

                Ok(connections)
            }
            rule => {
                unreachable!(
                    "build_connection: grammar guarantees second pair is arrow or fat_arrow_step, got {:?}",
                    rule
                );
            }
        }
    }

    /// Build the tail of a connection (inline or indented pipeline)
    fn build_connection_tail(
        &mut self,
        first: Endpoint,
        pair: Pair<Rule>,
    ) -> Result<Vec<Connection>, ParseError> {
        // Handle different rule types - grammar may emit COMMENT between connection parts
        if pair.as_rule() != Rule::connection_tail {
            // Not a connection tail - might be end of connection
            return Ok(vec![]);
        }

        let mut connections = vec![];
        let mut prev = first;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::endpoint => {
                    let next = self.build_endpoint(inner)?;
                    connections.push(Connection {
                        source: prev,
                        destination: next.clone(),
                    });
                    prev = next;
                }
                Rule::fat_arrow_step => {
                    let next = self.build_fat_arrow_step(inner)?;
                    connections.push(Connection {
                        source: prev,
                        destination: next.clone(),
                    });
                    prev = next;
                }
                Rule::indented_pipeline => {
                    let pipeline_conns = self.build_indented_pipeline(prev, inner)?;
                    connections.extend(pipeline_conns);
                    return Ok(connections);
                }
                _ => {}
            }
        }

        Ok(connections)
    }

    /// Build the tail after a reshape step (connection_tail_after_reshape)
    fn build_connection_tail_after_reshape(
        &mut self,
        first: Endpoint,
        pair: Pair<Rule>,
    ) -> Result<Vec<Connection>, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::connection_tail_after_reshape);

        let mut connections = vec![];
        let mut prev = first;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::endpoint => {
                    let next = self.build_endpoint(inner)?;
                    connections.push(Connection {
                        source: prev,
                        destination: next.clone(),
                    });
                    prev = next;
                }
                Rule::fat_arrow_step => {
                    let next = self.build_fat_arrow_step(inner)?;
                    connections.push(Connection {
                        source: prev,
                        destination: next.clone(),
                    });
                    prev = next;
                }
                Rule::indented_pipeline => {
                    let pipeline_conns = self.build_indented_pipeline(prev, inner)?;
                    connections.extend(pipeline_conns);
                    return Ok(connections);
                }
                _ => {}
            }
        }

        Ok(connections)
    }

    /// Build an indented pipeline
    fn build_indented_pipeline(
        &mut self,
        first: Endpoint,
        pair: Pair<Rule>,
    ) -> Result<Vec<Connection>, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::indented_pipeline);

        let mut connections = vec![];
        let mut prev = first;

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::indented_pipeline_item {
                for item_inner in inner.into_inner() {
                    match item_inner.as_rule() {
                        Rule::endpoint => {
                            let next = self.build_endpoint(item_inner)?;
                            connections.push(Connection {
                                source: prev,
                                destination: next.clone(),
                            });
                            prev = next;
                        }
                        Rule::fat_arrow_step => {
                            let next = self.build_fat_arrow_step(item_inner)?;
                            connections.push(Connection {
                                source: prev,
                                destination: next.clone(),
                            });
                            prev = next;
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(connections)
    }

    /// Build an endpoint
    fn build_endpoint(&mut self, pair: Pair<Rule>) -> Result<Endpoint, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::endpoint);

        let inner = pair.into_inner().next().unwrap();

        match inner.as_rule() {
            Rule::match_eval_expr => Ok(Endpoint::Match(self.build_match_eval_expr(inner)?)),
            Rule::match_expr => Ok(Endpoint::Match(self.build_match_expr(inner)?)),
            Rule::if_expr => Ok(Endpoint::If(self.build_if_expr(inner)?)),
            Rule::wrap_endpoint => self.build_wrap_endpoint(inner),
            Rule::tuple_endpoint => self.build_tuple_endpoint(inner),
            Rule::call_endpoint => self.build_call_endpoint(inner),
            Rule::ref_endpoint => self.build_ref_endpoint(inner),
            _ => Err(error::expected("endpoint", inner.as_str(), 0)),
        }
    }

    /// Build an if expression
    fn build_if_expr(&mut self, pair: Pair<Rule>) -> Result<crate::interfaces::IfExpr, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::if_expr);

        let mut branches = vec![];
        let mut else_branch = None;

        let mut inner = pair.into_inner();

        // if condition : pipeline
        inner.next(); // Skip 'if'
        let condition = self.build_value(inner.next().unwrap())?;
        inner.next(); // Skip ':'
        let pipeline = self.build_branch_pipeline(inner.next().unwrap())?;
        branches.push(crate::interfaces::IfBranch {
            condition,
            pipeline,
        });

        // elifs or else
        while let Some(token) = inner.next() {
            match token.as_rule() {
                Rule::keyword_elif => {
                    let condition = self.build_value(inner.next().unwrap())?;
                    inner.next(); // Skip ':'
                    let pipeline = self.build_branch_pipeline(inner.next().unwrap())?;
                    branches.push(crate::interfaces::IfBranch {
                        condition,
                        pipeline,
                    });
                }
                Rule::keyword_else => {
                    inner.next(); // Skip ':'
                    else_branch = Some(self.build_branch_pipeline(inner.next().unwrap())?);
                }
                Rule::NEWLINE => continue,
                _ => break,
            }
        }

        Ok(crate::interfaces::IfExpr {
            branches,
            else_branch,
            id: self.next_id(),
        })
    }

    /// Build a pipeline for an if/elif/else branch
    fn build_branch_pipeline(&mut self, pair: Pair<Rule>) -> Result<Vec<Endpoint>, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::branch_pipeline);

        let inner = pair.clone().into_inner().next().unwrap();
        match inner.as_rule() {
            Rule::indented_pipeline => {
                // We're already in a pipeline structure, but we need a starting endpoint for the indented pipeline logic
                // The indented_pipeline rule expects to be processed item by item.
                // However, `build_indented_pipeline` expects a `first` endpoint because it was designed for
                // `a -> \n b`.

                // Wait, `indented_pipeline` in `if` branch behaves differently. It doesn't have a "previous" source
                // from the line before because `if ... :` is the start.
                // Actually, the `if` *is* the source if strictly following flow, but inside the branch,
                // the first element is the start of that branch's pipe.

                // Let's modify `build_indented_pipeline` or create a new helper.
                // The `indented_pipeline` rule is: `indented_pipeline_item+`
                // `indented_pipeline_item` is `endpoint ~ arrow ~ NEWLINE` or `endpoint ~ NEWLINE`

                self.build_standalone_indented_pipeline(inner)
            }
            Rule::endpoint => {
                // Inline: endpoint ~ (arrow ~ endpoint)*
                // The rule in pest is: (endpoint ~ (arrow ~ endpoint)*)
                // But wait, `branch_pipeline` -> `(endpoint ~ (arrow ~ endpoint)*)`
                // Pest might give us a sequence of pairs.

                // If `inner` is just the first `endpoint`, we need to iterate siblings.
                // Actually `pair.into_inner()` will give all the inline components.
                // Let's re-parse `pair` directly if it's inline.

                // If the rule matched `endpoint`, it means it's the inline case.
                self.build_inline_pipeline(pair)
            }
            _ => Ok(vec![]),
        }
    }

    /// Build an inline pipeline: a -> b -> c or a -> b => [shape] -> c
    fn build_inline_pipeline(&mut self, pair: Pair<Rule>) -> Result<Vec<Endpoint>, ParseError> {
        let mut endpoints = vec![];
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::endpoint => {
                    endpoints.push(self.build_endpoint(inner)?);
                }
                Rule::fat_arrow_step => {
                    endpoints.push(self.build_fat_arrow_step(inner)?);
                }
                _ => {}
            }
        }
        Ok(endpoints)
    }

    /// Build an indented pipeline that doesn't start from an existing previous endpoint
    fn build_standalone_indented_pipeline(
        &mut self,
        pair: Pair<Rule>,
    ) -> Result<Vec<Endpoint>, ParseError> {
        let mut endpoints = vec![];

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::indented_pipeline_item {
                for item_inner in inner.into_inner() {
                    match item_inner.as_rule() {
                        Rule::endpoint => {
                            endpoints.push(self.build_endpoint(item_inner)?);
                        }
                        Rule::fat_arrow_step => {
                            endpoints.push(self.build_fat_arrow_step(item_inner)?);
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(endpoints)
    }
    /// Build a @wrap endpoint: @wrap(Wrapper, args): ref_or_pipeline
    fn build_wrap_endpoint(&mut self, pair: Pair<Rule>) -> Result<Endpoint, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::wrap_endpoint);
        let span_start = pair.as_span().start();

        let mut inner = pair.into_inner();

        // Skip: at, keyword_wrap, lparen
        inner.next(); // at
        inner.next(); // keyword_wrap
        inner.next(); // lparen

        // Parse call_args (contains wrapper name + extra args)
        let call_args_pair = inner.next().ok_or_else(|| {
            crate::grammar::error::expected("call arguments after @wrap(", "end of input", span_start)
        })?;
        let (all_args, kwargs) = self.build_call_args(call_args_pair)?;

        // First arg must be the wrapper neuron name (a bare identifier).
        let wrapper_name = match all_args.first() {
            Some(Value::Name(n)) => n.clone(),
            Some(Value::Call { name, .. }) => {
                // The grammar's call_args rule parses `Name(args)` as
                // Value::Call. Reject this form explicitly — @wrap expects
                // a bare name, not a call expression.
                return Err(crate::grammar::error::expected(
                    "bare neuron name as first @wrap argument (not a call)",
                    &format!("{}(...)", name),
                    span_start,
                ));
            }
            _ => {
                return Err(crate::grammar::error::expected(
                    "neuron name as first @wrap argument",
                    "non-name value",
                    span_start,
                ));
            }
        };
        let wrapper_args = all_args[1..].to_vec();

        // Skip rparen and colon
        inner.next(); // rparen
        inner.next(); // colon

        // Now determine content: either ident (ref) or arrow (pipeline)
        let next = inner.next().ok_or_else(|| {
            crate::grammar::error::expected(
                "identifier or '->' after @wrap colon",
                "end of input",
                span_start,
            )
        })?;
        let content = match next.as_rule() {
            Rule::ident => WrapContent::Ref(next.as_str().to_string()),
            Rule::arrow => {
                // Pipeline form: collect subsequent endpoints
                let mut pipeline = Vec::new();

                if let Some(next_pair) = inner.next() {
                    match next_pair.as_rule() {
                        Rule::indented_pipeline => {
                            // Indented pipeline form
                            pipeline = self.build_standalone_indented_pipeline(next_pair)?;
                        }
                        Rule::endpoint => {
                            // Inline: first endpoint, then possibly more
                            pipeline.push(self.build_endpoint(next_pair)?);
                            for remaining in inner {
                                match remaining.as_rule() {
                                    Rule::endpoint => {
                                        pipeline.push(self.build_endpoint(remaining)?);
                                    }
                                    Rule::fat_arrow_step => {
                                        pipeline.push(self.build_fat_arrow_step(remaining)?);
                                    }
                                    Rule::arrow | Rule::NEWLINE => continue,
                                    _ => {}
                                }
                            }
                        }
                        _ => {
                            // Skip NEWLINE tokens and try again
                            if next_pair.as_rule() == Rule::NEWLINE {
                                for remaining in inner {
                                    match remaining.as_rule() {
                                        Rule::indented_pipeline => {
                                            pipeline =
                                                self.build_standalone_indented_pipeline(remaining)?;
                                            break;
                                        }
                                        Rule::endpoint => {
                                            pipeline.push(self.build_endpoint(remaining)?);
                                        }
                                        Rule::NEWLINE => continue,
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }

                WrapContent::Pipeline(pipeline)
            }
            _ => {
                return Err(crate::grammar::error::expected(
                    "identifier or '->' after @wrap colon",
                    next.as_str(),
                    span_start,
                ));
            }
        };

        let id = self.next_id();
        Ok(Endpoint::Wrap(WrapExpr {
            wrapper_name,
            wrapper_args,
            wrapper_kwargs: kwargs,
            content,
            id,
        }))
    }

    /// Build a reference endpoint (in, out, name, name.port)
    fn build_ref_endpoint(&mut self, pair: Pair<Rule>) -> Result<Endpoint, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::ref_endpoint);

        let mut inner = pair.into_inner();
        let first = inner.next().unwrap();

        let node = match first.as_rule() {
            Rule::keyword_in => "in".to_string(),
            Rule::keyword_out => "out".to_string(),
            Rule::ident => first.as_str().to_string(),
            _ => first.as_str().to_string(),
        };

        // Check for port access
        if let Some(dot) = inner.next() {
            if dot.as_rule() == Rule::dot {
                if let Some(port) = inner.next() {
                    return Ok(Endpoint::Ref(PortRef {
                        node,
                        port: port.as_str().to_string(),
                    }));
                }
            }
        }

        Ok(Endpoint::Ref(PortRef {
            node,
            port: "default".to_string(),
        }))
    }

    /// Build a call endpoint (Name(args) or Freeze(Name(args)))
    fn build_call_endpoint(&mut self, pair: Pair<Rule>) -> Result<Endpoint, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::call_endpoint);

        let mut inner = pair.into_inner();
        let first = inner.next().unwrap();

        let (name, args, kwargs, frozen) = match first.as_rule() {
            Rule::call_expr => {
                let (name, args, kwargs) = self.build_call_expr(first)?;
                (name, args, kwargs, false)
            }
            Rule::keyword_freeze => {
                inner.next(); // Skip lparen
                let sub_expr = inner.next().unwrap();
                let (sub_name, sub_args, sub_kwargs, _) = self.build_neuron_expr(sub_expr)?;

                (
                    "Freeze".to_string(),
                    vec![Value::Call {
                        name: sub_name,
                        args: sub_args,
                        kwargs: sub_kwargs,
                    }],
                    vec![],
                    true,
                )
            }
            _ => unreachable!("Unexpected rule in call_endpoint: {:?}", first.as_rule()),
        };

        Ok(Endpoint::Call {
            name,
            args,
            kwargs,
            id: self.next_id(),
            frozen,
        })
    }

    /// Build a tuple endpoint ((a, b, c))
    fn build_tuple_endpoint(&mut self, pair: Pair<Rule>) -> Result<Endpoint, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::tuple_endpoint);

        let mut refs = vec![];

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::ref_endpoint {
                if let Endpoint::Ref(port_ref) = self.build_ref_endpoint(inner)? {
                    refs.push(port_ref);
                }
            }
        }

        Ok(Endpoint::Tuple(refs))
    }

    /// Build a match expression
    fn build_match_expr(&mut self, pair: Pair<Rule>) -> Result<MatchExpr, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::match_expr);

        let mut arms = vec![];

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::match_arm {
                arms.push(self.build_match_arm(inner)?);
            }
        }

        let id = self.next_id();

        Ok(MatchExpr { subject: MatchSubject::Implicit, arms, id })
    }

    /// Build a match evaluation expression: `match(param_name): neuron_match_arms...`
    fn build_match_eval_expr(&mut self, pair: Pair<Rule>) -> Result<MatchExpr, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::match_eval_expr);

        let mut inner = pair.into_inner();

        // Skip keyword_match, lparen
        inner.next(); // keyword_match
        inner.next(); // lparen

        // Extract subject name from value
        let subject_value = inner.next().unwrap();
        let subject_name = match self.build_value(subject_value)? {
            Value::Name(name) => name,
            other => {
                return Err(crate::grammar::error::expected(
                    "parameter name",
                    &format!("{}", other),
                    0,
                ));
            }
        };

        // Skip rparen, colon
        inner.next(); // rparen
        inner.next(); // colon

        let mut arms = vec![];
        for p in inner {
            if p.as_rule() == Rule::neuron_match_arm {
                arms.push(self.build_neuron_match_arm(p)?);
            }
        }

        let id = self.next_id();

        Ok(MatchExpr {
            subject: MatchSubject::Named(subject_name),
            arms,
            id,
        })
    }

    /// Build a neuron match arm: `neuron_port_contract [where guard]: pipeline`
    fn build_neuron_match_arm(&mut self, pair: Pair<Rule>) -> Result<MatchArm, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::neuron_match_arm);

        let mut inner = pair.into_inner();

        // Parse port contract
        let contract = self.build_neuron_port_contract(inner.next().unwrap())?;

        // Optional guard and pipeline
        let mut guard = None;
        let mut pipeline = vec![];

        for p in inner {
            match p.as_rule() {
                Rule::value => {
                    guard = Some(self.build_value(p)?);
                }
                Rule::match_pipeline | Rule::neuron_match_pipeline => {
                    pipeline = self.build_match_pipeline(p)?;
                }
                _ => {}
            }
        }

        Ok(MatchArm {
            pattern: MatchPattern::NeuronContract(contract),
            guard,
            pipeline,
            is_reachable: true,
        })
    }

    /// Build a neuron port contract: `in [shape] -> out [shape]`
    fn build_neuron_port_contract(
        &mut self,
        pair: Pair<Rule>,
    ) -> Result<NeuronPortContract, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::neuron_port_contract);

        let mut inner = pair.into_inner();

        // Parse input: keyword_in, optional ident, shape
        inner.next(); // keyword_in
        let mut next = inner.next().unwrap();
        let in_name = if next.as_rule() == Rule::ident {
            let name = next.as_str().to_string();
            next = inner.next().unwrap(); // advance to shape
            name
        } else {
            "default".to_string()
        };
        let in_shape = self.build_shape(next)?;

        // Skip arrow
        inner.next();

        // Parse output: keyword_out, optional ident, shape
        inner.next(); // keyword_out
        let mut next = inner.next().unwrap();
        let out_name = if next.as_rule() == Rule::ident {
            let name = next.as_str().to_string();
            next = inner.next().unwrap(); // advance to shape
            name
        } else {
            "default".to_string()
        };
        let out_shape = self.build_shape(next)?;

        Ok(NeuronPortContract {
            input_ports: vec![(in_name, in_shape)],
            output_ports: vec![(out_name, out_shape)],
        })
    }

    /// Build a match arm
    fn build_match_arm(&mut self, pair: Pair<Rule>) -> Result<MatchArm, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::match_arm);

        let mut inner = pair.into_inner();

        // Pattern (shape)
        let pattern = self.build_shape(inner.next().unwrap())?;

        // Optional guard and pipeline
        let mut guard = None;
        let mut pipeline = vec![];

        for p in inner {
            match p.as_rule() {
                Rule::value => {
                    guard = Some(self.build_value(p)?);
                }
                Rule::match_pipeline => {
                    pipeline = self.build_match_pipeline(p)?;
                }
                _ => {}
            }
        }

        Ok(MatchArm {
            pattern: MatchPattern::Shape(pattern),
            guard,
            pipeline,
            is_reachable: true,
        })
    }

    /// Build a match pipeline (supports both inline and indented)
    fn build_match_pipeline(&mut self, pair: Pair<Rule>) -> Result<Vec<Endpoint>, ParseError> {
        debug_assert!(
            pair.as_rule() == Rule::match_pipeline
                || pair.as_rule() == Rule::neuron_match_pipeline
        );

        let mut endpoints = vec![];

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::endpoint => {
                    endpoints.push(self.build_endpoint(inner)?);
                }
                Rule::fat_arrow_step => {
                    endpoints.push(self.build_fat_arrow_step(inner)?);
                }
                Rule::indented_pipeline | Rule::neuron_match_indented_pipeline => {
                    // Handle indented multi-line pipelines
                    for item in inner.into_inner() {
                        if item.as_rule() == Rule::indented_pipeline_item {
                            for endpoint_pair in item.into_inner() {
                                match endpoint_pair.as_rule() {
                                    Rule::endpoint => {
                                        endpoints.push(self.build_endpoint(endpoint_pair)?);
                                    }
                                    Rule::fat_arrow_step => {
                                        endpoints.push(self.build_fat_arrow_step(endpoint_pair)?);
                                    }
                                    _ => {}
                                }
                            }
                        } else if item.as_rule() == Rule::neuron_match_indented_item {
                            // neuron_match_indented_item wraps indented_pipeline_item
                            for pipeline_item in item.into_inner() {
                                if pipeline_item.as_rule() == Rule::indented_pipeline_item {
                                    for endpoint_pair in pipeline_item.into_inner() {
                                        match endpoint_pair.as_rule() {
                                            Rule::endpoint => {
                                                endpoints.push(self.build_endpoint(endpoint_pair)?);
                                            }
                                            Rule::fat_arrow_step => {
                                                endpoints.push(self.build_fat_arrow_step(endpoint_pair)?);
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(endpoints)
    }

    /// Build a Value from a value pair
    fn build_value(&mut self, pair: Pair<Rule>) -> Result<Value, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::value);

        let inner = pair.into_inner().next().unwrap();
        self.build_logical_or(inner)
    }

    /// Build a logical OR expression (lowest precedence among logical ops)
    fn build_logical_or(&mut self, pair: Pair<Rule>) -> Result<Value, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::value_logical_or);

        let mut inner: Vec<_> = pair.into_inner().collect();

        if inner.len() == 1 {
            return self.build_logical_and(inner.remove(0));
        }

        // Build left-associative binary ops
        let mut result = self.build_logical_and(inner.remove(0))?;

        while inner.len() >= 2 {
            // Skip the or_op token
            inner.remove(0);
            let right = self.build_logical_and(inner.remove(0))?;
            result = Value::BinOp {
                op: BinOp::Or,
                left: Box::new(result),
                right: Box::new(right),
            };
        }

        Ok(result)
    }

    /// Build a logical AND expression
    fn build_logical_and(&mut self, pair: Pair<Rule>) -> Result<Value, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::value_logical_and);

        let mut inner: Vec<_> = pair.into_inner().collect();

        if inner.len() == 1 {
            return self.build_value_comparison(inner.remove(0));
        }

        // Build left-associative binary ops
        let mut result = self.build_value_comparison(inner.remove(0))?;

        while inner.len() >= 2 {
            // Skip the and_op token
            inner.remove(0);
            let right = self.build_value_comparison(inner.remove(0))?;
            result = Value::BinOp {
                op: BinOp::And,
                left: Box::new(result),
                right: Box::new(right),
            };
        }

        Ok(result)
    }

    /// Build a comparison expression
    fn build_value_comparison(&mut self, pair: Pair<Rule>) -> Result<Value, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::value_comparison);

        let mut inner: Vec<_> = pair.into_inner().collect();

        if inner.len() == 1 {
            return self.build_value_additive(inner.remove(0));
        }

        // Binary comparison
        let left = self.build_value_additive(inner.remove(0))?;
        let op = self.build_comparison_op(inner.remove(0))?;
        let right = self.build_value_additive(inner.remove(0))?;

        Ok(Value::BinOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// Build a comparison operator
    fn build_comparison_op(&mut self, pair: Pair<Rule>) -> Result<BinOp, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::comparison_op);

        let inner = pair.into_inner().next().unwrap();

        match inner.as_rule() {
            Rule::eq_op => Ok(BinOp::Eq),
            Rule::ne_op => Ok(BinOp::Ne),
            Rule::le_op => Ok(BinOp::Le),
            Rule::ge_op => Ok(BinOp::Ge),
            Rule::lt_op => Ok(BinOp::Lt),
            Rule::gt_op => Ok(BinOp::Gt),
            _ => Ok(BinOp::Eq),
        }
    }

    /// Build an additive expression
    fn build_value_additive(&mut self, pair: Pair<Rule>) -> Result<Value, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::value_additive);

        let mut inner: Vec<_> = pair.into_inner().collect();

        if inner.len() == 1 {
            return self.build_value_multiplicative(inner.remove(0));
        }

        // Build left-associative binary ops
        let mut result = self.build_value_multiplicative(inner.remove(0))?;

        while inner.len() >= 2 {
            let op_pair = inner.remove(0);
            let op = match op_pair.as_rule() {
                Rule::plus => BinOp::Add,
                Rule::minus => BinOp::Sub,
                _ => BinOp::Add,
            };
            let right = self.build_value_multiplicative(inner.remove(0))?;
            result = Value::BinOp {
                op,
                left: Box::new(result),
                right: Box::new(right),
            };
        }

        Ok(result)
    }

    /// Build a multiplicative expression
    fn build_value_multiplicative(&mut self, pair: Pair<Rule>) -> Result<Value, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::value_multiplicative);

        let mut inner: Vec<_> = pair.into_inner().collect();

        if inner.len() == 1 {
            return self.build_value_primary(inner.remove(0));
        }

        // Build left-associative binary ops
        let mut result = self.build_value_primary(inner.remove(0))?;

        while inner.len() >= 2 {
            let op_pair = inner.remove(0);
            let op = match op_pair.as_rule() {
                Rule::star => BinOp::Mul,
                Rule::slash => BinOp::Div,
                _ => BinOp::Mul,
            };
            let right = self.build_value_primary(inner.remove(0))?;
            result = Value::BinOp {
                op,
                left: Box::new(result),
                right: Box::new(right),
            };
        }

        Ok(result)
    }

    /// Build a primary value
    fn build_value_primary(&mut self, pair: Pair<Rule>) -> Result<Value, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::value_primary);

        let mut inner_iter = pair.into_inner();
        let inner = inner_iter.next().unwrap();

        // Handle parenthesized expressions: lparen ~ value ~ rparen
        if inner.as_rule() == Rule::lparen {
            let value_pair = inner_iter.next().unwrap();
            return self.build_value(value_pair);
        }

        match inner.as_rule() {
            Rule::float => {
                let s = inner.as_str();
                let f: f64 = s.parse().unwrap_or(0.0);
                // Convert negative floats to (0 - x) for compatibility with old parser
                if s.starts_with('-') {
                    Ok(Value::BinOp {
                        op: BinOp::Sub,
                        left: Box::new(Value::Int(0)),
                        right: Box::new(Value::Float(-f)),
                    })
                } else {
                    Ok(Value::Float(f))
                }
            }
            Rule::integer => {
                let s = inner.as_str();
                let n: i64 = s.parse().unwrap_or(0);
                // Convert negative integers to (0 - x) for compatibility with old parser
                if s.starts_with('-') {
                    Ok(Value::BinOp {
                        op: BinOp::Sub,
                        left: Box::new(Value::Int(0)),
                        right: Box::new(Value::Int(-n)),
                    })
                } else {
                    Ok(Value::Int(n))
                }
            }
            Rule::string => {
                // Remove backticks
                let s = inner.as_str();
                let s = s.trim_start_matches('`').trim_end_matches('`');
                Ok(Value::String(s.to_string()))
            }
            Rule::bool => {
                let inner2 = inner.into_inner().next().unwrap();
                match inner2.as_rule() {
                    Rule::keyword_true => Ok(Value::Bool(true)),
                    Rule::keyword_false => Ok(Value::Bool(false)),
                    _ => Ok(Value::Bool(false)),
                }
            }
            Rule::call_expr => {
                let (name, args, kwargs) = self.build_call_expr(inner)?;
                Ok(Value::Call { name, args, kwargs })
            }
            Rule::global_ref => {
                let mut inner_pairs = inner.into_inner();
                // Skip '@', 'global'
                inner_pairs.next();
                inner_pairs.next();
                let ident = inner_pairs.next().unwrap().as_str().to_string();
                Ok(Value::Global(ident))
            }
            Rule::ident => Ok(Value::Name(inner.as_str().to_string())),
            Rule::value => self.build_value(inner),
            _ => Ok(Value::Int(0)),
        }
    }

    // ========================================================================
    // Fat Arrow / Reshape builders
    // ========================================================================

    /// Build a fat_arrow_step: `=> [reshape_expr]` or `=> @annotation [reshape_expr]`
    fn build_fat_arrow_step(&mut self, pair: Pair<Rule>) -> Result<Endpoint, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::fat_arrow_step);

        let mut annotation = None;
        let mut reshape_expr = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::transform_annotation => {
                    annotation = Some(self.build_transform_annotation(inner)?);
                }
                Rule::reshape_expr => {
                    reshape_expr = Some(self.build_reshape_expr(inner)?);
                }
                Rule::fat_arrow => {
                    // Skip the `=>` token itself
                }
                _ => {}
            }
        }

        let mut expr = reshape_expr.expect(
            "grammar guarantees reshape_expr is present in fat_arrow_step",
        );
        expr.annotation = annotation;

        Ok(Endpoint::Reshape(expr))
    }

    /// Build a reshape_expr: `[dim_spec, dim_spec, ...]`
    fn build_reshape_expr(&mut self, pair: Pair<Rule>) -> Result<ReshapeExpr, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::reshape_expr);

        let span = pair.as_span();
        let source_span = SourceSpan::new(span.start().into(), span.end() - span.start());

        let mut dims = vec![];

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::reshape_dim {
                dims.push(self.build_reshape_dim(inner)?);
            }
        }

        Ok(ReshapeExpr {
            dims,
            annotation: None,
            id: self.next_id(),
            span: Some(source_span),
        })
    }

    /// Build a reshape_dim: named, literal, binding, others, or expression
    fn build_reshape_dim(&mut self, pair: Pair<Rule>) -> Result<ReshapeDim, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::reshape_dim);

        let inner: Vec<_> = pair.into_inner().collect();

        if inner.is_empty() {
            return Err(error::expected(
                "reshape dimension (name, literal, binding, or 'others')",
                "<empty>",
                0,
            ));
        }

        let first = &inner[0];

        match first.as_rule() {
            Rule::keyword_others => Ok(ReshapeDim::Others),
            Rule::ident if inner.len() >= 3 && inner[1].as_rule() == Rule::assign => {
                // Binding: name = dim_expr
                let name = first.as_str().to_string();
                // inner[1] is assign, inner[2] is dim
                let dim = self.build_dim(inner[2].clone())?;
                let expr = dim_to_value(dim)?;
                Ok(ReshapeDim::Binding {
                    name,
                    expr: Box::new(expr),
                })
            }
            Rule::dim => {
                // Plain dimension: named, literal, expr, wildcard, variadic
                let dim = self.build_dim(first.clone())?;
                match dim {
                    Dim::Named(name) => Ok(ReshapeDim::Named(name)),
                    Dim::Literal(n) => Ok(ReshapeDim::Literal(n)),
                    Dim::Expr(expr) => Ok(ReshapeDim::Expr(expr)),
                    Dim::Global(name) => Err(error::expected(
                        "named dimension or literal in reshape expression (globals not supported in reshape dims)",
                        &format!("@global {}", name),
                        0,
                    )),
                    Dim::Wildcard => Err(error::expected(
                        "named dimension, literal, or 'others'",
                        "*",
                        0,
                    )),
                    Dim::Inferred => Err(error::expected(
                        "named dimension, literal, or 'others'",
                        "inferred dimension",
                        0,
                    )),
                    Dim::Variadic(_) => Err(error::expected(
                        "named dimension, literal, or 'others'",
                        first.as_str(),
                        0,
                    )),
                }
            }
            _ => Ok(ReshapeDim::Named(first.as_str().to_string())),
        }
    }

    /// Build a transform_annotation: `@reduce(mean)` or `@repeat(copy)` or `@reduce(AttentionPool(dim))`
    fn build_transform_annotation(
        &mut self,
        pair: Pair<Rule>,
    ) -> Result<TransformAnnotation, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::transform_annotation);

        let span = pair.as_span();
        let source_span = Some(SourceSpan::new(span.start().into(), span.end() - span.start()));

        let mut inner = pair.into_inner();
        inner.next(); // Skip '@'
        let annotation_name = inner.next().unwrap(); // ident: "reduce" or "repeat"
        let name = annotation_name.as_str().to_string();
        inner.next(); // Skip '('
        let arg = inner.next().unwrap(); // annotation_arg
        let strategy = self.build_annotation_arg(arg)?;
        // Skip ')' (if present)

        match name.as_str() {
            "reduce" => Ok(TransformAnnotation::Reduce { strategy, span: source_span }),
            "repeat" => Ok(TransformAnnotation::Repeat { strategy, span: source_span }),
            other => Err(error::expected("reduce or repeat", other, 0))
        }
    }

    /// Build an annotation_arg: either a call_expr or a simple ident
    fn build_annotation_arg(&mut self, pair: Pair<Rule>) -> Result<TransformStrategy, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::annotation_arg);

        let inner = pair.into_inner().next().unwrap();

        match inner.as_rule() {
            Rule::call_expr => {
                let (name, args, kwargs) = self.build_call_expr(inner)?;
                Ok(TransformStrategy::Neuron { name, args, kwargs })
            }
            Rule::ident => {
                let name = inner.as_str().to_string();
                Ok(TransformStrategy::Intrinsic(name))
            }
            _ => Ok(TransformStrategy::Intrinsic(inner.as_str().to_string())),
        }
    }

    /// Extract identifier string from an ident pair
    fn extract_ident(&mut self, pair: Pair<Rule>) -> Result<String, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::ident);
        Ok(pair.as_str().to_string())
    }
}

impl Default for AstBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a Dim to a Value for use in reshape bindings (e.g., `h=dim/heads`).
/// Returns Err if the dim contains Wildcard or Variadic (invalid in reshape bindings).
fn dim_to_value(dim: Dim) -> Result<Value, ParseError> {
    match dim {
        Dim::Literal(n) => Ok(Value::Int(n)),
        Dim::Named(name) => Ok(Value::Name(name)),
        Dim::Global(name) => Ok(Value::Global(name)),
        Dim::Wildcard => Err(error::expected(
            "named dimension or literal in reshape binding expression",
            "*",
            0,
        )),
        Dim::Inferred => unreachable!("Dim::Inferred is only produced by ReshapeExpr::to_shape(), not by the parser"),
        Dim::Variadic(name) => Err(error::expected(
            "named dimension or literal in reshape binding expression",
            &format!("*{}", name),
            0,
        )),
        Dim::Expr(expr) => {
            let op = expr.op;
            let left = dim_to_value(expr.left)?;
            let right = dim_to_value(expr.right)?;
            Ok(Value::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            })
        }
    }
}

#[cfg(test)]
mod tests;
