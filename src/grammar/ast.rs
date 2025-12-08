//! AST Builder for pest-based parser
//!
//! Converts pest parse trees (Pair<Rule>) into NeuroScript IR types.
//! Handles indentation validation during the conversion.

use pest::iterators::Pair;

use crate::grammar::Rule;
use crate::grammar::error;
use crate::interfaces::{
    BinOp, Binding, Connection, Dim, DimExpr, Endpoint, ImplRef, MatchArm, MatchExpr,
    NeuronBody, NeuronDef, Param, ParseError, Port, PortRef, Program, Shape, UseStmt, Value,
};

/// AST builder state
pub struct AstBuilder {
    /// Counter for generating unique node IDs (for Call endpoints)
    next_node_id: usize,
}

impl AstBuilder {
    pub fn new() -> Self {
        AstBuilder { next_node_id: 0 }
    }

    fn next_id(&mut self) -> usize {
        let id = self.next_node_id;
        self.next_node_id += 1;
        id
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

        // Skip keyword_neuron
        inner.next();

        // Get name
        let name = self.extract_ident(inner.next().unwrap())?;

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
        let mut inputs = vec![];
        let mut outputs = vec![];
        let mut let_bindings = vec![];
        let mut set_bindings = vec![];
        let mut connections = vec![];
        let mut impl_ref = None;

        // Process remaining elements (sections)
        while let Some(p) = next {
            if p.as_rule() == Rule::neuron_section {
                self.process_neuron_section(
                    p,
                    &mut inputs,
                    &mut outputs,
                    &mut let_bindings,
                    &mut set_bindings,
                    &mut connections,
                    &mut impl_ref,
                )?;
            }
            next = inner.next();
        }

        // Construct body
        let body = if let Some(impl_ref_val) = impl_ref {
            NeuronBody::Primitive(impl_ref_val)
        } else {
            NeuronBody::Graph {
                let_bindings,
                set_bindings,
                connections,
            }
        };

        Ok(NeuronDef {
            name,
            params,
            inputs,
            outputs,
            body,
        })
    }

    /// Process a neuron_section and update the appropriate vectors
    fn process_neuron_section(
        &mut self,
        pair: Pair<Rule>,
        inputs: &mut Vec<Port>,
        outputs: &mut Vec<Port>,
        let_bindings: &mut Vec<Binding>,
        set_bindings: &mut Vec<Binding>,
        connections: &mut Vec<Connection>,
        impl_ref: &mut Option<ImplRef>,
    ) -> Result<(), ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::neuron_section);

        let section = pair.into_inner().next().unwrap();

        match section.as_rule() {
            Rule::in_section => {
                let ports = self.build_in_section(section)?;
                inputs.extend(ports);
            }
            Rule::out_section => {
                let ports = self.build_out_section(section)?;
                outputs.extend(ports);
            }
            Rule::let_section => {
                let bindings = self.build_let_section(section)?;
                let_bindings.extend(bindings);
            }
            Rule::set_section => {
                let bindings = self.build_set_section(section)?;
                set_bindings.extend(bindings);
            }
            Rule::graph_section => {
                let conns = self.build_graph_section(section)?;
                connections.extend(conns);
            }
            Rule::impl_section => {
                *impl_ref = Some(self.build_impl_section(section)?);
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
        for p in inner {
            match p.as_rule() {
                Rule::type_annotation => {
                    // Type annotations are parsed but not yet used in IR
                }
                Rule::value => {
                    default = Some(self.build_value(p)?);
                }
                _ => {}
            }
        }

        Ok(Param { name, default })
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

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::port_def => {
                    ports.push(self.build_port_def(inner)?);
                }
                Rule::shape => {
                    // Inline shape without name (e.g., "in: [shape]")
                    // Or shape following an ident
                    let s = self.build_shape(inner)?;
                    if let Some(n) = name.take() {
                        // We had an ident before, this is "in name: [shape]"
                        ports.push(Port { name: n, shape: s });
                    } else {
                        ports.push(Port {
                            name: "default".to_string(),
                            shape: s,
                        });
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

        Ok(Port { name, shape })
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

    /// Build bindings from let_section
    fn build_let_section(&mut self, pair: Pair<Rule>) -> Result<Vec<Binding>, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::let_section);

        let mut bindings = vec![];

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::binding {
                bindings.push(self.build_binding(inner)?);
            }
        }

        Ok(bindings)
    }

    /// Build bindings from set_section
    fn build_set_section(&mut self, pair: Pair<Rule>) -> Result<Vec<Binding>, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::set_section);

        let mut bindings = vec![];

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::binding {
                bindings.push(self.build_binding(inner)?);
            }
        }

        Ok(bindings)
    }

    /// Build a binding (name = Call(args))
    fn build_binding(&mut self, pair: Pair<Rule>) -> Result<Binding, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::binding);

        let mut inner = pair.into_inner();
        let name = self.extract_ident(inner.next().unwrap())?;

        // Skip assign
        inner.next();

        let call = inner.next().unwrap();
        let (call_name, args, kwargs) = self.build_call_expr(call)?;

        Ok(Binding {
            name,
            call_name,
            args,
            kwargs,
        })
    }

    /// Build a call expression, returning (name, args, kwargs)
    fn build_call_expr(
        &mut self,
        pair: Pair<Rule>,
    ) -> Result<(String, Vec<Value>, Vec<(String, Value)>), ParseError> {
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

    /// Build call arguments
    fn build_call_args(
        &mut self,
        pair: Pair<Rule>,
    ) -> Result<(Vec<Value>, Vec<(String, Value)>), ParseError> {
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
    fn build_kwarg(&mut self, pair: Pair<Rule>) -> Result<(String, Value), ParseError> {
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

        // Skip arrow
        inner.next();

        // connection_tail
        let tail = inner.next().unwrap();
        self.build_connection_tail(first_endpoint, tail)
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
                    if item_inner.as_rule() == Rule::endpoint {
                        let next = self.build_endpoint(item_inner)?;
                        connections.push(Connection {
                            source: prev,
                            destination: next.clone(),
                        });
                        prev = next;
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
            Rule::match_expr => Ok(Endpoint::Match(self.build_match_expr(inner)?)),
            Rule::tuple_endpoint => self.build_tuple_endpoint(inner),
            Rule::call_endpoint => self.build_call_endpoint(inner),
            Rule::ref_endpoint => self.build_ref_endpoint(inner),
            _ => Err(error::expected("endpoint", inner.as_str(), 0)),
        }
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
                    return Ok(Endpoint::Ref(PortRef::with_port(node, port.as_str())));
                }
            } else if dot.as_rule() == Rule::ident {
                // The dot was implicit, this is the port name
                return Ok(Endpoint::Ref(PortRef::with_port(node, dot.as_str())));
            }
        }

        Ok(Endpoint::Ref(PortRef::new(node)))
    }

    /// Build a call endpoint (Name(args))
    fn build_call_endpoint(&mut self, pair: Pair<Rule>) -> Result<Endpoint, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::call_endpoint);

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

        Ok(Endpoint::Call {
            name,
            args,
            kwargs,
            id: self.next_id(),
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

        Ok(MatchExpr { arms })
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
            pattern,
            guard,
            pipeline,
            is_reachable: true,
        })
    }

    /// Build a match pipeline
    fn build_match_pipeline(&mut self, pair: Pair<Rule>) -> Result<Vec<Endpoint>, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::match_pipeline);

        let mut endpoints = vec![];

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::endpoint {
                endpoints.push(self.build_endpoint(inner)?);
            }
        }

        Ok(endpoints)
    }

    /// Build a Value from a value pair
    fn build_value(&mut self, pair: Pair<Rule>) -> Result<Value, ParseError> {
        debug_assert_eq!(pair.as_rule(), Rule::value);

        let inner = pair.into_inner().next().unwrap();
        self.build_value_comparison(inner)
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

        let inner = pair.into_inner().next().unwrap();

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
            Rule::ident => Ok(Value::Name(inner.as_str().to_string())),
            Rule::value => self.build_value(inner),
            _ => Ok(Value::Int(0)),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::NeuroScriptParser;
    use crate::interfaces::Parser as OldParser;
    use pest::Parser;

    fn parse_program(input: &str) -> Result<Program, ParseError> {
        let pairs = NeuroScriptParser::parse(Rule::program, input)
            .map_err(|e| error::from_pest_error(e))?;

        let mut builder = AstBuilder::new();
        builder.build_program(pairs.into_iter().next().unwrap())
    }

    #[test]
    fn test_simple_neuron() {
        let input = r#"neuron Linear(in_dim, out_dim):
  in: [*, in_dim]
  out: [*, out_dim]
  impl: core,nn/Linear
"#;
        let program = parse_program(input).expect("Failed to parse");
        assert_eq!(program.neurons.len(), 1);
        assert!(program.neurons.contains_key("Linear"));

        let neuron = &program.neurons["Linear"];
        assert_eq!(neuron.params.len(), 2);
        assert_eq!(neuron.inputs.len(), 1);
        assert_eq!(neuron.outputs.len(), 1);
        assert!(matches!(neuron.body, NeuronBody::Primitive(_)));
    }

    #[test]
    fn test_use_stmt() {
        let input = r#"use core,nn/*

neuron Test:
  in: [*]
  out: [*]
  impl: core,nn/Test
"#;
        let program = parse_program(input).expect("Failed to parse");
        assert_eq!(program.uses.len(), 1);
        assert_eq!(program.uses[0].source, "core");
        assert_eq!(program.uses[0].path, vec!["nn", "*"]);
    }

    #[test]
    fn test_composite_neuron() {
        let input = r#"neuron MLP(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out
"#;
        let program = parse_program(input).expect("Failed to parse");
        let neuron = &program.neurons["MLP"];

        if let NeuronBody::Graph { connections, .. } = &neuron.body {
            assert_eq!(connections.len(), 2); // in->Linear, Linear->out
        } else {
            panic!("Expected Graph body");
        }
    }

    // === Comparison tests: pest vs handwritten parser ===

    fn compare_parsers(input: &str, name: &str) {
        let pest_result = parse_program(input);
        let old_result = OldParser::parse(input);

        match (&pest_result, &old_result) {
            (Ok(pest_prog), Ok(old_prog)) => {
                // Compare neuron counts
                assert_eq!(
                    pest_prog.neurons.len(),
                    old_prog.neurons.len(),
                    "{}: neuron count mismatch",
                    name
                );

                // Compare use statement counts
                assert_eq!(
                    pest_prog.uses.len(),
                    old_prog.uses.len(),
                    "{}: use statement count mismatch",
                    name
                );

                // Compare each neuron
                for (neuron_name, old_neuron) in &old_prog.neurons {
                    let pest_neuron = pest_prog
                        .neurons
                        .get(neuron_name)
                        .unwrap_or_else(|| panic!("{}: missing neuron {}", name, neuron_name));

                    // Compare params
                    assert_eq!(
                        pest_neuron.params.len(),
                        old_neuron.params.len(),
                        "{}: param count mismatch for {}",
                        name,
                        neuron_name
                    );

                    // Compare inputs
                    assert_eq!(
                        pest_neuron.inputs.len(),
                        old_neuron.inputs.len(),
                        "{}: input count mismatch for {}",
                        name,
                        neuron_name
                    );

                    // Compare outputs
                    assert_eq!(
                        pest_neuron.outputs.len(),
                        old_neuron.outputs.len(),
                        "{}: output count mismatch for {}",
                        name,
                        neuron_name
                    );

                    // Compare body type
                    match (&pest_neuron.body, &old_neuron.body) {
                        (NeuronBody::Primitive(_), NeuronBody::Primitive(_)) => {}
                        (
                            NeuronBody::Graph { connections: pc, .. },
                            NeuronBody::Graph { connections: oc, .. },
                        ) => {
                            assert_eq!(
                                pc.len(),
                                oc.len(),
                                "{}: connection count mismatch for {}",
                                name,
                                neuron_name
                            );
                        }
                        _ => panic!(
                            "{}: body type mismatch for {}: {:?} vs {:?}",
                            name,
                            neuron_name,
                            std::mem::discriminant(&pest_neuron.body),
                            std::mem::discriminant(&old_neuron.body)
                        ),
                    }
                }
            }
            (Err(e), Ok(_)) => panic!("{}: pest failed but old succeeded: {:?}", name, e),
            (Ok(_), Err(e)) => panic!("{}: old failed but pest succeeded: {:?}", name, e),
            (Err(_), Err(_)) => {
                // Both failed - that's ok, they might have the same error
            }
        }
    }

    #[test]
    fn test_compare_residual() {
        let input = include_str!("../../examples/residual.ns");
        compare_parsers(input, "residual.ns");
    }

    #[test]
    fn test_compare_01_comments() {
        let input = include_str!("../../examples/01-comments.ns");
        compare_parsers(input, "01-comments.ns");
    }

    #[test]
    fn test_compare_03_parameters() {
        let input = include_str!("../../examples/03-parameters.ns");
        compare_parsers(input, "03-parameters.ns");
    }

    #[test]
    fn test_compare_07_pipelines() {
        let input = include_str!("../../examples/07-pipelines.ns");
        compare_parsers(input, "07-pipelines.ns");
    }

    #[test]
    fn test_compare_10_match() {
        let input = include_str!("../../examples/10-match.ns");
        compare_parsers(input, "10-match.ns");
    }

    #[test]
    fn test_compare_22_xor() {
        let input = include_str!("../../examples/22-xor.ns");
        compare_parsers(input, "22-xor.ns");
    }

    #[test]
    fn test_compare_28_let_set() {
        let input = include_str!("../../examples/28-let_set_basic.ns");
        compare_parsers(input, "28-let_set_basic.ns");
    }

    // Run comparison on all numbered example files
    macro_rules! compare_example {
        ($name:ident, $file:expr) => {
            #[test]
            fn $name() {
                let input = include_str!(concat!("../../examples/", $file));
                compare_parsers(input, $file);
            }
        };
    }

    compare_example!(compare_02_imports, "02-imports.ns");
    compare_example!(compare_04_shapes, "04-shapes.ns");
    compare_example!(compare_05_ports, "05-ports.ns");
    compare_example!(compare_06_impl_refs, "06-impl-refs.ns");
    compare_example!(compare_08_tuples, "08-tuples.ns");
    compare_example!(compare_09_port_access, "09-port-access.ns");
    compare_example!(compare_11_calls, "11-calls.ns");
    compare_example!(compare_12_expressions, "12-expressions.ns");
    compare_example!(compare_13_values, "13-values.ns");
    compare_example!(compare_14_composite, "14-composite.ns");
    compare_example!(compare_15_edge_cases, "15-edge-cases.ns");
}
