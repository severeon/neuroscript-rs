use crate::interfaces::*;
use std::collections::HashMap;

impl InferenceContext {
    pub fn new() -> Self {
        InferenceContext {
            node_outputs: HashMap::new(),
            call_outputs: HashMap::new(),
            resolved_dims: HashMap::new(),
            resolved_variadics: HashMap::new(),
            pending_constraints: Vec::new(),
            equivalences: HashMap::new(),
        }
    }

    /// Register a solved dimension value
    pub fn resolve_dim(&mut self, name: String, value: usize) -> Result<(), String> {
        if let Some(existing) = self.resolved_dims.get(&name) {
            if *existing != value {
                return Err(format!(
                    "Dimension mismatch: {} already resolved to {}, but now trying to set to {}",
                    name, existing, value
                ));
            }
        }
        self.resolved_dims.insert(name, value);
        Ok(())
    }

    /// Try to unify two dimensions, possibly solving for unknowns
    pub fn unify(&mut self, d1: &Dim, d2: &Dim) -> Result<(), String> {
        match (d1, d2) {
            (Dim::Literal(n1), Dim::Literal(n2)) => {
                if n1 != n2 {
                    return Err(format!("Literal dimension mismatch: {} != {}", n1, n2));
                }
                Ok(())
            }
            (Dim::Literal(n), Dim::Named(name)) | (Dim::Named(name), Dim::Literal(n)) => {
                self.resolve_dim(name.clone(), *n as usize)
            }
            (Dim::Named(n1), Dim::Named(n2)) => {
                let v1 = self.resolved_dims.get(n1).copied();
                let v2 = self.resolved_dims.get(n2).copied();
                match (v1, v2) {
                    (Some(val1), Some(val2)) => {
                        if val1 != val2 {
                            return Err(format!("Named dimension mismatch: {} != {}", val1, val2));
                        }
                        Ok(())
                    }
                    (Some(val), None) => self.resolve_dim(n2.clone(), val),
                    (None, Some(val)) => self.resolve_dim(n1.clone(), val),
                    (None, None) => {
                        // Both unknown - for now, we just record they must be equal
                        // In a more advanced engine, we'd use a Union-Find
                        self.equivalences.insert(n1.clone(), n2.clone());
                        Ok(())
                    }
                }
            }
            (Dim::Literal(n), Dim::Expr(expr)) | (Dim::Expr(expr), Dim::Literal(n)) => {
                self.solve_expr_for_unknown(expr, *n as usize)
            }
            (Dim::Wildcard, _) | (_, Dim::Wildcard) => Ok(()), // Wildcard matches anything
            (Dim::Inferred, _) | (_, Dim::Inferred) => Ok(()), // Inferred matches anything
            (Dim::Global(g1), Dim::Global(g2)) => {
                if g1 == g2 {
                    Ok(())
                } else {
                    Err(format!(
                        "Global dimension mismatch: @global {} != @global {}",
                        g1, g2
                    ))
                }
            }
            // Variadic-Variadic: record equivalence (like Named-Named)
            (Dim::Variadic(v1), Dim::Variadic(v2)) => {
                if v1 != v2 {
                    self.equivalences.insert(v1.clone(), v2.clone());
                }
                Ok(())
            }
            // Variadic vs any non-Variadic: error
            (Dim::Variadic(v), other) | (other, Dim::Variadic(v)) => {
                Err(format!(
                    "Cannot unify variadic dimension *{} with non-variadic dimension {}",
                    v, other
                ))
            }
            // Named vs Expr: try to solve if named is resolved, otherwise record constraint
            (Dim::Named(n), Dim::Expr(expr)) | (Dim::Expr(expr), Dim::Named(n)) => {
                if let Some(val) = self.resolved_dims.get(n).copied() {
                    self.solve_expr_for_unknown(expr, val)
                } else {
                    // Cannot solve yet — record as pending constraint
                    self.pending_constraints.push((
                        Dim::Named(n.clone()),
                        (**expr).clone(),
                        format!("{} ~ ({} {} {})", n, expr.left, expr.op, expr.right),
                    ));
                    Ok(())
                }
            }
            // TODO(SHAPE-5): Expr vs Expr — too complex to solve generically;
            // could attempt structural equality or partial evaluation in the future.
            (Dim::Expr(_), Dim::Expr(_)) => Ok(()),
            // Global vs Named: resolve named to global's value if possible
            (Dim::Global(g), Dim::Named(n)) | (Dim::Named(n), Dim::Global(g)) => {
                if let Some(val) = self.resolved_dims.get(g).copied() {
                    self.resolve_dim(n.clone(), val)
                } else if let Some(val) = self.resolved_dims.get(n).copied() {
                    self.resolve_dim(g.clone(), val)
                } else {
                    self.equivalences.insert(n.clone(), g.clone());
                    Ok(())
                }
            }
            // Global vs Literal: resolve global
            (Dim::Global(g), Dim::Literal(n)) | (Dim::Literal(n), Dim::Global(g)) => {
                self.resolve_dim(g.clone(), *n as usize)
            }
            // Global vs Expr: solve if global is resolved
            (Dim::Global(g), Dim::Expr(expr)) | (Dim::Expr(expr), Dim::Global(g)) => {
                if let Some(val) = self.resolved_dims.get(g).copied() {
                    self.solve_expr_for_unknown(expr, val)
                } else {
                    self.pending_constraints.push((
                        Dim::Global(g.clone()),
                        (**expr).clone(),
                        format!("@global {} ~ ({} {} {})", g, expr.left, expr.op, expr.right),
                    ));
                    Ok(())
                }
            }
            // All Dim variant combinations are covered above
        }
    }

    /// Solve for an unknown in a DimExpr given a target value
    pub(crate) fn solve_expr_for_unknown(
        &mut self,
        expr: &DimExpr,
        target: usize,
    ) -> Result<(), String> {
        match (&expr.left, &expr.right) {
            (Dim::Named(left_name), Dim::Literal(right_val)) => {
                // Solve: left op right = target
                let right_usize = *right_val as usize;
                match expr.op {
                    BinOp::Add => {
                        if target >= right_usize {
                            self.resolve_dim(left_name.clone(), target - right_usize)
                        } else {
                            Err(format!(
                                "Illegal negative dimension: {} + {} = {}",
                                left_name, right_val, target
                            ))
                        }
                    }
                    BinOp::Sub => self.resolve_dim(left_name.clone(), target + right_usize),
                    BinOp::Mul => {
                        if right_usize == 0 {
                            return Err(format!(
                                "Division by zero: cannot solve {} * 0 = {}",
                                left_name, target
                            ));
                        }
                        #[allow(clippy::manual_is_multiple_of)]
                        if target % right_usize != 0 {
                            return Err(format!(
                                "Cannot solve {} * {} = {}: {} is not divisible by {}",
                                left_name, right_val, target, target, right_val
                            ));
                        }
                        self.resolve_dim(left_name.clone(), target / right_usize)
                    }
                    BinOp::Div => {
                        self.resolve_dim(left_name.clone(), target * right_usize)
                    }
                    _ => Err(format!(
                        "Unsupported operator '{}' in dimension constraint solving: {} {} {} = {}",
                        expr.op, left_name, expr.op, right_val, target
                    )),
                }
            }
            (Dim::Literal(left_val), Dim::Named(right_name)) => {
                // Solve: left op right = target
                let left_usize = *left_val as usize;
                match expr.op {
                    BinOp::Add => {
                        if target >= left_usize {
                            self.resolve_dim(right_name.clone(), target - left_usize)
                        } else {
                            Err(format!(
                                "Illegal negative dimension: {} + {} = {}",
                                left_val, right_name, target
                            ))
                        }
                    }
                    BinOp::Sub => {
                        // left - right = target  =>  right = left - target
                        if left_usize >= target {
                            self.resolve_dim(right_name.clone(), left_usize - target)
                        } else {
                            Err(format!(
                                "Illegal negative dimension: {} - {} = {}",
                                left_val, right_name, target
                            ))
                        }
                    }
                    BinOp::Mul => {
                        if left_usize == 0 {
                            return Err(format!(
                                "Division by zero: cannot solve 0 * {} = {}",
                                right_name, target
                            ));
                        }
                        #[allow(clippy::manual_is_multiple_of)]
                        if target % left_usize != 0 {
                            return Err(format!(
                                "Cannot solve {} * {} = {}: {} is not divisible by {}",
                                left_val, right_name, target, target, left_val
                            ));
                        }
                        self.resolve_dim(right_name.clone(), target / left_usize)
                    }
                    BinOp::Div => {
                        // left / right = target  =>  right = left / target
                        if target == 0 {
                            return Err(format!(
                                "Division by zero: cannot solve {} / {} = 0",
                                left_val, right_name
                            ));
                        }
                        #[allow(clippy::manual_is_multiple_of)]
                        if left_usize % target == 0 {
                            self.resolve_dim(right_name.clone(), left_usize / target)
                        } else {
                            Err(format!(
                                "Cannot solve {} / {} = {}: {} is not divisible by {}",
                                left_val, right_name, target, left_val, target
                            ))
                        }
                    }
                    _ => Err(format!(
                        "Unsupported operator '{}' in dimension constraint solving: {} {} {} = {}",
                        expr.op, left_val, expr.op, right_name, target
                    )),
                }
            }
            // Solve recursive expressions
            (Dim::Expr(left_expr), Dim::Literal(right_val)) => {
                // Solve: left op right = target  =>  left = target inv_op right
                let left_val = match expr.op {
                    BinOp::Add => {
                        if target >= *right_val as usize {
                            Some(target - *right_val as usize)
                        } else {
                            None
                        }
                    }
                    BinOp::Sub => Some(target + *right_val as usize),
                    BinOp::Mul => {
                        #[allow(clippy::manual_is_multiple_of)]
                        if target % (*right_val as usize) != 0 {
                            return Err(format!(
                                "Cannot solve expr * {} = {}: {} is not divisible by {}",
                                right_val, target, target, right_val
                            ));
                        }
                        Some(target / (*right_val as usize))
                    }
                    BinOp::Div => Some(target * (*right_val as usize)),
                    _ => {
                        return Err(format!(
                            "Unsupported operator '{}' in dimension constraint solving for expression",
                            expr.op
                        ));
                    }
                };

                if let Some(val) = left_val {
                    self.solve_expr_for_unknown(left_expr, val)
                } else {
                    Err(format!(
                        "Cannot solve dimension constraint: expr {} {} = {}",
                        expr.op, right_val, target
                    ))
                }
            }
            (Dim::Literal(left_val), Dim::Expr(right_expr)) => {
                // Solve: left op right = target  =>  right = ...
                let right_val = match expr.op {
                    BinOp::Add => {
                        if target >= *left_val as usize {
                            Some(target - *left_val as usize)
                        } else {
                            None
                        }
                    }
                    BinOp::Sub => {
                        if *left_val as usize >= target {
                            Some(*left_val as usize - target)
                        } else {
                            None
                        }
                    }
                    BinOp::Mul => {
                        #[allow(clippy::manual_is_multiple_of)]
                        if target % (*left_val as usize) != 0 {
                            return Err(format!(
                                "Cannot solve {} * expr = {}: {} is not divisible by {}",
                                left_val, target, target, left_val
                            ));
                        }
                        Some(target / (*left_val as usize))
                    }
                    BinOp::Div =>
                    {
                        #[allow(clippy::manual_is_multiple_of)]
                        if target != 0 && (*left_val as usize) % target == 0 {
                            Some((*left_val as usize) / target)
                        } else {
                            None
                        }
                    }
                    _ => {
                        return Err(format!(
                            "Unsupported operator '{}' in dimension constraint solving for expression",
                            expr.op
                        ));
                    }
                };

                if let Some(val) = right_val {
                    self.solve_expr_for_unknown(right_expr, val)
                } else {
                    Err(format!(
                        "Cannot solve dimension constraint: {} {} expr = {}",
                        left_val, expr.op, target
                    ))
                }
            }
            _ => Err(format!(
                "Cannot solve dimension constraint: expression with operator '{}' is too complex \
                 (only single-unknown equations with +, -, *, / are supported)",
                expr.op
            )),
        }
    }

    /// Evaluate a DimExpr with current resolved dimensions
    pub fn evaluate_expr(&self, expr: &DimExpr) -> Option<usize> {
        let left = match &expr.left {
            Dim::Literal(n) => Some(*n as usize),
            Dim::Named(name) => self.resolved_dims.get(name).copied(),
            Dim::Expr(e) => self.evaluate_expr(e),
            _ => None,
        }?;
        let right = match &expr.right {
            Dim::Literal(n) => Some(*n as usize),
            Dim::Named(name) => self.resolved_dims.get(name).copied(),
            Dim::Expr(e) => self.evaluate_expr(e),
            _ => None,
        }?;
        match expr.op {
            BinOp::Add => left.checked_add(right),
            BinOp::Sub => left.checked_sub(right),
            BinOp::Mul => left.checked_mul(right),
            BinOp::Div => left.checked_div(right),
            _ => None,
        }
    }

    /// Evaluate a dimension
    pub fn evaluate_dim(&self, dim: &Dim) -> Option<usize> {
        match dim {
            Dim::Literal(n) => Some(*n as usize),
            Dim::Named(name) => self.resolved_dims.get(name).copied(),
            Dim::Expr(e) => self.evaluate_expr(e),
            Dim::Global(name) => self.resolved_dims.get(name).copied(),
            _ => None,
        }
    }
}

#[derive(Default)]
pub struct ShapeInferenceEngine;

impl ShapeInferenceEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn infer(&mut self, program: &Program) -> Result<(), Vec<ShapeError>> {
        let mut errors = Vec::new();

        for neuron in program.neurons.values() {
            if let Err(e) = self.infer_neuron(neuron, program) {
                errors.push(e);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn infer_neuron(&mut self, neuron: &NeuronDef, program: &Program) -> Result<(), ShapeError> {
        if neuron.is_primitive() {
            return Ok(());
        }

        let mut ctx = InferenceContext::new();

        // 1. Initialize context with known params
        for param in &neuron.params {
            if let Some(Value::Int(val)) = param.default {
                ctx.resolved_dims.insert(param.name.clone(), val as usize);
            }
        }

        // 2. Register input shapes
        // "in" node outputs = neuron inputs
        let input_shapes: Vec<Shape> = neuron.inputs.iter().map(|p| p.shape.clone()).collect();
        ctx.node_outputs
            .insert("in".to_string(), input_shapes.clone());

        // Also register individual input ports if they are named
        for port in &neuron.inputs {
            if port.name != "default" {
                ctx.node_outputs
                    .insert(port.name.clone(), vec![port.shape.clone()]);
            }
        }

        // 3. Walk the graph
        if let NeuronBody::Graph { connections, .. } = &neuron.body {
            for conn in connections {
                self.check_connection(conn, &mut ctx, program)?;
            }
        }

        // 4. Flush pending constraints — retry now that more dims may be resolved
        self.flush_pending_constraints(&mut ctx)?;

        Ok(())
    }

    /// Retry solving pending constraints after all direct unifications complete.
    /// Constraints are deferred when one operand is unknown at the time of initial
    /// unification but may have been resolved by later connections.
    /// Loops until no further progress is made, handling chains of 3+ dependencies.
    fn flush_pending_constraints(
        &self,
        ctx: &mut InferenceContext,
    ) -> Result<(), ShapeError> {
        loop {
            let constraints = std::mem::take(&mut ctx.pending_constraints);
            if constraints.is_empty() {
                break;
            }

            let mut still_pending = Vec::new();
            let mut made_progress = false;

            for (result_dim, expr, constraint_ctx) in constraints {
                // Try to evaluate the result dimension now
                let target = ctx.evaluate_dim(&result_dim);
                if let Some(target_val) = target {
                    // Result dim is now known — solve the expression
                    if let Err(e) = ctx.solve_expr_for_unknown(&expr, target_val) {
                        return Err(ShapeError::ConstraintViolation {
                            message: format!("Deferred constraint failed: {}", e),
                            context: constraint_ctx,
                        });
                    }
                    made_progress = true;
                } else {
                    // Still can't resolve — retry on next iteration
                    still_pending.push((result_dim, expr, constraint_ctx));
                }
            }

            ctx.pending_constraints = still_pending;

            if !made_progress {
                // No constraints were resolved this pass — remaining are truly unresolvable
                break;
            }
        }
        Ok(())
    }

    pub fn check_connection(
        &self,
        conn: &Connection,
        ctx: &mut InferenceContext,
        program: &Program,
    ) -> Result<(), ShapeError> {
        // Build connection context for better error messages
        let conn_context = format!(
            "{} -> {}",
            self.format_endpoint(&conn.source),
            self.format_endpoint(&conn.destination)
        );

        // 1. Resolve source shapes
        let source_shapes = resolve_endpoint_shape(&conn.source, ctx, program).map_err(|e| {
            // Add connection context to error
            match e {
                ShapeError::UnknownNode(node) => {
                    ShapeError::UnknownNode(format!("{} (in connection: {})", node, conn_context))
                }
                ShapeError::NodeInferenceFailed { node, message } => {
                    ShapeError::NodeInferenceFailed {
                        node,
                        message: format!("{} (in connection: {})", message, conn_context),
                    }
                }
                _ => e,
            }
        })?;

        // 2. Validate and process destination
        match &conn.destination {
            Endpoint::Call {
                name,
                args: _,
                kwargs: _,
                id,
                frozen: _,
            } => {
                // Validate call destination
                let called_neuron = program.neurons.get(name).ok_or_else(|| {
                    ShapeError::UnknownNode(format!("{} (called in connection)", name))
                })?;

                // Check for variadic input port
                let has_variadic_input = called_neuron.inputs.len() == 1
                    && called_neuron.inputs[0].variadic;

                // Validate input arity (skip for variadic inputs)
                if !has_variadic_input && source_shapes.len() != called_neuron.inputs.len() {
                    return Err(ShapeError::Mismatch {
                         expected: Shape { dims: vec![] },
                         got: Shape { dims: vec![] },
                         context: format!(
                             "Arity mismatch calling {}: expected {} input(s), got {}. Connection: {} -> {}()",
                             name,
                             called_neuron.inputs.len(),
                             source_shapes.len(),
                             self.format_endpoint(&conn.source),
                             name
                         )
                     });
                }

                // Create call context to isolate variadic bindings
                let mut call_ctx = ctx.clone();

                if has_variadic_input {
                    // Variadic: validate each source shape individually against the single
                    // variadic port's shape pattern. Cross-input compatibility (e.g., all
                    // non-concat dims must match) is left to the runtime — the type system
                    // only ensures each input satisfies the declared shape constraint.
                    let variadic_port = &called_neuron.inputs[0];
                    for (i, src_shape) in source_shapes.iter().enumerate() {
                        self.validate_connection_shapes(src_shape, &variadic_port.shape, &mut call_ctx, &format!(
                            "Connection: {} -> {}() variadic input {} ({})",
                            self.format_endpoint(&conn.source),
                            name,
                            i,
                            variadic_port.name
                        )).map_err(|msg| {
                            ShapeError::ConstraintViolation {
                                message: format!("Variadic input {} shape mismatch: {}", i, msg),
                                context: format!(
                                    "Connection: {} -> {}()\n  Source shape: {}\n  Expected shape: {}\n  Resolved dimensions: {:?}",
                                    self.format_endpoint(&conn.source),
                                    name,
                                    src_shape,
                                    variadic_port.shape,
                                    call_ctx.resolved_dims.iter().map(|(k, v)| format!("{}={}", k, v)).collect::<Vec<_>>().join(", ")
                                ),
                            }
                        })?;
                    }
                } else {
                    // Validate each input shape with full compatibility checking
                    // Use call_ctx to capture variadics
                    for (i, (src_shape, input_port)) in source_shapes
                        .iter()
                        .zip(called_neuron.inputs.iter())
                        .enumerate()
                    {
                        // Check shape compatibility
                        self.validate_connection_shapes(src_shape, &input_port.shape, &mut call_ctx, &format!(
                            "Connection: {} -> {}() input {} ({})",
                            self.format_endpoint(&conn.source),
                            name,
                            i,
                            input_port.name
                        )).map_err(|msg| {
                            ShapeError::ConstraintViolation {
                                message: format!("Input {} ({}) shape mismatch: {}", i, input_port.name, msg),
                            context: format!(
                                "Connection: {} -> {}()\n  Source shape: {}\n  Expected shape: {}\n  Resolved dimensions: {:?}",
                                self.format_endpoint(&conn.source),
                                name,
                                src_shape,
                                input_port.shape,
                                call_ctx.resolved_dims.iter().map(|(k, v)| format!("{}={}", k, v)).collect::<Vec<_>>().join(", ")
                            ),
                            }
                        })?;
                    }
                }

                // Compute output shapes
                // Substitute variadics captured during input validation
                // Also substitute parameter values from args (TODO)
                let output_shapes = called_neuron
                    .outputs
                    .iter()
                    .map(|p| self.substitute_variadics(&p.shape, &call_ctx))
                    .collect();

                ctx.call_outputs.insert(*id, output_shapes);
            }

            Endpoint::Ref(port_ref) => {
                // Validate reference destination
                self.validate_port_ref_destination(port_ref, &source_shapes, ctx, program)?;

                if port_ref.node == "out" {
                    // Special case: validate against neuron output signature
                    // This is done at the end of neuron inference
                } else {
                    // Intermediate node - store shapes
                    if port_ref.port != "default" {
                        // Named port access - validate it exists
                        // For now, just store with the node name
                        ctx.node_outputs
                            .insert(port_ref.node.clone(), source_shapes.to_vec());
                    } else {
                        // Default port
                        ctx.node_outputs
                            .insert(port_ref.node.clone(), source_shapes.to_vec());
                    }
                }
            }

            Endpoint::Tuple(refs) => {
                if source_shapes.len() == 1 && refs.len() > 1 {
                    // Implicit fork: replicate single shape to all bindings
                    let shape = &source_shapes[0];
                    for (i, port_ref) in refs.iter().enumerate() {
                        validate_tuple_port_ref(port_ref, i)?;
                        ctx.node_outputs
                            .insert(port_ref.node.clone(), vec![shape.clone()]);
                    }
                } else if source_shapes.len() != refs.len() {
                    return Err(ShapeError::ConstraintViolation {
                        message: format!(
                            "Tuple unpacking arity mismatch: source produces {} output(s), but {} binding(s) provided",
                            source_shapes.len(),
                            refs.len(),
                        ),
                        context: format!(
                            "Connection: {} -> ({})",
                            self.format_endpoint(&conn.source),
                            refs.iter().map(|r| r.node.as_str()).collect::<Vec<_>>().join(", ")
                        ),
                    });
                } else {
                    // Standard 1:1 tuple unpacking
                    for (i, (shape, port_ref)) in source_shapes.iter().zip(refs.iter()).enumerate() {
                        validate_tuple_port_ref(port_ref, i)?;
                        ctx.node_outputs
                            .insert(port_ref.node.clone(), vec![shape.clone()]);
                    }
                }
            }

            Endpoint::Match(match_expr) => {
                // Validate match expression
                self.validate_match_destination(&match_expr.arms, &source_shapes, ctx, program)?;
            }

            Endpoint::If(if_expr) => {
                self.validate_if_destination(if_expr, &source_shapes, ctx, program)?;
            }

            Endpoint::Reshape(reshape) => {
                // Reshape: compute the output shape from the reshape dims
                // and store it keyed by the reshape's unique id
                let output_shape = reshape.to_shape();
                ctx.call_outputs.insert(reshape.id, vec![output_shape]);
            }

            Endpoint::Wrap(_) => {
                // @wrap is desugared before shape inference
            }
        }

        Ok(())
    }

    fn validate_if_destination(
        &self,
        if_expr: &IfExpr,
        source_shapes: &[Shape],
        ctx: &mut InferenceContext,
        program: &Program,
    ) -> Result<(), ShapeError> {
        let mut branch_outputs = Vec::new();

        // Helper to validate a pipeline branch
        let check_branch = |pipeline: &[Endpoint]| -> Result<Vec<Shape>, ShapeError> {
            let mut branch_ctx = ctx.clone();
            let mut current_shapes = source_shapes.to_vec();

            for endpoint in pipeline {
                // Synthesize connection
                let temp_conn = Connection {
                    source: Endpoint::Ref(PortRef::new("_temp")), // Dummy source
                    destination: endpoint.clone(),
                };

                // Inject shapes
                branch_ctx
                    .node_outputs
                    .insert("_temp".to_string(), current_shapes.clone());

                // Validate
                self.check_connection(&temp_conn, &mut branch_ctx, program)?;

                // Resolve output
                current_shapes = resolve_match_endpoint(endpoint, &branch_ctx, program)?;
            }
            Ok(current_shapes)
        };

        // Check if/elifs
        for branch in &if_expr.branches {
            let shapes = check_branch(&branch.pipeline)?;
            branch_outputs.push(shapes);
        }

        // Check else
        if let Some(else_pipeline) = &if_expr.else_branch {
            let shapes = check_branch(else_pipeline)?;
            branch_outputs.push(shapes);
        } else {
            // Implicit else: Identity / Pass-through
            branch_outputs.push(source_shapes.to_vec());
        }

        // Validate consistency across all branches
        if branch_outputs.is_empty() {
            return Ok(());
        }

        let first_branch = &branch_outputs[0];
        for (i, other_branch) in branch_outputs.iter().skip(1).enumerate() {
            // Check arity
            if first_branch.len() != other_branch.len() {
                return Err(ShapeError::ConstraintViolation {
                    message: format!(
                        "Arity mismatch: expected {} outputs, got {}",
                        first_branch.len(),
                        other_branch.len()
                    ),
                    context: format!(
                        "If branches produce different number of outputs: branch 0 produces {} outputs, branch {} produces {} outputs",
                        first_branch.len(),
                        i + 1,
                        other_branch.len()
                    ),
                });
            }

            // Check each shape
            for (j, (s_first, s_other)) in first_branch.iter().zip(other_branch.iter()).enumerate()
            {
                if !self.shapes_compatible(s_first, s_other) {
                    return Err(ShapeError::Mismatch {
                        expected: s_first.clone(),
                        got: s_other.clone(),
                        context: format!(
                            "If branches produce incompatible output shapes at index {}: branch 0 produces {}, branch {} produces {}",
                            j,
                            s_first,
                            i + 1,
                            s_other
                        ),
                    });
                }
            }
        }

        // Store resolved output shapes for this If expression
        ctx.call_outputs.insert(if_expr.id, first_branch.clone());

        Ok(())
    }

    fn validate_port_ref_destination(
        &self,
        port_ref: &PortRef,
        _source_shapes: &[Shape],
        ctx: &InferenceContext,
        _program: &Program,
    ) -> Result<(), ShapeError> {
        if port_ref.node == "in" {
            return Err(ShapeError::ConstraintViolation {
                message: "Cannot use 'in' as connection destination".to_string(),
                context: format!("Port reference: {}", format_port_ref(port_ref)),
            });
        }

        // Check for multiple assignments?
        if let Some(_existing) = ctx.node_outputs.get(&port_ref.node) {
            // Multiple assignments are allowed and result in unification?
            // Or only one source per node is allowed?
            // NeuroScript currently allows multiple sources resulting in a cycle check.
        }

        Ok(())
    }

    fn validate_match_destination(
        &self,
        arms: &[MatchArm],
        source_shapes: &[Shape],
        ctx: &mut InferenceContext,
        program: &Program,
    ) -> Result<(), ShapeError> {
        let mut arm_outputs = Vec::new();

        for arm in arms {
            // Each arm gets its own isolated context for capture
            let mut arm_ctx = ctx.clone();

            // Source shapes must be unifiable with match pattern
            // Combine all source shapes into a single pattern match?
            // Match expressions often take a single input.
            if source_shapes.is_empty() {
                let empty_shape = Shape::new(vec![]);
                return Err(ShapeError::Mismatch {
                    expected: arm.pattern.as_shape().cloned().unwrap_or_else(|| empty_shape.clone()),
                    got: empty_shape,
                    context: "Match source produces no output".to_string(),
                });
            }

            // Unify with ARM pattern (only for Shape patterns)
            if let Some(shape_pattern) = arm.pattern.as_shape() {
                self.unify_pattern_with_shape(shape_pattern, &source_shapes[0], &mut arm_ctx)
                    .map_err(|msg| ShapeError::ConstraintViolation {
                        message: format!("Pattern mismatch in match arm: {}", msg),
                        context: format!(
                            "Pattern: {}\nSource shape: {}",
                            shape_pattern, source_shapes[0]
                        ),
                    })?;
            }

            // Track current output shapes as we walk the arm pipeline
            let mut current_shapes = source_shapes.to_vec();

            for endpoint in &arm.pipeline {
                // Synthesize a connection from previous output to this endpoint
                let temp_conn = Connection {
                    source: Endpoint::Ref(PortRef::new("_temp")), // Dummy source
                    destination: endpoint.clone(),
                };

                // Inject current_shapes into context for the dummy source
                {
                    arm_ctx
                        .node_outputs
                        .insert("_temp".to_string(), current_shapes.clone());
                }

                // Validate this connection
                self.check_connection(&temp_conn, &mut arm_ctx, program)?;

                // Get output shapes from this endpoint
                current_shapes = resolve_match_endpoint(endpoint, &arm_ctx, program)?;
            }

            // Collect output shapes from this arm
            arm_outputs.push(current_shapes);
        }

        // Validate that all arms produce compatible output shapes
        if arm_outputs.is_empty() {
            return Ok(());
        }

        let first_arm = &arm_outputs[0];
        for (i, other_arm) in arm_outputs.iter().skip(1).enumerate() {
            // Check arity
            if first_arm.len() != other_arm.len() {
                return Err(ShapeError::ConstraintViolation {
                    message: format!(
                        "Match arms produce different number of outputs: arm 0 produces {} outputs, arm {} produces {} outputs",
                        first_arm.len(),
                        i + 1,
                        other_arm.len()
                    ),
                    context: "Match expression outputs must have consistent arity across all arms"
                        .to_string(),
                });
            }

            // Check each shape
            for (j, (s_first, s_other)) in first_arm.iter().zip(other_arm.iter()).enumerate() {
                if !self.shapes_compatible(s_first, s_other) {
                    return Err(ShapeError::Mismatch {
                        expected: s_first.clone(),
                        got: s_other.clone(),
                        context: format!(
                            "Match arms produce incompatible output shapes at index {}: arm 0 produces {}, arm {} produces {}",
                            j,
                            s_first,
                            i + 1,
                            s_other
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    /// Unify a pattern shape with a concrete shape, binding captured dimensions
    pub(crate) fn unify_pattern_with_shape(
        &self,
        pattern: &Shape,
        concrete: &Shape,
        ctx: &mut InferenceContext,
    ) -> Result<(), String> {
        // Handle variadic patterns
        if self.has_variadic(pattern) {
            return self.unify_with_variadic_pattern(
                pattern,
                pattern
                    .dims
                    .iter()
                    .position(|d| matches!(d, Dim::Variadic(_)))
                    .unwrap(),
                concrete,
                ctx,
            );
        }

        if pattern.rank() != concrete.rank() {
            return Err(format!(
                "Rank mismatch: pattern rank {}, concrete rank {}",
                pattern.rank(),
                concrete.rank()
            ));
        }

        for (pat_dim, conc_dim) in pattern.dims.iter().zip(concrete.dims.iter()) {
            ctx.unify(pat_dim, conc_dim)?;
        }

        Ok(())
    }

    pub(crate) fn has_variadic(&self, s: &Shape) -> bool {
        s.dims.iter().any(|d| matches!(d, Dim::Variadic(_)))
    }

    fn unify_shapes_with_variadic(
        &self,
        s1: &Shape,
        s2: &Shape,
        ctx: &mut InferenceContext,
    ) -> Result<(), String> {
        // Find variadic positions
        let s1_variadic_pos = s1.dims.iter().position(|d| matches!(d, Dim::Variadic(_)));
        let s2_variadic_pos = s2.dims.iter().position(|d| matches!(d, Dim::Variadic(_)));

        match (s1_variadic_pos, s2_variadic_pos) {
            (Some(pos1), None) => {
                // s1 has variadic, s2 doesn't
                self.unify_with_variadic_pattern(s1, pos1, s2, ctx)
            }
            (None, Some(pos2)) => {
                // s2 has variadic, s1 doesn't
                self.unify_with_variadic_pattern(s2, pos2, s1, ctx)
            }
            (Some(pos1), Some(pos2)) => {
                // Both have variadics - complex case
                // Instead of strict equality, check structural compatibility:
                // 1. Match constant prefixes
                // 2. Match constant suffixes
                // 3. Assume variadics absorb the difference

                // 1. Unify common prefix
                let common_prefix_len = std::cmp::min(pos1, pos2);
                for (i, (d1, d2)) in s1.dims[..common_prefix_len]
                    .iter()
                    .zip(s2.dims[..common_prefix_len].iter())
                    .enumerate()
                {
                    ctx.unify(d1, d2).map_err(|e| {
                        format!(
                            "Prefix dimension {} mismatch in variadic unification: {}",
                            i, e
                        )
                    })?;
                }

                // 2. Unify common suffix
                let tail1_len = s1.dims.len() - 1 - pos1;
                let tail2_len = s2.dims.len() - 1 - pos2;
                let common_suffix_len = std::cmp::min(tail1_len, tail2_len);

                let suffix1_start = s1.dims.len() - common_suffix_len;
                let suffix2_start = s2.dims.len() - common_suffix_len;

                for (i, (d1, d2)) in s1.dims[suffix1_start..]
                    .iter()
                    .zip(s2.dims[suffix2_start..].iter())
                    .enumerate()
                {
                    ctx.unify(d1, d2).map_err(|e| {
                        format!(
                            "Suffix dimension {} mismatch in variadic unification: {}",
                            i, e
                        )
                    })?;
                }

                // 3. Middle gap: fixed dims between each variadic and the
                // prefix/suffix boundaries are absorbed by the other shape's
                // variadic. Prefix unification already checks dims before
                // min(pos1, pos2), and suffix unification checks dims after
                // each variadic's tail. Any remaining fixed dims in the gap
                // (e.g., [A, C, *y, B] vs [A, *x, B] — C is absorbed by *x)
                // are unconstrained and valid by construction.

                Ok(())
            }
            (None, None) => {
                unreachable!("has_variadic returned true but no variadic found")
            }
        }
    }

    fn unify_with_variadic_pattern(
        &self,
        pattern: &Shape,
        variadic_pos: usize,
        concrete: &Shape,
        ctx: &mut InferenceContext,
    ) -> Result<(), String> {
        let prefix_len = variadic_pos;
        let suffix_len = pattern.dims.len() - variadic_pos - 1;

        if concrete.dims.len() < prefix_len + suffix_len {
            return Err(format!(
                "Shape {} too short to match pattern {} (needs at least {} dimensions, has {})",
                concrete,
                pattern,
                prefix_len + suffix_len,
                concrete.dims.len()
            ));
        }

        // Unify prefix
        for (i, (pat_dim, conc_dim)) in pattern.dims[..prefix_len]
            .iter()
            .zip(concrete.dims[..prefix_len].iter())
            .enumerate()
        {
            ctx.unify(pat_dim, conc_dim)
                .map_err(|e| format!("Prefix dimension {} mismatch: {}", i, e))?;
        }

        // Unify suffix
        let concrete_suffix_start = concrete.dims.len() - suffix_len;
        for (i, (pat_dim, conc_dim)) in pattern.dims[variadic_pos + 1..]
            .iter()
            .zip(concrete.dims[concrete_suffix_start..].iter())
            .enumerate()
        {
            ctx.unify(pat_dim, conc_dim)
                .map_err(|e| format!("Suffix dimension {} mismatch: {}", i, e))?;
        }

        // Capture variadic segment
        if let Dim::Variadic(name) = &pattern.dims[variadic_pos] {
            let variadic_dims = &concrete.dims[prefix_len..concrete_suffix_start];
            let segment = variadic_dims.to_vec();

            if let Some(existing) = ctx.resolved_variadics.get(name).cloned() {
                // If already bound, they must be equal rank and unify element-wise
                if existing.len() != segment.len() {
                    return Err(format!("Variadic binding mismatch for {}: existing variant rank {}, new variant rank {}", name, existing.len(), segment.len()));
                }
                // Deep unification: unify each dim pair
                for (existing_dim, new_dim) in existing.iter().zip(segment.iter()) {
                    ctx.unify(existing_dim, new_dim).map_err(|e| {
                        format!("Variadic '{}' deep unification failed: {}", name, e)
                    })?;
                }
            } else {
                ctx.resolved_variadics.insert(name.clone(), segment);
            }
        }

        Ok(())
    }

    fn substitute_variadics(&self, shape: &Shape, ctx: &InferenceContext) -> Shape {
        let mut new_dims = Vec::new();

        for dim in &shape.dims {
            match dim {
                Dim::Variadic(name) => {
                    if let Some(segment) = ctx.resolved_variadics.get(name) {
                        new_dims.extend(segment.clone());
                    } else {
                        new_dims.push(dim.clone());
                    }
                }
                Dim::Expr(expr) => {
                    // Try to evaluate expression
                    if let Some(val) = ctx.evaluate_expr(expr) {
                        new_dims.push(Dim::Literal(val as i64));
                    } else {
                        new_dims.push(dim.clone());
                    }
                }
                Dim::Named(name) => {
                    if let Some(val) = ctx.resolved_dims.get(name) {
                        new_dims.push(Dim::Literal(*val as i64));
                    } else {
                        new_dims.push(dim.clone());
                    }
                }
                _ => new_dims.push(dim.clone()),
            }
        }

        Shape::new(new_dims)
    }

    /// Check if two shapes are compatible (can be unified)
    pub(crate) fn shapes_compatible(&self, s1: &Shape, s2: &Shape) -> bool {
        // Create a temporary context for testing
        let mut test_ctx = InferenceContext::new();
        self.unify_shapes(s1, s2, &mut test_ctx).is_ok()
    }

    fn format_endpoint(&self, ep: &Endpoint) -> String {
        match ep {
            Endpoint::Ref(r) => format_port_ref(r),
            Endpoint::Call { name, .. } => format!("{}()", name),
            Endpoint::Tuple(refs) => format!(
                "({})",
                refs.iter()
                    .map(format_port_ref)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Endpoint::Match(_) => "match".to_string(),
            Endpoint::If(_) => "if".to_string(),
            Endpoint::Reshape(reshape) => {
                format!(
                    "=> [{}]",
                    reshape
                        .dims
                        .iter()
                        .map(|d| format!("{}", d))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Endpoint::Wrap(w) => format!("@wrap({})", w.wrapper_name),
            // Endpoint::Unroll removed
        }
    }

    pub(crate) fn unify_shapes(
        &self,
        s1: &Shape,
        s2: &Shape,
        ctx: &mut InferenceContext,
    ) -> Result<(), String> {
        // Handle variadic dimensions first
        if self.has_variadic(s1) || self.has_variadic(s2) {
            return self.unify_shapes_with_variadic(s1, s2, ctx);
        }

        if s1.rank() != s2.rank() {
            return Err(format!(
                "Rank mismatch: {} (rank {}) != {} (rank {})",
                s1,
                s1.rank(),
                s2,
                s2.rank()
            ));
        }

        for (d1, d2) in s1.dims.iter().zip(s2.dims.iter()) {
            ctx.unify(d1, d2)?;
        }

        Ok(())
    }

    fn validate_connection_shapes(
        &self,
        source: &Shape,
        expected: &Shape,
        ctx: &mut InferenceContext,
        context: &str,
    ) -> Result<(), String> {
        self.unify_shapes(source, expected, ctx).map_err(|e| {
            format!(
                "{} - Incompatible shapes: {} vs {}\n  Reason: {}",
                context, source, expected, e
            )
        })?;

        // Check expressions
        for dim in &expected.dims {
            if let Dim::Expr(expr) = dim {
                self.validate_dim_expr(expr, ctx, context)?;
            }
        }

        Ok(())
    }

    fn validate_dim_expr(
        &self,
        expr: &DimExpr,
        ctx: &InferenceContext,
        context: &str,
    ) -> Result<(), String> {
        // Check if we can evaluate the expression with current context
        match ctx.evaluate_expr(expr) {
            Some(_value) => {
                // Expression is satisfied
                Ok(())
            }
            None => {
                // Check if we have enough information to potentially solve it
                let left_resolvable = is_dim_resolvable(&expr.left, ctx);
                let right_resolvable = is_dim_resolvable(&expr.right, ctx);

                if !left_resolvable || !right_resolvable {
                    Err(format!(
                        "Cannot resolve expression dimension: {} {} {}\n  Missing: {}{}\n  Context: {}",
                        expr.left,
                        match expr.op {
                            BinOp::Add => "+",
                            BinOp::Sub => "-",
                            BinOp::Mul => "*",
                            BinOp::Div => "/",
                            _ => "?",
                        },
                        expr.right,
                        if !left_resolvable { format!("{} ", expr.left) } else { String::new() },
                        if !right_resolvable { format!("{}", expr.right) } else { String::new() },
                        context
                    ))
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// Resolve the output shapes of a match endpoint
fn resolve_match_endpoint(
    endpoint: &Endpoint,
    ctx: &InferenceContext,
    program: &Program,
) -> Result<Vec<Shape>, ShapeError> {
    match endpoint {
        Endpoint::Ref(port_ref) => {
            if port_ref.node == "out" {
                return Ok(vec![]);
            }
            // Look up the output shape for this reference
            if let Some(shapes) = ctx.node_outputs.get(&port_ref.node) {
                Ok(shapes.clone())
            } else {
                Err(ShapeError::UnknownNode(format!(
                    "Unknown node '{}' in match pipeline",
                    port_ref.node
                )))
            }
        }
        Endpoint::Call { name, id, .. } => {
            // Look up output from call_outputs
            if let Some(shapes) = ctx.call_outputs.get(id) {
                Ok(shapes.clone())
            } else {
                // Get output shapes from neuron definition
                if let Some(neuron) = program.neurons.get(name) {
                    Ok(neuron.outputs.iter().map(|p| p.shape.clone()).collect())
                } else {
                    Err(ShapeError::UnknownNode(format!(
                        "Unknown neuron '{}' in match pipeline",
                        name
                    )))
                }
            }
        }
        Endpoint::Tuple(refs) => {
            let mut shapes = Vec::new();
            for r in refs {
                let s = resolve_match_endpoint(&Endpoint::Ref(r.clone()), ctx, program)?;
                shapes.extend(s);
            }
            Ok(shapes)
        }
        Endpoint::Match(_) => Err(ShapeError::UnsupportedFeature(
            "Nested match expressions not yet supported".to_string(),
        )),
        Endpoint::If(_) => Err(ShapeError::UnsupportedFeature(
            "Nested if expressions not yet supported in match pipeline".to_string(),
        )),
        Endpoint::Reshape(reshape) => {
            // In a match pipeline, a reshape produces its declared output shape
            if let Some(shapes) = ctx.call_outputs.get(&reshape.id) {
                Ok(shapes.clone())
            } else {
                Ok(vec![reshape.to_shape()])
            }
        }
        Endpoint::Wrap(_) => {
            // @wrap is desugared before shape inference
            Ok(vec![])
        }
        // Endpoint::Unroll removed — expanded before shape inference
    }
}

/// Validate that a tuple binding uses a simple name (no port access like `node.port`)
fn validate_tuple_port_ref(port_ref: &PortRef, position: usize) -> Result<(), ShapeError> {
    if port_ref.port != "default" {
        return Err(ShapeError::ConstraintViolation {
            message: format!(
                "Cannot use port access in tuple unpacking position {}",
                position
            ),
            context: format!(
                "Tuple binding {} should be a simple name, not {}",
                position,
                format_port_ref(port_ref)
            ),
        });
    }
    Ok(())
}

fn format_port_ref(r: &PortRef) -> String {
    if r.port != "default" {
        format!("{}.{}", r.node, r.port)
    } else {
        r.node.clone()
    }
}

fn resolve_endpoint_shape(
    endpoint: &Endpoint,
    ctx: &InferenceContext,
    _program: &Program,
) -> Result<Vec<Shape>, ShapeError> {
    match endpoint {
        Endpoint::Ref(port_ref) => {
            // println!("DEBUG: Resolving Ref: node='{}' port='{}'", port_ref.node, port_ref.port);
            if port_ref.node == "out" {
                return Ok(vec![]);
            }
            if let Some(shapes) = ctx.node_outputs.get(&port_ref.node) {
                // TODO: Handle port_ref.port selection
                Ok(shapes.clone())
            } else {
                Err(ShapeError::UnknownNode(port_ref.node.clone()))
            }
        }
        Endpoint::Call { id, .. } => {
            if let Some(shapes) = ctx.call_outputs.get(id) {
                Ok(shapes.clone())
            } else {
                // This happens if we try to read from a Call that hasn't been processed as a destination yet?
                // But connections are ordered?
                // Or if it's a source-only call (generator)?
                // If it's a source, we need to instantiate it and get outputs.
                // But we don't have inputs to unify.
                // TODO: Handle source calls
                Ok(vec![])
            }
        }
        Endpoint::Tuple(refs) => {
            let mut shapes = Vec::new();
            for r in refs {
                let s = resolve_endpoint_shape(&Endpoint::Ref(r.clone()), ctx, _program)?;
                shapes.extend(s);
            }
            Ok(shapes)
        }
        Endpoint::Match(_) => Ok(vec![]), // TODO
        Endpoint::If(if_expr) => {
            if let Some(shapes) = ctx.call_outputs.get(&if_expr.id) {
                Ok(shapes.clone())
            } else {
                Ok(vec![])
            }
        }
        Endpoint::Reshape(reshape) => {
            if let Some(shapes) = ctx.call_outputs.get(&reshape.id) {
                Ok(shapes.clone())
            } else {
                Ok(vec![reshape.to_shape()])
            }
        }
        Endpoint::Wrap(_) => {
            // @wrap is desugared before shape inference
            Ok(vec![])
        }
        // Endpoint::Unroll removed
    }
}

/// Check if a dimension can be resolved with current context
pub(crate) fn is_dim_resolvable(dim: &Dim, ctx: &InferenceContext) -> bool {
    match dim {
        Dim::Literal(_) => true,
        Dim::Named(name) => ctx.resolved_dims.contains_key(name),
        Dim::Wildcard => true,
        // Treated as resolvable: PyTorch computes the actual size at runtime from total element count
        Dim::Inferred => true,
        Dim::Variadic(_) => true,
        Dim::Expr(expr) => {
            is_dim_resolvable(&expr.left, ctx) && is_dim_resolvable(&expr.right, ctx)
        }
        Dim::Global(_) => true,
    }
}
