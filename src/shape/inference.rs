use crate::interfaces::*;

impl InferenceContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn resolve_dim(&self, dim: &Dim) -> Option<usize> {
        match dim {
            Dim::Literal(n) => Some(*n as usize),
            Dim::Named(name) => self.resolved_dims.get(name).copied(),
            Dim::Expr(expr) => self.evaluate_expr(expr),
            _ => None,
        }
    }

    pub fn evaluate_expr(&self, expr: &DimExpr) -> Option<usize> {
        let left = self.resolve_dim(&expr.left)?;
        let right = self.resolve_dim(&expr.right)?;

        match expr.op {
            BinOp::Add => Some(left + right),
            BinOp::Sub => Some(left.checked_sub(right)?),
            BinOp::Mul => Some(left * right),
            BinOp::Div => Some(left / right),
            _ => None,
        }
    }

    pub fn unify(&mut self, d1: &Dim, d2: &Dim) -> Result<(), String> {
        match (d1, d2) {
            (Dim::Literal(v1), Dim::Literal(v2)) => {
                if v1 != v2 {
                    return Err(format!("Literal mismatch: {} != {}", v1, v2));
                }
            }
            (Dim::Named(n1), Dim::Named(n2)) => {
                if n1 != n2 {
                    let v1 = self.resolved_dims.get(n1);
                    let v2 = self.resolved_dims.get(n2);

                    if let (Some(val1), Some(val2)) = (v1, v2) {
                        if val1 != val2 {
                            return Err(format!(
                                "Variable mismatch: {}={} != {}={}",
                                n1, val1, n2, val2
                            ));
                        }
                    } else if let Some(val) = v1 {
                        self.resolved_dims.insert(n2.clone(), *val);
                    } else if let Some(val) = v2 {
                        self.resolved_dims.insert(n1.clone(), *val);
                    }
                }
            }
            (Dim::Named(n), Dim::Literal(v)) | (Dim::Literal(v), Dim::Named(n)) => {
                if let Some(current) = self.resolved_dims.get(n) {
                    if *current != *v as usize {
                        return Err(format!(
                            "Variable {} already bound to {}, cannot bind to {}",
                            n, current, v
                        ));
                    }
                } else {
                    self.resolved_dims.insert(n.clone(), *v as usize);
                }
            }
            // Unify expressions - try to solve for unknowns
            (Dim::Expr(expr), Dim::Literal(v)) | (Dim::Literal(v), Dim::Expr(expr)) => {
                // Try to solve the expression backwards
                self.solve_expr_for_unknown(expr, *v as usize)?;
            }
            (Dim::Expr(expr), Dim::Named(name)) | (Dim::Named(name), Dim::Expr(expr)) => {
                // If the named dim is resolved, try to solve the expression
                if let Some(value) = self.resolved_dims.get(name).copied() {
                    self.solve_expr_for_unknown(expr, value)?;
                }
            }
            (Dim::Expr(e1), Dim::Expr(e2)) => {
                // Both are expressions - try to evaluate both and unify results
                let v1 = self.evaluate_expr(e1);
                let v2 = self.evaluate_expr(e2);
                if let (Some(val1), Some(val2)) = (v1, v2) {
                    if val1 != val2 {
                        return Err(format!(
                            "Expression mismatch: {} = {} != {} = {}",
                            self.format_expr(e1),
                            val1,
                            self.format_expr(e2),
                            val2
                        ));
                    }
                }
                // Otherwise, we can't unify them yet
            }
            _ => {}
        }
        Ok(())
    }

    /// Attempt to solve an expression for an unknown variable
    /// For example: if expr is "dim * 4" and target is 2048, solve for dim = 512
    pub(crate) fn solve_expr_for_unknown(
        &mut self,
        expr: &DimExpr,
        target: usize,
    ) -> Result<(), String> {
        // Try to solve for the left operand
        if let Dim::Named(left_name) = &expr.left {
            if !self.resolved_dims.contains_key(left_name) {
                if let Some(right_val) = self.resolve_dim(&expr.right) {
                    // Solve: left op right = target  =>  left = target inv_op right
                    let left_val = match expr.op {
                        BinOp::Mul => {
                            if target % right_val != 0 {
                                return Err(format!(
                                    "Cannot solve {} * {} = {}: {} is not divisible by {}",
                                    left_name, right_val, target, target, right_val
                                ));
                            }
                            target / right_val
                        }
                        BinOp::Div => target * right_val,
                        BinOp::Add => {
                            if target < right_val {
                                return Err(format!(
                                    "Cannot solve {} + {} = {}: result would be negative",
                                    left_name, right_val, target
                                ));
                            }
                            target - right_val
                        }
                        BinOp::Sub => target + right_val,
                        _ => return Ok(()), // Can't solve for other ops yet
                    };
                    self.resolved_dims.insert(left_name.clone(), left_val);
                    return Ok(());
                }
            }
        }

        // Try to solve for the right operand
        if let Dim::Named(right_name) = &expr.right {
            if !self.resolved_dims.contains_key(right_name) {
                if let Some(left_val) = self.resolve_dim(&expr.left) {
                    // Solve: left op right = target  =>  right = ...
                    let right_val = match expr.op {
                        BinOp::Mul => {
                            if target % left_val != 0 {
                                return Err(format!(
                                    "Cannot solve {} * {} = {}: {} is not divisible by {}",
                                    left_val, right_name, target, target, left_val
                                ));
                            }
                            target / left_val
                        }
                        BinOp::Div => {
                            if left_val % target != 0 {
                                return Err(format!(
                                    "Cannot solve {} / {} = {}: {} is not divisible by {}",
                                    left_val, right_name, target, left_val, target
                                ));
                            }
                            left_val / target
                        }
                        BinOp::Add => {
                            if target < left_val {
                                return Err(format!(
                                    "Cannot solve {} + {} = {}: result would be negative",
                                    left_val, right_name, target
                                ));
                            }
                            target - left_val
                        }
                        BinOp::Sub => {
                            if left_val < target {
                                return Err(format!(
                                    "Cannot solve {} - {} = {}: result would be negative",
                                    left_val, right_name, target
                                ));
                            }
                            left_val - target
                        }
                        _ => return Ok(()), // Can't solve for other ops yet
                    };
                    self.resolved_dims.insert(right_name.clone(), right_val);
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    fn format_expr(&self, expr: &DimExpr) -> String {
        format!(
            "{} {} {}",
            expr.left,
            match expr.op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                _ => "?",
            },
            expr.right
        )
    }
}
pub struct ShapeInferenceEngine;

impl ShapeInferenceEngine {
    pub fn new() -> Self {
        Self {}
    }

    pub fn infer(&mut self, program: &Program) -> Result<(), Vec<ShapeError>> {
        let mut errors = Vec::new();

        for (_name, neuron) in &program.neurons {
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
        for (_i, port) in neuron.inputs.iter().enumerate() {
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
        let source_shapes = self
            .resolve_endpoint_shape(&conn.source, ctx, program)
            .map_err(|e| {
                // Add connection context to error
                match e {
                    ShapeError::UnknownNode(node) => ShapeError::UnknownNode(format!(
                        "{} (in connection: {})",
                        node, conn_context
                    )),
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
            } => {
                // Validate call destination
                let called_neuron = program.neurons.get(name).ok_or_else(|| {
                    ShapeError::UnknownNode(format!("{} (called in connection)", name))
                })?;

                // Validate input arity
                if source_shapes.len() != called_neuron.inputs.len() {
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
                            .insert(port_ref.node.clone(), source_shapes);
                    } else {
                        // Default port
                        ctx.node_outputs
                            .insert(port_ref.node.clone(), source_shapes);
                    }
                }
            }

            Endpoint::Tuple(refs) => {
                // Validate tuple unpacking
                if source_shapes.len() != refs.len() {
                    return Err(ShapeError::Mismatch {
                        expected: Shape::new(vec![]),
                        got: Shape::new(vec![]),
                        context: format!(
                            "Tuple unpacking arity mismatch: source produces {} output(s), but {} binding(s) provided. Connection: {} -> ({})",
                            source_shapes.len(),
                            refs.len(),
                            self.format_endpoint(&conn.source),
                            refs.iter().map(|r| r.node.as_str()).collect::<Vec<_>>().join(", ")
                        )
                    });
                }

                // Assign each shape to its binding
                for (i, (shape, port_ref)) in source_shapes.iter().zip(refs.iter()).enumerate() {
                    // Validate the port reference is valid
                    if port_ref.port != "default" {
                        return Err(ShapeError::ConstraintViolation {
                            message: format!(
                                "Cannot use port access in tuple unpacking position {}",
                                i
                            ),
                            context: format!(
                                "Tuple binding {} should be a simple name, not {}",
                                i,
                                self.format_port_ref(port_ref)
                            ),
                        });
                    }

                    ctx.node_outputs
                        .insert(port_ref.node.clone(), vec![shape.clone()]);
                }
            }

            Endpoint::Match(match_expr) => {
                // Validate match expression
                self.validate_match_destination(&match_expr.arms, &source_shapes, ctx, program)?;
            }
        }

        Ok(())
    }

    fn validate_port_ref_destination(
        &self,
        port_ref: &PortRef,
        _source_shapes: &[Shape],
        _ctx: &mut InferenceContext,
        _program: &Program,
    ) -> Result<(), ShapeError> {
        // For now, basic validation
        // TODO: Check that if port_ref.port is not "default", the port actually exists on the neuron

        if port_ref.port != "default" {
            // Named port access - in a destination context, this is unusual
            // Typically you'd see this in a source context like "fork.left"
            // In a destination, it means we're assigning to a specific port
            // For MVP, allow it but don't validate port existence yet
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
        if source_shapes.len() != 1 {
            return Err(ShapeError::ConstraintViolation {
                message: format!(
                    "Match expression requires exactly one input, got {}",
                    source_shapes.len()
                ),
                context: "Match expression validation".to_string(),
            });
        }

        let source_shape = &source_shapes[0];
        let mut output_shapes = Vec::new();

        // Validate each arm
        for (arm_idx, arm) in arms.iter().enumerate() {
            // Fork context for this arm to avoid cross-contamination
            let mut arm_ctx = ctx.clone();

            // Unify pattern with source shape and bind captured dimensions
            self.unify_pattern_with_shape(&arm.pattern, source_shape, &mut arm_ctx)
                .map_err(|e| ShapeError::ConstraintViolation {
                    message: format!("Match arm {} pattern mismatch: {}", arm_idx, e),
                    context: format!("Pattern: {}, Source: {}", arm.pattern, source_shape),
                })?;

            // Validate guard expression if present
            if let Some(guard) = &arm.guard {
                self.validate_guard_expr(guard, &arm_ctx).map_err(|e| {
                    ShapeError::ConstraintViolation {
                        message: format!("Match arm {} guard error: {}", arm_idx, e),
                        context: format!("Guard in pattern: {}", arm.pattern),
                    }
                })?;
            }

            // Validate each endpoint in the pipeline with the forked context
            let mut current_shapes = vec![source_shape.clone()];
            for (ep_idx, endpoint) in arm.pipeline.iter().enumerate() {
                // Create a temporary connection for validation
                let temp_conn = Connection {
                    source: if ep_idx == 0 {
                        Endpoint::Ref(PortRef::new("in"))
                    } else {
                        // Use a dummy ref - we track shapes in current_shapes
                        Endpoint::Ref(PortRef::new("_temp"))
                    },
                    destination: endpoint.clone(),
                };

                // Register current shapes in context
                if ep_idx == 0 {
                    arm_ctx
                        .node_outputs
                        .insert("in".to_string(), current_shapes.clone());
                } else {
                    arm_ctx
                        .node_outputs
                        .insert("_temp".to_string(), current_shapes.clone());
                }

                // Validate this connection
                self.check_connection(&temp_conn, &mut arm_ctx, program)?;

                // Get output shapes from this endpoint
                current_shapes = self.resolve_match_endpoint(endpoint, &arm_ctx, program)?;
            }

            // Collect output shape from this arm
            if !current_shapes.is_empty() {
                output_shapes.push(current_shapes[0].clone());
            }
        }

        // Validate that all arms produce compatible output shapes
        if output_shapes.len() > 1 {
            let first_output = &output_shapes[0];
            for (idx, arm_output) in output_shapes.iter().skip(1).enumerate() {
                // Check if outputs are structurally compatible
                if !self.shapes_compatible(first_output, arm_output) {
                    return Err(ShapeError::Mismatch {
                        expected: first_output.clone(),
                        got: arm_output.clone(),
                        context: format!(
                            "Match arms produce incompatible output shapes: arm 0 produces {}, arm {} produces {}",
                            first_output,
                            idx + 1,
                            arm_output
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

        // Check rank matches for non-variadic patterns
        if pattern.dims.len() != concrete.dims.len() {
            return Err(format!(
                "Rank mismatch: pattern has {} dimensions, shape has {}",
                pattern.dims.len(),
                concrete.dims.len()
            ));
        }

        // Unify each dimension
        for (i, (pat_dim, conc_dim)) in pattern.dims.iter().zip(concrete.dims.iter()).enumerate() {
            match (pat_dim, conc_dim) {
                (Dim::Wildcard, _) => {
                    // Wildcard matches anything, no binding
                    continue;
                }
                (Dim::Literal(_lit), _) => {
                    // Literal must match exactly - use unify to check
                    ctx.unify(pat_dim, conc_dim)
                        .map_err(|e| format!("Dimension {} mismatch: {}", i, e))?;
                }
                (Dim::Named(name), _) => {
                    // Named dimension: bind it if not already bound
                    ctx.unify(pat_dim, conc_dim)
                        .map_err(|e| format!("Dimension {} ({}): {}", i, name, e))?;
                }
                (Dim::Expr(_), _) => {
                    // Expression in pattern - unify it
                    ctx.unify(pat_dim, conc_dim)
                        .map_err(|e| format!("Dimension {} expression mismatch: {}", i, e))?;
                }
                (Dim::Variadic(_), _) => {
                    // Shouldn't reach here if has_variadic check worked
                    unreachable!("Variadic should have been handled above")
                }
            }
        }

        Ok(())
    }

    /// Validate a guard expression can be evaluated
    fn validate_guard_expr(&self, _guard: &Value, _ctx: &InferenceContext) -> Result<(), String> {
        // For MVP, assume guard is valid if it compiles
        // TODO: Check that all referenced names are either:
        // 1. Captured dimensions (in ctx.resolved_dims)
        // 2. Neuron parameters
        Ok(())
    }

    /// Resolve the output shapes of a match endpoint
    fn resolve_match_endpoint(
        &self,
        endpoint: &Endpoint,
        ctx: &InferenceContext,
        program: &Program,
    ) -> Result<Vec<Shape>, ShapeError> {
        match endpoint {
            Endpoint::Ref(port_ref) => {
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
                    let s = self.resolve_match_endpoint(&Endpoint::Ref(r.clone()), ctx, program)?;
                    shapes.extend(s);
                }
                Ok(shapes)
            }
            Endpoint::Match(_) => Err(ShapeError::UnsupportedFeature(
                "Nested match expressions not yet supported".to_string(),
            )),
        }
    }

    /// Check if two shapes are compatible (can be unified)
    pub(crate) fn shapes_compatible(&self, s1: &Shape, s2: &Shape) -> bool {
        // Create a temporary context for testing
        let mut test_ctx = InferenceContext::new();
        self.unify_shapes(s1, s2, &mut test_ctx).is_ok()
    }

    fn format_endpoint(&self, ep: &Endpoint) -> String {
        match ep {
            Endpoint::Ref(r) => self.format_port_ref(r),
            Endpoint::Call { name, .. } => format!("{}()", name),
            Endpoint::Tuple(refs) => format!(
                "({})",
                refs.iter()
                    .map(|r| r.node.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Endpoint::Match(_) => "match".to_string(),
        }
    }

    fn format_port_ref(&self, r: &PortRef) -> String {
        if r.port != "default" {
            format!("{}.{}", r.node, r.port)
        } else {
            r.node.clone()
        }
    }

    fn resolve_endpoint_shape(
        &self,
        endpoint: &Endpoint,
        ctx: &InferenceContext,
        _program: &Program,
    ) -> Result<Vec<Shape>, ShapeError> {
        match endpoint {
            Endpoint::Ref(port_ref) => {
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
                    let s =
                        self.resolve_endpoint_shape(&Endpoint::Ref(r.clone()), ctx, _program)?;
                    shapes.extend(s);
                }
                Ok(shapes)
            }
            Endpoint::Match(_) => Ok(vec![]), // TODO
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

        // Handle wildcards
        if self.has_wildcard(s1) || self.has_wildcard(s2) {
            return self.unify_shapes_with_wildcard(s1, s2, ctx);
        }

        // No wildcards or variadics - strict rank check
        if s1.dims.len() != s2.dims.len() {
            return Err(format!(
                "Rank mismatch: {} (rank {}) vs {} (rank {})",
                s1,
                s1.dims.len(),
                s2,
                s2.dims.len()
            ));
        }

        // Unify dimension by dimension
        for (i, (d1, d2)) in s1.dims.iter().zip(s2.dims.iter()).enumerate() {
            ctx.unify(d1, d2).map_err(|e| {
                format!(
                    "Dimension {} mismatch: {} - in shapes {} vs {}",
                    i, e, s1, s2
                )
            })?;
        }
        Ok(())
    }

    pub(crate) fn has_wildcard(&self, shape: &Shape) -> bool {
        shape.dims.iter().any(|d| matches!(d, Dim::Wildcard))
    }

    pub(crate) fn has_variadic(&self, shape: &Shape) -> bool {
        shape.dims.iter().any(|d| matches!(d, Dim::Variadic(_)))
    }

    fn unify_shapes_with_wildcard(
        &self,
        s1: &Shape,
        s2: &Shape,
        ctx: &mut InferenceContext,
    ) -> Result<(), String> {
        if s1.dims.len() != s2.dims.len() {
            return Err(format!(
                "Rank mismatch even with wildcards: {} (rank {}) vs {} (rank {})",
                s1,
                s1.dims.len(),
                s2,
                s2.dims.len()
            ));
        }

        for (i, (d1, d2)) in s1.dims.iter().zip(s2.dims.iter()).enumerate() {
            if matches!(d1, Dim::Wildcard) || matches!(d2, Dim::Wildcard) {
                continue;
            }

            ctx.unify(d1, d2).map_err(|e| {
                format!(
                    "Dimension {} mismatch: {} - in shapes {} vs {}",
                    i, e, s1, s2
                )
            })?;
        }

        Ok(())
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

                // 3. Check for obvious conflicts (e.g. constant vs constant) in the middle?
                // For MVP, we assume variadics are flexible enough to handle the rest.
                // TODO: More rigorous overlap check

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

        // Capture variadic dimensions
        if let Some(Dim::Variadic(name)) = pattern.dims.get(variadic_pos) {
            let captured_dims = concrete.dims[prefix_len..concrete_suffix_start].to_vec();

            // Should check if already bound to ensure consistency
            // Need to handle borrow checker carefully
            let existing_dims = ctx.resolved_variadics.get(name).cloned();

            if let Some(existing) = existing_dims {
                if existing.len() != captured_dims.len() {
                    return Err(format!(
                        "Variadic binding mismatch for {}: captured {} dims vs previous {}",
                        name,
                        captured_dims.len(),
                        existing.len()
                    ));
                }
                for (i, (d1, d2)) in existing.iter().zip(captured_dims.iter()).enumerate() {
                    ctx.unify(d1, d2)
                        .map_err(|e| format!("Variadic binding dim {} mismatch: {}", i, e))?;
                }
            } else {
                ctx.resolved_variadics.insert(name.clone(), captured_dims);
            }
        }

        Ok(())
    }

    /// Substitute captured variadics into a shape
    fn substitute_variadics(&self, shape: &Shape, ctx: &InferenceContext) -> Shape {
        let mut new_dims = Vec::new();

        for dim in &shape.dims {
            match dim {
                Dim::Variadic(name) => {
                    if let Some(captured) = ctx.resolved_variadics.get(name) {
                        new_dims.extend(captured.clone());
                    } else {
                        // Not captured? Keep as is or handle error?
                        // For now keep as is, it might be resolved later or valid to keep
                        new_dims.push(dim.clone());
                    }
                }
                _ => new_dims.push(dim.clone()),
            }
        }

        Shape { dims: new_dims }
    }

    /// Validate shape compatibility between source and destination with detailed context
    fn validate_connection_shapes(
        &self,
        source: &Shape,
        dest: &Shape,
        ctx: &mut InferenceContext,
        context: &str,
    ) -> Result<(), String> {
        // Try to unify shapes
        self.unify_shapes(source, dest, ctx)
            .map_err(|e| format!("{}\n  Context: {}", e, context))?;

        // Check for expression constraints that need solving
        for dim in &dest.dims {
            if let Dim::Expr(expr) = dim {
                self.validate_expr_constraint(expr, ctx, context)?;
            }
        }

        Ok(())
    }

    /// Validate that an expression constraint can be satisfied
    fn validate_expr_constraint(
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
                let left_resolvable = self.is_dim_resolvable(&expr.left, ctx);
                let right_resolvable = self.is_dim_resolvable(&expr.right, ctx);

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

    /// Check if a dimension can be resolved with current context
    pub(crate) fn is_dim_resolvable(&self, dim: &Dim, ctx: &InferenceContext) -> bool {
        match dim {
            Dim::Literal(_) => true,
            Dim::Named(name) => ctx.resolved_dims.contains_key(name),
            Dim::Wildcard => true,
            Dim::Variadic(_) => true,
            Dim::Expr(expr) => {
                self.is_dim_resolvable(&expr.left, ctx) && self.is_dim_resolvable(&expr.right, ctx)
            }
        }
    }

    // /// Validate shape compatibility for a specific operation type
    // fn validate_shape_compatibility(&self, op: &str, source: &Shape, dest: &Shape, ctx: &mut InferenceContext) -> Result<(), String> {
    //     match op {
    //         // Operations that require exact shape match
    //         "add" | "sub" | "mul" | "div" => {
    //             self.unify_shapes(source, dest, ctx)?;
    //         }
    //         // Operations that support broadcasting
    //         "broadcast_add" | "broadcast_mul" => {
    //             // Check if shapes are broadcastable
    //             // For MVP, just check they unify or one has wildcards
    //             self.unify_shapes(source, dest, ctx)?;
    //         }
    //         // Default: require unification
    //         _ => {
    //             self.unify_shapes(source, dest, ctx)?;
    //         }
    //     }
    //     Ok(())
    // }
}
