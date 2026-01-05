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
            _ => {
                // TODO: Handle more complex unifications (e.g. named vs expr)
                Ok(())
            }
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
                match expr.op {
                    BinOp::Add => {
                        if target >= *right_val as usize {
                            self.resolve_dim(left_name.clone(), target - *right_val as usize)
                        } else {
                            Err(format!(
                                "Illegal negative dimension: {} + {} = {}",
                                left_name, right_val, target
                            ))
                        }
                    }
                    BinOp::Sub => self.resolve_dim(left_name.clone(), target + *right_val as usize),
                    BinOp::Mul => {
                        #[allow(clippy::manual_is_multiple_of)]
                        if target % (*right_val as usize) != 0 {
                            return Err(format!(
                                "Cannot solve {} * {} = {}: {} is not divisible by {}",
                                left_name, right_val, target, target, right_val
                            ));
                        }
                        self.resolve_dim(left_name.clone(), target / (*right_val as usize))
                    }
                    BinOp::Div => {
                        self.resolve_dim(left_name.clone(), target * (*right_val as usize))
                    }
                    _ => Ok(()), // TODO
                }
            }
            (Dim::Literal(left_val), Dim::Named(right_name)) => {
                // Solve: left op right = target
                match expr.op {
                    BinOp::Add => {
                        if target >= *left_val as usize {
                            self.resolve_dim(right_name.clone(), target - *left_val as usize)
                        } else {
                            Err(format!(
                                "Illegal negative dimension: {} + {} = {}",
                                left_val, right_name, target
                            ))
                        }
                    }
                    BinOp::Sub => {
                        // left - right = target  =>  right = left - target
                        if *left_val as usize >= target {
                            self.resolve_dim(right_name.clone(), *left_val as usize - target)
                        } else {
                            Err(format!(
                                "Illegal negative dimension: {} - {} = {}",
                                left_val, right_name, target
                            ))
                        }
                    }
                    BinOp::Mul => {
                        #[allow(clippy::manual_is_multiple_of)]
                        if target % (*left_val as usize) != 0 {
                            return Err(format!(
                                "Cannot solve {} * {} = {}: {} is not divisible by {}",
                                left_val, right_name, target, target, left_val
                            ));
                        }
                        self.resolve_dim(right_name.clone(), target / (*left_val as usize))
                    }
                    BinOp::Div => {
                        // left / right = target  =>  right = left / target
                        #[allow(clippy::manual_is_multiple_of)]
                        if target != 0 && (*left_val as usize) % target == 0 {
                            self.resolve_dim(right_name.clone(), (*left_val as usize) / target)
                        } else {
                            Ok(())
                        }
                    }
                    _ => Ok(()), // TODO
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
                    _ => None,
                };

                if let Some(val) = left_val {
                    self.solve_expr_for_unknown(left_expr, val)
                } else {
                    Ok(()) // or error
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
                    _ => None,
                };

                if let Some(val) = right_val {
                    self.solve_expr_for_unknown(right_expr, val)
                } else {
                    Ok(())
                }
            }
            _ => Ok(()), // Too complex to solve for now
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
            BinOp::Add => Some(left + right),
            BinOp::Sub => left.checked_sub(right),
            BinOp::Mul => Some(left * right),
            BinOp::Div => left.checked_div(right),
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
                            .insert(port_ref.node.clone(), source_shapes.to_vec());
                    } else {
                        // Default port
                        ctx.node_outputs
                            .insert(port_ref.node.clone(), source_shapes.to_vec());
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
                                format_port_ref(port_ref)
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
        let mut output_shapes = Vec::new();

        for arm in arms {
            // Each arm gets its own isolated context for capture
            let mut arm_ctx = ctx.clone();

            // Source shapes must be unifiable with match pattern
            // Combine all source shapes into a single pattern match?
            // Match expressions often take a single input.
            if source_shapes.is_empty() {
                return Err(ShapeError::Mismatch {
                    expected: arm.pattern.clone(),
                    got: Shape::new(vec![]),
                    context: "Match source produces no output".to_string(),
                });
            }

            // Unify with ARM pattern
            self.unify_pattern_with_shape(&arm.pattern, &source_shapes[0], &mut arm_ctx)
                .map_err(|msg| ShapeError::ConstraintViolation {
                    message: format!("Pattern mismatch in match arm: {}", msg),
                    context: format!(
                        "Pattern: {}\nSource shape: {}",
                        arm.pattern, source_shapes[0]
                    ),
                })?;

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

    pub(crate) fn has_wildcard(&self, s: &Shape) -> bool {
        s.dims.iter().any(|d| matches!(d, Dim::Wildcard))
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

        // Capture variadic segment
        if let Dim::Variadic(name) = &pattern.dims[variadic_pos] {
            let variadic_dims = &concrete.dims[prefix_len..concrete_suffix_start];
            let segment = variadic_dims.to_vec();

            if let Some(existing) = ctx.resolved_variadics.get(name) {
                // If already bound, they must be equal (or at least same rank and compatible)
                if existing.len() != segment.len() {
                    return Err(format!("Variadic binding mismatch for {}: existing variant rank {}, new variant rank {}", name, existing.len(), segment.len()));
                }
                // TODO: Deep unification of variadic segments
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
    }
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
    }
}

/// Check if a dimension can be resolved with current context
pub(crate) fn is_dim_resolvable(dim: &Dim, ctx: &InferenceContext) -> bool {
    match dim {
        Dim::Literal(_) => true,
        Dim::Named(name) => ctx.resolved_dims.contains_key(name),
        Dim::Wildcard => true,
        Dim::Variadic(_) => true,
        Dim::Expr(expr) => {
            is_dim_resolvable(&expr.left, ctx) && is_dim_resolvable(&expr.right, ctx)
        }
    }
}
