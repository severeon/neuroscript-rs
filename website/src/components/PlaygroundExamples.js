// Curated examples showcasing NeuroScript language features
export const EXAMPLES = [
  // ============================================================================
  // BASICS
  // ============================================================================
  {
    id: 'mlp',
    title: 'Simple MLP',
    category: 'Basics',
    description: 'Feed-forward network with dimension parameters and expressions',
    code: `# A simple multi-layer perceptron with expansion and contraction
neuron MLP(dim):
    in: [*, dim]
    out: [*, dim]
    graph:
        in ->
            Linear(dim, dim * 4)
            GELU()
            Linear(dim * 4, dim)
            out`,
    targetNeuron: 'MLP',
    features: ['Dimension variables', 'Pipeline syntax', 'Dimension expressions']
  },

  {
    id: 'linear-projection',
    title: 'Linear Projection',
    category: 'Basics',
    description: 'Simple projection layer changing dimensions',
    code: `# Project from one dimension to another
neuron Projection(in_dim, out_dim):
    in: [batch, in_dim]
    out: [batch, out_dim]
    graph:
        in -> Linear(in_dim, out_dim) -> out`,
    targetNeuron: 'Projection',
    features: ['Shape signatures', 'Dimension propagation']
  },

  {
    id: 'normalization',
    title: 'Normalization Layer',
    category: 'Basics',
    description: 'Layer normalization with dimension preservation',
    code: `# Normalize activations along the feature dimension
neuron NormalizedLayer(dim):
    in: [*, dim]
    out: [*, dim]
    graph:
        in ->
            LayerNorm(dim)
            Linear(dim, dim)
            out`,
    targetNeuron: 'NormalizedLayer',
    features: ['Normalization', 'Shape preservation']
  },

  // ============================================================================
  // PATTERNS
  // ============================================================================
  {
    id: 'residual',
    title: 'Residual Block',
    category: 'Patterns',
    description: 'Skip connections with Fork and Add primitives',
    code: `# Classic residual connection pattern
neuron ResidualBlock(dim):
    in: [*, dim]
    out: [*, dim]
    graph:
        # Fork creates two copies of the input
        in -> Fork() -> (main, skip)

        # Process the main path
        main ->
            LayerNorm(dim)
            Linear(dim, dim * 4)
            GELU()
            Linear(dim * 4, dim)
            processed

        # Add skip connection back
        (processed, skip) -> Add() -> out`,
    targetNeuron: 'ResidualBlock',
    features: ['Fork primitive', 'Tuple unpacking', 'Residual connections']
  },

  {
    id: 'parallel-paths',
    title: 'Parallel Processing',
    category: 'Patterns',
    description: 'Multiple parallel paths with concatenation',
    code: `# Process input through three parallel paths
neuron ParallelPaths(dim):
    in: [*, dim]
    out: [*, dim * 3]
    graph:
        # Split into three paths
        in -> Fork3() -> (path1, path2, path3)

        # Process each independently
        path1 -> Linear(dim, dim) -> p1
        path2 -> Linear(dim, dim) -> p2
        path3 -> Linear(dim, dim) -> p3

        # Combine results
        (p1, p2, p3) -> Concat(-1) -> out`,
    targetNeuron: 'ParallelPaths',
    features: ['Multi-way fork', 'Parallel paths', 'Concatenation']
  },

  {
    id: 'pre-activation',
    title: 'Pre-Activation Residual',
    category: 'Patterns',
    description: 'Residual with normalization before transformation',
    code: `# Pre-activation residual block (normalize first)
neuron PreActResidual(dim):
    in: [batch, dim]
    out: [batch, dim]
    graph:
        in -> Fork() -> (main, skip)

        main ->
            LayerNorm(dim)
            Linear(dim, dim * 4)
            GELU()
            Linear(dim * 4, dim)
            processed

        (processed, skip) -> Add() -> out`,
    targetNeuron: 'PreActResidual',
    features: ['Pre-activation', 'Residual pattern', 'Named dimensions']
  },

  // ============================================================================
  // ADVANCED
  // ============================================================================
  {
    id: 'match-basic',
    title: 'Shape Pattern Matching',
    category: 'Advanced',
    description: 'Match expressions with dimension capture',
    code: `# Route based on input dimension size
neuron AdaptiveProjection:
    in: [*, dim]
    out: [*, 512]
    graph:
        in -> match:
            [*, d] where d > 512: Linear(d, 512) -> out
            [*, d] where d < 512: Linear(d, 512) -> out
            [*, d]: Identity() -> out`,
    targetNeuron: 'AdaptiveProjection',
    features: ['Match expressions', 'Dimension capture', 'Guard conditions']
  },

  {
    id: 'match-routing',
    title: 'Shape-Based Routing',
    category: 'Advanced',
    description: 'Different processing paths based on tensor dimensions',
    code: `# Choose processing based on feature dimension
neuron DimensionRouter(out_dim):
    in: [batch, in_dim]
    out: [batch, out_dim]
    graph:
        in -> match:
            [batch, d] where d > 1024:
                Linear(d, 512) ->
                Linear(512, out_dim)
            [batch, d] where d > 256:
                Linear(d, out_dim)
            [batch, d]:
                Linear(d, out_dim)
        -> out`,
    targetNeuron: 'DimensionRouter',
    features: ['Shape matching', 'Dimension binding', 'Multi-line syntax']
  },

  {
    id: 'variadic-shapes',
    title: 'Variadic Wildcards',
    category: 'Advanced',
    description: 'Match variable-length shape prefixes',
    code: `# Process tensors with arbitrary leading dimensions
neuron FlexibleNorm(dim):
    in: [*shape, dim]
    out: [*shape, dim]
    graph:
        in ->
            LayerNorm(dim)
            Linear(dim, dim)
            out`,
    targetNeuron: 'FlexibleNorm',
    features: ['Variadic wildcards', 'Shape prefixes', 'Rank-agnostic']
  },

  // ============================================================================
  // REAL WORLD
  // ============================================================================
  {
    id: 'attention-head',
    title: 'Attention Head',
    category: 'Real World',
    description: 'Single attention head with Q, K, V projections',
    code: `# Single-head scaled dot-product attention
neuron AttentionHead(dim, head_dim):
    in: [batch, seq, dim]
    out: [batch, seq, head_dim]
    graph:
        # Project to Q, K, V
        in -> Fork3() -> (q_in, k_in, v_in)
        q_in -> Linear(dim, head_dim) -> q
        k_in -> Linear(dim, head_dim) -> k
        v_in -> Linear(dim, head_dim) -> v

        # Compute attention
        (q, k, v) -> ScaledDotProductAttention(head_dim) -> out`,
    targetNeuron: 'AttentionHead',
    features: ['Multi-input operations', 'Attention mechanism', 'Fork-join pattern']
  },

  {
    id: 'transformer-ffn',
    title: 'Transformer FFN',
    category: 'Real World',
    description: 'Feed-forward network from transformer',
    code: `# Transformer feed-forward network with residual
neuron TransformerFFN(dim):
    in: [batch, seq, dim]
    out: [batch, seq, dim]
    graph:
        in -> Fork() -> (main, skip)

        main ->
            LayerNorm(dim)
            Linear(dim, dim * 4)
            GELU()
            Linear(dim * 4, dim)
            Dropout(0.1)
            processed

        (processed, skip) -> Add() -> out`,
    targetNeuron: 'TransformerFFN',
    features: ['Transformer components', 'Dropout', 'Standard architecture']
  },

  {
    id: 'simple-cnn',
    title: 'CNN Block',
    category: 'Real World',
    description: 'Convolutional block with batch norm',
    code: `# Basic CNN block with conv + norm + activation
neuron ConvBlock(channels):
    in: [batch, channels, h, w]
    out: [batch, channels, h, w]
    graph:
        in ->
            Conv2d(channels, channels, kernel_size=3, padding=1)
            BatchNorm(channels)
            ReLU()
            out`,
    targetNeuron: 'ConvBlock',
    features: ['Convolutional layers', 'Batch normalization', '2D operations']
  }
];

// Get examples by category
export function getExamplesByCategory() {
  const categories = {};
  EXAMPLES.forEach(ex => {
    if (!categories[ex.category]) {
      categories[ex.category] = [];
    }
    categories[ex.category].push(ex);
  });
  return categories;
}

// Get example by ID
export function getExampleById(id) {
  return EXAMPLES.find(ex => ex.id === id);
}
