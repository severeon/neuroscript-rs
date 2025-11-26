# NeuroScript Runtime

PyTorch implementations of primitive neurons for compiled NeuroScript models.

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Architecture

### Primitives
Level 0 building blocks that map to NeuroScript neurons with `impl:` references:

- **Linear** - Dense/fully-connected layer with shape tracking
- **GELU** - Gaussian Error Linear Unit activation
- **Dropout** - Regularization with training/eval modes
- **LayerNorm** - Layer normalization
- **Embedding** - Token → vector embedding

### Shape Awareness

All primitives are shape-aware and validate tensor dimensions according to NeuroScript's shape algebra rules. This ensures that shape mismatches are caught early and reported clearly.

## Usage

```python
import torch
from neuroscript_runtime import Linear, GELU, LayerNorm

# Primitives work like standard PyTorch modules
layer = Linear(in_features=512, out_features=256)
x = torch.randn(32, 10, 512)  # [batch, seq, dim]
out = layer(x)  # [32, 10, 256]

# Activations
gelu = GELU()
activated = gelu(out)

# Normalization
norm = LayerNorm(256)
normalized = norm(activated)
```

## Testing

```bash
pytest tests/
```

## License

MIT
