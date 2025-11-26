1. Fat Arrow passes var `in`, in this case, to each neuron in parallel
   * this is different from skinny arrow, which essentially joins outs to ins, forming a sequential pipeline

```neuroscript
# Now
neuron ActivationComparison(dim):
    in: [batch, dim]
    out relu: [batch, dim]
    out gelu: [batch, dim]
    out silu: [batch, dim]
    out tanh: [batch, dim]

    graph:
        in -> ReLU() -> relu
        in -> GELU() -> gelu
        in -> SiLU() -> silu
        in -> Tanh() -> tanh
        
# Fat Arrow
neuron ActivationComparison(dim):
    in: [batch, dim]
    out relu: [batch, dim]
    out gelu: [batch, dim]
    out silu: [batch, dim]
    out tanh: [batch, dim]

    graph:
        in => 
          ReLU() -> relu
          GELU() -> gelu
          SiLU() -> silu
          Tanh() -> tanh
```
