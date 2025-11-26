# NeuroScript v3: Neuron Candidates

---

## Hierarchy Levels

```
Level 0: Atomic (primitives)
    ↓
Level 1: Composite (common patterns)
    ↓
Level 2: Architectural (well-known Neurons)
    ↓
Level 3: Model (complete models as Neurons)
    ↓
Level 4: Meta (ensembles, routers, control flow)
```

---

## Level 0: Atomic Neurons (Primitives)

**The building Neurons of building Neurons.**

### Core Operations
- [ ] `Linear` - dense/fully-connected layer
- [ ] `Bias` - additive bias
- [ ] `Scale` - multiplicative scaling
- [ ] `MatMul` - matrix multiplication
- [ ] `Einsum` - Einstein summation (generalized tensor operations)

### Activations
- [ ] `ReLU` - rectified linear unit
- [ ] `GELU` - Gaussian error linear unit
- [ ] `SiLU` / `Swish` - sigmoid linear unit
- [ ] `Tanh` - hyperbolic tangent
- [ ] `Sigmoid` - logistic function
- [ ] `Softmax` - normalized exponential
- [ ] `Mish` - self-regularized non-monotonic activation
- [ ] `PReLU` - parametric ReLU
- [ ] `ELU` - exponential linear unit

### Normalizations
- [ ] `LayerNorm` - layer normalization
- [ ] `BatchNorm` - batch normalization
- [ ] `RMSNorm` - root mean square normalization
- [ ] `GroupNorm` - group normalization
- [ ] `InstanceNorm` - instance normalization
- [ ] `WeightNorm` - weight normalization

### Regularization
- [ ] `Dropout` - random neuron dropout
- [ ] `DropPath` - stochastic depth
- [ ] `Dropblock` - structured dropout for CNNs
- [ ] `DropConnect` - connection dropout
- [ ] `SpecAugment` - frequency/time masking (audio)

### Convolutions
- [ ] `Conv1d` - 1D convolution (sequences)
- [ ] `Conv2d` - 2D convolution (images)
- [ ] `Conv3d` - 3D convolution (video/volumetric)
- [ ] `DepthwiseConv` - channel-wise convolution
- [ ] `SeparableConv` - depthwise + pointwise
- [ ] `TransposedConv` / `Deconv` - upsampling convolution
- [ ] `DilatedConv` - atrous convolution

### Pooling
- [ ] `MaxPool` - max pooling
- [ ] `AvgPool` - average pooling
- [ ] `AdaptiveAvgPool` - output-size-adaptive pooling
- [ ] `AdaptiveMaxPool` - output-size-adaptive max pooling
- [ ] `GlobalAvgPool` - spatial averaging
- [ ] `GlobalMaxPool` - spatial max

### Embeddings
- [ ] `Embedding` - discrete token → dense vector
- [ ] `PositionalEncoding` - sinusoidal position embeddings
- [ ] `LearnedPositionalEmbedding` - trainable positions
- [ ] `RotaryEmbedding` (RoPE) - rotary position embeddings
- [ ] `ALiBi` - attention with linear biases

### Utility
- [ ] `Reshape` - tensor reshaping
- [ ] `Transpose` - dimension permutation
- [ ] `Concatenate` - tensor concatenation
- [ ] `Split` - tensor splitting
- [ ] `Slice` - tensor slicing
- [ ] `Pad` - tensor padding
- [ ] `Crop` - tensor cropping
- [ ] `Cast` - dtype conversion
- [ ] `Clone` - tensor duplication
- [ ] `Identity` - pass-through (useful for routing)

---

## Level 1: Composite Neurons (Common Patterns)

**Built from atomic Neurons. Reusable patterns.**

### Attention Mechanisms
- [ ] `ScaledDotProductAttention` - core attention operation
- [ ] `MultiHeadAttention` - parallel attention heads
- [ ] `SelfAttention` - query=key=value
- [ ] `CrossAttention` - query ≠ key=value
- [ ] `FlashAttention` - memory-efficient attention
- [ ] `SparseAttention` - local/strided/block-sparse patterns
- [ ] `LinearAttention` - kernel-based O(n) attention
- [ ] `GroupedQueryAttention` (GQA) - shared key/value heads
- [ ] `MultiQueryAttention` (MQA) - single key/value head

### Feed-Forward Networks
- [ ] `FFN` / `MLP` - linear → activation → linear
- [ ] `GatedFFN` - gating mechanism (GLU-style)
- [ ] `SwiGLU` - SiLU gated FFN
- [ ] `GeGLU` - GELU gated FFN
- [ ] `Expert` - single expert block (for MoE)

### Residual Connections
- [ ] `Residual` - add(x, f(x))
- [ ] `PreNormResidual` - norm → f(x) → add
- [ ] `PostNormResidual` - f(x) → norm → add
- [ ] `HighwayConnection` - gated residual
- [ ] `DenseConnection` - concatenate all previous layers

### Gating Mechanisms
- [ ] `GLU` - gated linear unit
- [ ] `LSTM-Gate` - forget/input/output gates
- [ ] `GRU-Gate` - update/reset gates

### Positional Processing
- [ ] `RelativePositionBias` - T5-style learned biases
- [ ] `ConditionalPositionEncoding` - conditional position embeddings

---

## Level 2: Architectural Neurons (Well-Known Components)

**Famous architectures broken into composable Neurons.**

### Transformer Family
- [ ] `TransformerBlock` - attention + FFN + residuals
- [ ] `TransformerEncoderBlock` - self-attention block
- [ ] `TransformerDecoderBlock` - cross-attention block
- [ ] `TransformerStack` - N stacked transformer Neurons
- [ ] `PrefixLM` - prefix language model block
- [ ] `EncoderDecoder` - full transformer architecture

### Convolutional Architectures
- [ ] `ResNetBlock` - residual block (basic/bottleneck)
- [ ] `ResNeXtBlock` - grouped convolutions
- [ ] `DenseBlock` - densely connected block
- [ ] `InceptionBlock` - multi-scale convolutions
- [ ] `SEBlock` - squeeze-and-excitation
- [ ] `MBConvBlock` - mobile inverted bottleneck
- [ ] `FusedMBConv` - fused mobile conv (EfficientNet)
- [ ] `BottleneckBlock` - 1x1 → 3x3 → 1x1 conv

### Recurrent/State Space
- [ ] `LSTM` - long short-term memory
- [ ] `GRU` - gated recurrent unit
- [ ] `BiLSTM` / `BiGRU` - bidirectional variants
- [ ] `MambaBlock` - selective state space model
- [ ] `S4Block` - structured state space sequence model
- [ ] `H3Block` - hybrid H3 layer
- [ ] `RetNetBlock` - retentive network
- [ ] `RWKVBlock` - receptance weighted key value

### Vision-Specific
- [ ] `ConvNeXtBlock` - modernized ResNet block
- [ ] `SwinBlock` - shifted window attention
- [ ] `ViTBlock` - vision transformer block
- [ ] `PatchEmbedding` - image → patch embeddings
- [ ] `PatchMerging` - hierarchical patch reduction
- [ ] `FPN` - feature pyramid network
- [ ] `UNetBlock` - U-Net encoder/decoder block
- [ ] `ResUNetBlock` - residual U-Net block

### Generative
- [ ] `VAEEncoder` - variational autoencoder encoder
- [ ] `VAEDecoder` - VAE decoder
- [ ] `GANGenerator` - GAN generator block
- [ ] `GANDiscriminator` - GAN discriminator block
- [ ] `DiffusionUNet` - diffusion model U-Net
- [ ] `DiffusionTimestepEmbed` - timestep conditioning
- [ ] `NoiseScheduler` - diffusion noise scheduling

### Graph Neural Networks
- [ ] `GCNLayer` - graph convolutional network
- [ ] `GATLayer` - graph attention network
- [ ] `GraphSAGE` - graph sample and aggregate
- [ ] `GINLayer` - graph isomorphism network
- [ ] `MessagePassing` - generic message passing
- [ ] `EdgeConv` - edge convolution
- [ ] `GlobalPooling` - graph-level pooling

### Audio/Speech
- [ ] `MelSpectrogram` - mel-scale spectrogram
- [ ] `MFCC` - mel-frequency cepstral coefficients
- [ ] `WaveNetBlock` - dilated causal convolution
- [ ] `Conformer` - convolution-augmented transformer
- [ ] `WhisperEncoder` - speech encoder block
- [ ] `WhisperDecoder` - speech decoder block

### Normalization Variants
- [ ] `AdaptiveLayerNorm` - adaptive LN (e.g., for diffusion)
- [ ] `ConditionalLayerNorm` - class-conditional normalization
- [ ] `SpectralNorm` - spectral normalization (GANs)

---

## Level 3: Model Neurons (Complete Models as Composable Neurons)

**Entire models packaged as single Neurons. Embed GPT-2 in your architecture. Go wild.**

### Language Models
- [ ] `GPT` - autoregressive language model (any size)
- [ ] `BERT` - bidirectional encoder
- [ ] `T5` - encoder-decoder model
- [ ] `LLaMA` - LLaMA architecture (any variant)
- [ ] `Mistral` - Mistral model
- [ ] `Mamba` - full Mamba model
- [ ] `RWKV` - full RWKV model
- [ ] `Pythia` - Pythia model family
- [ ] `Falcon` - Falcon model
- [ ] `Phi` - Phi small language models

### Vision Models
- [ ] `ResNet` - full ResNet (18/34/50/101/152)
- [ ] `EfficientNet` - EfficientNet (B0-B7)
- [ ] `ViT` - vision transformer
- [ ] `Swin` - Swin transformer
- [ ] `ConvNeXt` - ConvNeXt architecture
- [ ] `CLIP-VisualEncoder` - CLIP image encoder
- [ ] `DINOv2` - self-supervised vision model

### Multimodal
- [ ] `CLIP` - contrastive language-image model
- [ ] `BLIP` - bootstrapped language-image model
- [ ] `Flamingo` - visual language model
- [ ] `LLaVA` - large language and vision assistant
- [ ] `CoCa` - contrastive captioner

### Generative Models
- [ ] `StableDiffusion-UNet` - diffusion model
- [ ] `VAE-KL` - Kullback-Leibler VAE
- [ ] `VQVAE` - vector quantized VAE
- [ ] `StyleGAN` - StyleGAN generator
- [ ] `ControlNet` - conditional diffusion control

### Audio Models
- [ ] `Whisper` - speech recognition
- [ ] `Wav2Vec2` - speech representation
- [ ] `HuBERT` - hidden unit BERT
- [ ] `MusicGen` - music generation
- [ ] `AudioLDM` - audio latent diffusion

### Specialized
- [ ] `AlphaFold-Evoformer` - protein folding
- [ ] `ProteinMPNN` - protein design
- [ ] `ESMFold` - protein language model
- [ ] `MolFormer` - molecular transformer

---

## Level 4: Meta Neurons (Control Flow, Routing, Composition)

**Neurons that orchestrate other Neurons.**

### Routing
- [ ] `Switch` - conditional routing (if-else)
- [ ] `Router` - learned routing (multi-path)
- [ ] `MixtureOfExperts` (MoE) - sparse expert routing
- [ ] `TopKRouter` - route to top-k experts
- [ ] `LoadBalancedRouter` - expert load balancing
- [ ] `ConditionalCompute` - dynamic depth

### Composition
- [ ] `Sequential` - linear chain of Neurons
- [ ] `Parallel` - multiple Neurons in parallel
- [ ] `Residual` - skip connection wrapper
- [ ] `Ensemble` - average/vote multiple models
- [ ] `LoRA` - low-rank adaptation wrapper
- [ ] `Adapter` - adapter layer wrapper
- [ ] `PrefixTuning` - prefix tuning wrapper

### Recursive/Dynamic
- [ ] `RecurrentBlock` - apply block N times
- [ ] `DynamicDepth` - variable-depth networks
- [ ] `NeuralODE` - neural ordinary differential equations
- [ ] `UniversalTransformer` - adaptive computation time

### Multi-Scale
- [ ] `Pyramid` - multi-scale processing
- [ ] `Cascade` - coarse-to-fine processing
- [ ] `HierarchicalMerge` - merge across scales

### Utilities
- [ ] `Checkpoint` - gradient checkpointing
- [ ] `Quantize` - quantization (INT8/INT4)
- [ ] `Prune` - structured/unstructured pruning
- [ ] `DistillationHead` - knowledge distillation
- [ ] `EMA` - exponential moving average

---

## Level 5: External Model Neurons (Someone Else's Model)

**Import arbitrary models as Neurons.**

### Import Formats
- [ ] `HuggingFaceModel` - any model from HF Hub
- [ ] `TorchHubModel` - PyTorch Hub models
- [ ] `ONNXModel` - ONNX models
- [ ] `TensorFlowModel` - TF SavedModel
- [ ] `JAXModel` - JAX/Flax models
- [ ] `SafetensorsModel` - safetensors format

## Domain-Specific Block Collections

### Computer Vision
**Primitives:** Conv2d, MaxPool, BatchNorm, ReLU
**Composite:** ResNetBlock, SEBlock, ViTBlock
**Models:** ResNet50, EfficientNetB0, ViT-B/16
**Meta:** FPN, CascadeRCNN

### Natural Language Processing
**Primitives:** Embedding, Linear, LayerNorm, Dropout
**Composite:** TransformerBlock, MambaBlock
**Models:** GPT-2, BERT-base, LLaMA-7B
**Meta:** MoE, LoRA, PrefixTuning

### Audio/Speech
**Primitives:** Conv1d, MelSpectrogram, LSTM
**Composite:** ConformerBlock, WaveNetBlock
**Models:** Whisper-small, Wav2Vec2
**Meta:** CTC, Attention

### Graphs
**Primitives:** MessagePassing, EdgeConv
**Composite:** GATLayer, GCNLayer
**Models:** GraphSAGE, GIN
**Meta:** GlobalPooling

### Reinforcement Learning
- [ ] `PolicyHead` - policy network head
- [ ] `ValueHead` - value network head
- [ ] `ActorCritic` - actor-critic architecture
- [ ] `DQN` - deep Q-network
- [ ] `PPONetwork` - PPO architecture

---

## Adapter/Compatibility Neurons

**Make incompatible Neurons work together.**

### Shape Adapters
- [ ] `DimensionAdapter` - change feature dimensions
- [ ] `SequenceLengthAdapter` - resample sequence length
- [ ] `BatchAdapter` - handle batch size changes
- [ ] `ChannelAdapter` - adapt channel count
- [ ] `SpatialAdapter` - resize spatial dimensions

### Type Adapters
- [ ] `DTypeAdapter` - convert float32 ↔ float16 ↔ bfloat16
- [ ] `QuantizationAdapter` - float ↔ int8/int4
- [ ] `DeviceAdapter` - CPU ↔ GPU ↔ TPU

### Format Adapters
- [ ] `ImageToPatches` - image → patch sequence
- [ ] `PatchesToImage` - patch sequence → image
- [ ] `TokensToEmbedding` - discrete → continuous
- [ ] `EmbeddingToLogits` - continuous → discrete
- [ ] `AudioToSpectrogram` - waveform → frequency
- [ ] `SpectrogramToAudio` - frequency → waveform

### Connector Neurons
- [ ] `Projection` - arbitrary shape transformation
- [ ] `Upsampler` - increase resolution
- [ ] `Downsampler` - decrease resolution
- [ ] `Interpolate` - smooth resampling
- [ ] `BridgeBlock` - generic A → B converter

---

## Specialized/Exotic Neurons

### Memory/Context
- [ ] `MemoryBank` - external memory (NTM, DNC)
- [ ] `KNNMemory` - k-nearest neighbor lookup
- [ ] `VectorDatabase` - embedding search
- [ ] `ContextWindow` - sliding window buffer

### Symbolic/Neuro-Symbolic
- [ ] `LogicGate` - differentiable logic
- [ ] `ProgramSynthesis` - neural program synthesis
- [ ] `TreeLSTM` - tree-structured LSTM
- [ ] `GraphGrammar` - graph generation rules

### Equivariance/Invariance
- [ ] `RotationEquivariant` - rotation equivariance (e.g., E(n) layers)
- [ ] `TranslationInvariant` - translation invariance
- [ ] `PermutationInvariant` - permutation invariance (Set Transformer)
- [ ] `ScaleEquivariant` - scale equivariance

### Energy-Based
- [ ] `EnergyFunction` - energy-based model
- [ ] `Hopfield` - modern Hopfield network
- [ ] `Boltzmann` - restricted Boltzmann machine

### Emerging Research
- [ ] `KolmogorovArnold` (KAN) - KAN layers
- [ ] `LiquidNeuron` - liquid time-constant networks
- [ ] `HyperNetwork` - networks that generate networks
- [ ] `MetaLearner` - MAML, Reptile, etc.
- [ ] `NeuralTangentKernel` - NTK-based Neurons
