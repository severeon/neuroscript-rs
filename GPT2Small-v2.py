import torch
import torch.nn as nn
from neuroscript_runtime.primitives.activations import GELU
from neuroscript_runtime.primitives.attention import MultiHeadSelfAttention
from neuroscript_runtime.primitives.embeddings import Embedding
from neuroscript_runtime.primitives.embeddings import PositionalEncoding
from neuroscript_runtime.primitives.linear import Linear
from neuroscript_runtime.primitives.normalization import LayerNorm
from neuroscript_runtime.primitives.regularization import Dropout
from neuroscript_runtime.primitives.structural import Add
from neuroscript_runtime.primitives.structural import Fork

class FFN(nn.Module):
    def __init__(self, dim, expansion):
        super().__init__()
        self.dim = dim
        self.expansion = expansion
        self.g_e_l_u_1 = GELU()
        self.linear_0 = Linear(dim, expansion)
        self.linear_2 = Linear(expansion, dim)

    def forward(self, x):
        x0 = self.linear_0(x)
        # Linear() output shape: [*shape, out_dim]
        x1 = self.g_e_l_u_1(x0)
        # GELU() output shape: [*shape]
        x2 = self.linear_2(x1)
        # Linear() output shape: [*shape, out_dim]
        return x2

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.add_4 = Add()
        self.add_9 = Add()
        self.dropout_3 = Dropout(0.1)
        self.dropout_8 = Dropout(0.1)
        self.f_f_n_7 = FFN(d_model, d_ff)
        self.fork_0 = Fork()
        self.fork_5 = Fork()
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_6 = LayerNorm(d_model)
        self.multi_head_self_attention_2 = MultiHeadSelfAttention(d_model, num_heads)

    def forward(self, x):
        x0 = self.fork_0(x)
        # Fork() output shape: [batch, seq, d_model]
        x1, x2 = x0
        x3 = self.layer_norm_1(x2)
        # LayerNorm() output shape: [batch, seq, dim]
        x4 = self.multi_head_self_attention_2(x3)
        # MultiHeadSelfAttention() output shape: [*, seq_len, dim]
        x5 = self.dropout_3(x4)
        # Dropout() output shape: [*, seq_len, dim]
        # Expected shape: [*, seq_len, dim]
        x6 = self.add_4((x1, x5))
        # Add() output shape: [batch, seq, d_model]
        # Expected shape: [batch, seq, d_model]
        x7 = self.fork_5(x6)
        # Fork() output shape: [batch, seq, d_model]
        x8, x9 = x7
        x10 = self.layer_norm_6(x9)
        # LayerNorm() output shape: [batch, seq, dim]
        x11 = self.f_f_n_7(x10)
        # FFN() output shape: [batch, seq, dim]
        x12 = self.dropout_8(x11)
        # Dropout() output shape: [batch, seq, dim]
        # Expected shape: [batch, seq, dim]
        x13 = self.add_9((x8, x12))
        # Add() output shape: [batch, seq, d_model]
        return x13

class TransformerStack12(nn.Module):
    def __init__(self, dim, num_heads, d_ff):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.transformer_block_0 = TransformerBlock(dim, num_heads, d_ff)
        self.transformer_block_1 = TransformerBlock(dim, num_heads, d_ff)
        self.transformer_block_10 = TransformerBlock(dim, num_heads, d_ff)
        self.transformer_block_11 = TransformerBlock(dim, num_heads, d_ff)
        self.transformer_block_2 = TransformerBlock(dim, num_heads, d_ff)
        self.transformer_block_3 = TransformerBlock(dim, num_heads, d_ff)
        self.transformer_block_4 = TransformerBlock(dim, num_heads, d_ff)
        self.transformer_block_5 = TransformerBlock(dim, num_heads, d_ff)
        self.transformer_block_6 = TransformerBlock(dim, num_heads, d_ff)
        self.transformer_block_7 = TransformerBlock(dim, num_heads, d_ff)
        self.transformer_block_8 = TransformerBlock(dim, num_heads, d_ff)
        self.transformer_block_9 = TransformerBlock(dim, num_heads, d_ff)

    def forward(self, x):
        x0 = self.transformer_block_0(x)
        # TransformerBlock() output shape: [batch, seq, d_model]
        x1 = self.transformer_block_1(x0)
        # TransformerBlock() output shape: [batch, seq, d_model]
        x2 = self.transformer_block_2(x1)
        # TransformerBlock() output shape: [batch, seq, d_model]
        x3 = self.transformer_block_3(x2)
        # TransformerBlock() output shape: [batch, seq, d_model]
        x4 = self.transformer_block_4(x3)
        # TransformerBlock() output shape: [batch, seq, d_model]
        x5 = self.transformer_block_5(x4)
        # TransformerBlock() output shape: [batch, seq, d_model]
        x6 = self.transformer_block_6(x5)
        # TransformerBlock() output shape: [batch, seq, d_model]
        x7 = self.transformer_block_7(x6)
        # TransformerBlock() output shape: [batch, seq, d_model]
        x8 = self.transformer_block_8(x7)
        # TransformerBlock() output shape: [batch, seq, d_model]
        x9 = self.transformer_block_9(x8)
        # TransformerBlock() output shape: [batch, seq, d_model]
        x10 = self.transformer_block_10(x9)
        # TransformerBlock() output shape: [batch, seq, d_model]
        x11 = self.transformer_block_11(x10)
        # TransformerBlock() output shape: [batch, seq, d_model]
        return x11

class GPT2Small(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_0 = Embedding(vocab_size, 768)
        self.layer_norm_3 = LayerNorm(768)
        self.linear_4 = Linear(768, vocab_size)
        self.positional_encoding_1 = PositionalEncoding(768, max_len=1024)
        self.transformer_stack12_2 = TransformerStack12(768, 12, 3072)

    def forward(self, x):
        x0 = self.embedding_0(x)
        # Embedding() output shape: [*, seq_len, embedding_dim]
        x1 = self.positional_encoding_1(x0)
        # PositionalEncoding() output shape: [*, seq_len, dim]
        x2 = self.transformer_stack12_2(x1)
        # TransformerStack12() output shape: [batch, seq, dim]
        x3 = self.layer_norm_3(x2)
        # LayerNorm() output shape: [batch, seq, dim]
        x4 = self.linear_4(x3)
        # Linear() output shape: [batch, seq, out_dim]
        return x4

