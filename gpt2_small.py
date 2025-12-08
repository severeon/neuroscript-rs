import torch
import torch.nn as nn
from neuroscript_runtime.primitives.activations import GELU
from neuroscript_runtime.primitives.attention import MultiHeadSelfAttention
from neuroscript_runtime.primitives.embeddings import Embedding
from neuroscript_runtime.primitives.embeddings import PositionalEncoding
from neuroscript_runtime.primitives.linear import Linear
from neuroscript_runtime.primitives.normalization import LayerNorm

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
        # Linear() output shape: [batch, seq, out_dim]
        x1 = self.g_e_l_u_1(x0)
        # GELU() output shape: [batch, seq, dim]
        x2 = self.linear_2(x1)
        # Linear() output shape: [batch, seq, out_dim]
        return x2

class GPTTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, d_ff):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.f_f_n_2 = FFN(dim, d_ff)
        self.layer_norm_0 = LayerNorm(dim)
        self.multi_head_self_attention_1 = MultiHeadSelfAttention(dim, num_heads)

    def forward(self, x):
        x0 = self.layer_norm_0(x)
        # LayerNorm() output shape: [batch, seq, dim]
        x1 = self.multi_head_self_attention_1(x0)
        # MultiHeadSelfAttention() output shape: [batch, seq, dim]
        x2 = self.f_f_n_2(x1)
        # FFN() output shape: [*, seq_len, dim]
        return x2

class GPT2Small(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_0 = Embedding(vocab_size, 768)
        self.g_p_t_transformer_block_2 = GPTTransformerBlock(768, 12, 3072)
        self.layer_norm_3 = LayerNorm(768)
        self.linear_4 = Linear(768, vocab_size)
        self.positional_encoding_1 = PositionalEncoding(768, max_len=1024)

    def forward(self, x):
        x0 = self.embedding_0(x)
        # Embedding() output shape: [batch, seq, d_model]
        # Expected shape: [batch, seq, d_model]
        x1 = self.positional_encoding_1(x0)
        # PositionalEncoding() output shape: [batch, seq, d_model]
        # Expected shape: [batch, seq, d_model]
        x2 = self.g_p_t_transformer_block_2(x1)
        # GPTTransformerBlock() output shape: [batch, seq, dim]
        # Expected shape: [batch, seq, dim]
        x3 = self.layer_norm_3(x2)
        # LayerNorm() output shape: [batch, seq, dim]
        # Expected shape: [batch, seq, dim]
        x4 = self.linear_4(x3)
        # Linear() output shape: [batch, seq, out_dim]
        return x4

