import numpy as np
from einops import rearrange

import mlx.core as mx
import mlx.nn as nn

from local_attention import LocalAttention


def exists(val):
    return val is not None


class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout, bias):
        f"""
        Initializes the Multi-Layer Perceptron (MLP) layer.

        Args:
            config (GPTConfig): An instance of the configuration class
                specifying the hyperparameters for the MLP layer.

        Attributes:
            c_fc (nn.Linear): Linear layer for fully connected transformations.
            gelu (nn.GELU): GELU activation function.
            c_proj (nn.Linear): Linear layer for output projection.
            dropout (nn.Dropout): Dropout layer for regularization.

        Notes:
            - Ensure that the `config` parameter is an instance of `MLPConfig`.
            - The configuration class should contain the necessary hyperparameters for
              configuring the MLP layer.
            - The `c_fc` layer performs a fully connected transformation.
            - The `gelu` layer applies the GELU activation function.
            - The `c_proj` layer handles the output projection.
            - The `dropout` layer applies dropout regularization.
        """
        super().__init__()

        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        window_size: int = 512,
        causal: bool = True,
        look_backward: int = 1,
        look_forward: int = 0,
        dropout: float = 0.1,
        shared_qk: bool = True,
        exact_window_size: bool = False,
        heads: int = None,
        dim_head: int = None,
        ff_mult=2,
    ):
        super().__init__()

        self.layers = []
        self.ffn_layers = []
        for _ in range(depth):
            self.layers.append(
                LocalAttention(
                    dim=dim,
                    window_size=window_size,
                    causal=causal,
                    look_backward=look_backward,
                    look_forward=look_forward,
                    dropout=dropout,
                    shared_qk=shared_qk,
                ),
            )
            
            self.ffn_layers.append(
                FeedForward(n_embd=dim, dropout=dropout, bias=False),
            )
            

    def __call__(self, x):
        for attn, ffn in zip(self.layers, self.ffn_layers):
            x = ffn(x) + x
            x = attn(x, x, x) + x
            x = ffn(x) + x
            attn = attn(x, x, x)
            attn_mult = mx.matmul(attn, x)
            mlp = ffn(x)
            out = attn_mult + mlp

        return out


class SimplifiedTransformers(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_tokens,
        dim_head=64,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)

        self.transformer = Transformer(
            dim=dim, 
            depth=depth, 
            heads=heads, 
            dim_head=dim_head, 
            ff_mult=ff_mult)

        self.to_logits = nn.Sequential(nn.RMSNorm(dim), nn.Linear(dim, num_tokens))

    def __call__(self, x):
        x = self.emb(x)
        x = self.transformer(x)
        return self.to_logits(x)
