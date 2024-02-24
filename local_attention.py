# Adapted from Zeta's LocalAttention module
# Source: https://github.com/kyegomez/zeta/blob/master/zeta/nn/attention/local_attention.py


import numpy as np
from einops import pack, rearrange, repeat, unpack

import mlx.core as mx
import mlx.nn as nn

from utils import (
    default, 
    exists, 
    l2norm_mx, 
    look_around_mx, 
    max_neg_values
)


TOKEN_SELF_ATTN_VALUE = -5e4


def rotate_half_mx(x):
    """Rotate the numpy array by half."""
    # First, reshape x to separate the last dimension into two parts (r=2)
    # Assuming the original last dimension is evenly divisible by 2
    new_shape = x.shape[:-1] + (2, -1)  # Add an extra dimension for the split
    x_reshaped = x.reshape(new_shape)
    
    # Now, split the array along the new dimension to get x1 and x2
    # Equivalent to unbinding along the new axis in PyTorch
    x1, x2 = x_reshaped[..., 0, :], x_reshaped[..., 1, :]
    
    # Rotate and concatenate -x2 and x1 along the last dimension
    result = mx.concatenate((-x2, x1), axis=-1)
    
    return result


def apply_rotary_pos_emb(q, k, freqs, scale=1):
    """
    Apply rotary positional embeddings to the query and key tensors.

    Args:
        q (torch.Tensor): The query tensor.
        k (torch.Tensor): The key tensor.
        freqs (torch.Tensor): The frequencies.
        scale (torch.Tensor): The scale.

    """
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]

    inv_scale = scale**-1

    if scale.ndim == 2:
        scale = scale[-q_len:, :]

    q = (q * q_freqs.cos() * scale) + (rotate_half_mx(q) * q_freqs.sin() * scale)
    k = (k * freqs.cos() * inv_scale) + (
        rotate_half_mx(k) * freqs.sin() * inv_scale
    )
    return q, k


class SinusoidalEmbeddings(nn.Module):
    """
    Sinusoidal embeddings.

    Args:
        dim (int): The dimension of the embeddings.
        scale_base (int): The scale base for the positional embeddings.
        use_xpos (bool): Whether to use xpos or not.

    Attributes:
        inv_freq (torch.Tensor): The inverse frequencies.
        scale (torch.Tensor): The scale.

    Example:
        >>> module = SinusoidalEmbeddings(10)
        >>> x = torch.randn(10, 10)
        >>> y = module(x)
        >>> y.shape
        torch.Size([10, 10, 10])

    """

    def __init__(self, dim, scale_base=None, use_xpos=False):
        super().__init__()
        base = np.arange(0, dim, 2, dtype=np.float32)
        # Compute the encoding using np.power for element-wise exponentiation
        inv_freq = 1.0 / np.power(10000, base / dim)
        self.inv_freq = inv_freq #mx.array(inv_freq)

        # xpos related
        self.use_xpos = use_xpos
        self.scale_base = scale_base

        assert not (
            use_xpos and not exists(scale_base)
        ), "scale base must be defined if using xpos"

        scale = (mx.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale = scale
        

    def __call__(self, x):
        """forward"""
        seq_len = x.shape[-2]

        t = mx.arange(seq_len, dtype=mx.float32)
        # freqs = t[:, np.newaxis] * self.inv_freq[np.newaxis, :]
        freqs = np.einsum("i , j -> i j", np.array(t), self.inv_freq)
        freqs = mx.array(freqs)
        freqs = mx.concatenate((freqs, freqs), axis=-1)

        if not self.use_xpos:
            return freqs, mx.ones(1)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.cat((scale, scale), dim=-1)

        return freqs, scale


class LocalAttention(nn.Module):
    """

    The LocalAttention module provides a mechanism to perform local attention operations.
    Unlike global attention where every token can attend to every other token,
    in local attention each token can only attend to a subset of tokens within a defined window. This reduces the computational cost and captures the local structure in sequences like text or time-series data.

    Args:
        window_size: (int) The size of the attention window.
        causal: (bool, optional) If set to True, ensures causal attention. Default: False.
        look_backward: (int, optional) How many positions to look backward from the current position. Default: 1.
        look_forward: (int, optional) How many positions to look forward from the current position. Default: None which implies 0 if causal is True.
        dropout: (float, optional) Dropout rate for attention weights. Default: 0..
        shared_qk: (bool, optional) If set to True, the query and key are the same. Useful for certain types of attention mechanisms. Default: False.
        rel_pos_emb_config: (Optional) Deprecated. Configuration for the relative positional embeddings.
        dim: (int, optional) Dimension of embeddings. Only needed if rel_pos_emb_config is not provided.
        autopad: (bool, optional) If set to True, sequence will be automatically padded to be divisible by the window size. Default: False.
        exact_windowsize: (bool, optional) Ensures exact window size for non-causal attention. Default: False.
        scale: (Optional) Scaling factor for the queries.
        use_rotary_pos_emb: (bool, optional) If set to True, rotary positional embeddings will be used. Default: True.
        use_xpos: (bool, optional) If set to True, allows for extrapolation of window sizes. Requires use_rotary_pos_emb to be True. Default: False.
        xpos_scale_base: (Optional) Base scaling factor for extrapolated window sizes.

    Usage:
    >>> model = LocalAttention(64, 1, 1, 0.1)
    >>> x = torch.randn(1, 768)
    >>> model(x).shape

    """

    def __init__(
        self,
        window_size,
        causal=False,
        look_backward=1,
        look_forward=None,
        dropout=0.0,
        shared_qk=False,
        rel_pos_emb_config=None,
        dim=None,
        autopad=False,
        exact_windowsize=False,
        scale=None,
        use_rotary_pos_emb=True,
        use_xpos=False,
        xpos_scale_base=None,
    ):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (
            causal and look_forward > 0
        ), "you cannot look forward if causal"

        self.scale = scale

        self.window_size = window_size
        self.autopad = autopad
        self.exact_windowsize = exact_windowsize

        self.causal = causal

        self.look_backward = look_backward
        self.look_forward = look_forward

        self.dropout = nn.Dropout(dropout)

        self.shared_qk = shared_qk

        # relative positions

        self.rel_pos = None
        self.use_xpos = use_xpos

        if use_rotary_pos_emb and (
            exists(rel_pos_emb_config) or exists(dim)
        ):  # backwards compatible with old `rel_pos_emb_config` deprecated argument
            if exists(rel_pos_emb_config):
                dim = rel_pos_emb_config[0]

            self.rel_pos = SinusoidalEmbeddings(
                dim,
                use_xpos=use_xpos,
                scale_base=default(xpos_scale_base, window_size // 2),
            )

    """

    Forward Method
        Parameters
        q: (Tensor) The query tensor.

        k: (Tensor) The key tensor.

        v: (Tensor) The value tensor.

        mask: (Optional[Tensor]) A mask tensor for the keys. Can also be passed as input_mask.

        input_mask: (Optional[Tensor]) Another way to pass the mask tensor for keys.

        attn_bias: (Optional[Tensor]) Additional biases to add to the attention scores.

        window_size: (Optional[int]) If provided, this window size will override the default window size defined during initialization.

        Returns
        out: (Tensor) The output tensor after the attention operation.
    """

    def __call__(
        self,
        q,
        k,
        v,
        mask=None,
        input_mask=None,
        attn_bias=None,
        window_size=None,
    ):
        mask = default(mask, input_mask)

        assert not (
            exists(window_size) and not self.use_xpos
        ), "cannot perform window size extrapolation if xpos is not turned on"

        (
            shape,
            autopad,
            pad_value,
            window_size,
            causal,
            look_backward,
            look_forward,
            shared_qk,
        ) = (
            q.shape,
            self.autopad,
            -1,
            default(window_size, self.window_size),
            self.causal,
            self.look_backward,
            self.look_forward,
            self.shared_qk,
        )

        # # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
        (q, packed_shape), (k, _), (v, _) = map(
            lambda t: pack([t], "* n d"), (np.array(q), np.array(k), np.array(v))
        )
        q = mx.array(q)
        k = mx.array(k)
        v = mx.array(v)

        b, n, dim_head, dtype = *q.shape, q.dtype

        scale = default(self.scale, dim_head**-0.5)

        assert (n % window_size) == 0, (
            f"sequence length {n} must be divisible by window size"
            f" {window_size} for local attention"
        )

        windows = n // window_size

        if shared_qk:
            k = l2norm_mx(k)

        seq = mx.arange(n)
        # Reshape the sequence
        # First, ensure seq is reshapeable to the desired shape: (windows, window_size)
        # This requires that n is exactly divisible by windows * window_size
        if seq.size % (windows * window_size) != 0:
            raise ValueError("The total size of seq is not divisible by windows * window_size")

        b_t = rearrange(np.array(seq), "(w n) -> 1 w n", w=windows, n=window_size)
        b_t = mx.array(b_t)
        # bucketing

        # Applying the function to q, k, v
        bq, bk, bv = map(
            lambda t: rearrange(t, "b (w n) d -> b w n d", w=windows), (np.array(q), np.array(k), np.array(v))
        )
        bq, bk, bv = mx.array(bq), mx.array(bk), mx.array(bv)
        
        bq = bq * scale

        look_around_kwargs = dict(
            backward=look_backward, forward=look_forward, pad_value=pad_value
        )

        bk = look_around_mx(bk, **look_around_kwargs)
        bv = look_around_mx(bv, **look_around_kwargs)

        # rotary embeddings

        if exists(self.rel_pos):
            pos_emb, xpos_scale = self.rel_pos(bk)
            bq, bk = apply_rotary_pos_emb(bq, bk, pos_emb, scale=xpos_scale)

        # calculate positions for masking
        bq_t = b_t
        bq_k = look_around_mx(b_t, **look_around_kwargs)

        bq_t = rearrange(np.array(bq_t), "... i -> ... i 1")
        bq_k = rearrange(np.array(bq_k), "... j -> ... 1 j")
        pad_mask = bq_k == pad_value

        sim = np.einsum("b h i e, b h j e -> b h i j", np.array(bq), np.array(bk))

        # convert to mx.array
        sim = mx.array(sim)
        bq_t, bq_k, bq, bk = mx.array(bq_t), mx.array(bq_k), mx.array(bq), mx.array(bq)
        pad_mask = mx.array(pad_mask)

        if exists(attn_bias):
            heads = attn_bias.shape[0]
            assert (b % heads) == 0

            attn_bias = repeat(np.array(attn_bias), "h i j -> (b h) 1 i j", b=b // heads)
            attn_bias = mx.array(attn_bias)
            sim = sim + attn_bias

        mask_value = max_neg_values(sim)

        if shared_qk:
            self_mask = bq_t == bq_k
            sim = mx.where(self_mask, TOKEN_SELF_ATTN_VALUE, sim)
            del self_mask

        if causal:
            causal_mask = bq_t < bq_k

            if self.exact_windowsize:
                max_causal_window_size = self.window_size * self.look_backward
                causal_mask = causal_mask | (
                    bq_t > (bq_k + max_causal_window_size)
                )

            masked_fill = -1e9
            sim = mx.where(causal_mask, mask_value, sim)
            del causal_mask

        # masking out for exact window size for non-causal
        # as well as masking out for padding value

        if not causal and self.exact_windowsize:
            max_backward_window_size = self.window_size * self.look_backward
            max_forward_window_size = self.window_size * self.look_forward
            window_mask = (
                ((bq_k - max_forward_window_size) > bq_t)
                | (bq_t > (bq_k + max_backward_window_size))
                | pad_mask
            )
            sim = sim.masked_fill(window_mask, mask_value)
        else:
            sim = mx.where(pad_mask, mask_value, sim)

        # take care of key padding mask passed in
        if exists(mask):
            raise NotImplementedError("This part of the function is not yet implemented")

        # attention
        attn = mx.softmax(sim, axis=-1)
        attn = self.dropout(attn)

        out = np.einsum("b h i j, b h j e -> b h i e", np.array(attn), np.array(bv))
        out = rearrange(out, "b w n d -> b (w n) d")

        if autopad:
            raise NotImplementedError("This part of the function is not yet implemented")

        out, *_ = unpack(np.array(out), packed_shape, "* n d")
        out = mx.array(out)
        return out