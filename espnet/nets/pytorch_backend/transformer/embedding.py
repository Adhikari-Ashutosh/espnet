#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positional Encoding Module."""

import logging
import math

import torch
from packaging.version import parse as V

from espnet2.asr.frontend.cnn import dim_1_layer_norm


from typing import Optional

def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Perform pre-hook in load_state_dict for backward compatibility.

    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.

    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position. Only for
        the class LegacyRelPositionalEncoding. We remove it in the current
        class RelPositionalEncoding.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.

    See Sec. 3.2  https://arxiv.org/abs/1809.08895

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnableFourierPosEnc(torch.nn.Module):
    """Learnable Fourier Features for Positional Encoding.

    See https://arxiv.org/pdf/2106.02795.pdf

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        gamma (float): init parameter for the positional kernel variance
            see https://arxiv.org/pdf/2106.02795.pdf.
        apply_scaling (bool): Whether to scale the input before adding the pos encoding.
        hidden_dim (int): if not None, we modulate the pos encodings with
            an MLP whose hidden layer has hidden_dim neurons.
    """

    def __init__(
        self,
        d_model,
        dropout_rate=0.0,
        max_len=5000,
        gamma=1.0,
        apply_scaling=False,
        hidden_dim=None,
    ):
        """Initialize class."""
        super(LearnableFourierPosEnc, self).__init__()

        self.d_model = d_model

        if apply_scaling:
            self.xscale = math.sqrt(self.d_model)
        else:
            self.xscale = 1.0

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.max_len = max_len

        self.gamma = gamma
        if self.gamma is None:
            self.gamma = self.d_model // 2

        assert (
            d_model % 2 == 0
        ), "d_model should be divisible by two in order to use this layer."
        self.w_r = torch.nn.Parameter(torch.empty(1, d_model // 2))
        self._reset()  # init the weights

        self.hidden_dim = hidden_dim
        if self.hidden_dim is not None:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d_model, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, d_model),
            )

    def _reset(self):
        self.w_r.data = torch.normal(
            0, (1 / math.sqrt(self.gamma)), (1, self.d_model // 2)
        )

    def extend_pe(self, x):
        """Reset the positional encodings."""
        position_v = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1).to(x)

        cosine = torch.cos(torch.matmul(position_v, self.w_r))
        sine = torch.sin(torch.matmul(position_v, self.w_r))
        pos_enc = torch.cat((cosine, sine), -1)
        pos_enc /= math.sqrt(self.d_model)

        if self.hidden_dim is None:
            return pos_enc.unsqueeze(0)
        else:
            return self.mlp(pos_enc.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        pe = self.extend_pe(x)
        x = x * self.xscale + pe
        return self.dropout(x)


class LegacyRelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(
            d_model=d_model,
            dropout_rate=dropout_rate,
            max_len=max_len,
            reverse=True,
        )

    def forward(self, x):
        """Compute positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[:, : x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)


class StreamPositionalEncoding(torch.nn.Module):
    """Streaming Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super(StreamPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.tmp = torch.tensor(0.0).expand(1, max_len)
        self.extend_pe(self.tmp.size(1), self.tmp.device, self.tmp.dtype)
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, length, device, dtype):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= length:
                if self.pe.dtype != dtype or self.pe.device != device:
                    self.pe = self.pe.to(dtype=dtype, device=device)
                return
        pe = torch.zeros(length, self.d_model)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, start_idx: int = 0):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x.size(1) + start_idx, x.device, x.dtype)
        x = x * self.xscale + self.pe[:, start_idx : start_idx + x.size(1)]
        return self.dropout(x)


class ConvolutionalPositionalEmbedding(torch.nn.Module):
    """Convolutional positional embedding.

       Used in wav2vec2/HuBERT SSL models.
       https://arxiv.org/abs/1904.11660

    Args:
        embed_dim (int): Feature dimension of the input Tensor.
        dropout (float): unused
        max_len (int): unused
        num_layers (int): number of conv layers
        kernel_size (int): The number of frames to be use.
        groups (int): The number of groups in feature dimensions.
        weight_norm (str): [new, legacy, none].
            How to init conv weights. Recommended setting is
            none if num_layers > 1.
    """

    def __init__(
        self,
        embed_dim: int,
        dropout: float,
        max_len: int = 5000,
        num_layers: int = 1,
        kernel_size: int = 128,
        groups: int = 16,
        weight_norm: str = "new",
        use_residual: bool = False,
    ):
        """Initialize Convoluational Positional Embedding."""
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.weight_norm = weight_norm

        convs = []
        for layer in range(num_layers):
            conv = torch.nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=groups,
            )
            if weight_norm != "none" and weight_norm is not None:
                std = math.sqrt((4 * (1.0)) / (kernel_size * embed_dim))
                torch.nn.init.normal_(conv.weight, mean=0, std=std)
                torch.nn.init.constant_(conv.bias, 0)
                # torch.nn.utils.weight_norm leads to weird behavior
                # with copy.deepcopy(). Usually isnt needed,
                # but its important for models that use EMA
                if weight_norm == "new":
                    if V(torch.__version__) >= V("2.2.0"):
                        conv = torch.nn.utils.parametrizations.weight_norm(
                            conv, name="weight", dim=2
                        )
                    else:
                        weight_norm = "legacy"
                        logging.warning(
                            "torch.nn.utils.parametrizations.weight_norm is only "
                            + "supported for pytorch versions >= 2.2.0. "
                            + "Defaulting to torch.nn.utils.weight_norm."
                        )
                if weight_norm == "legacy":
                    conv = torch.nn.utils.weight_norm(conv, name="weight", dim=2)
            convs.append(conv)
        self.convs = torch.nn.ModuleList(convs)
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0
        self.use_residual = use_residual

    def __prepare_scriptable__(self):
        """Prepare Scriptable method."""
        for hook in self.conv._forward_pre_hooks.values():
            # The hook we want to remove is an instance of WeightNorm class, so
            # normally we would do `if isinstance(...)` but this class is not accessible
            # because of shadowing, so we check the module name directly.
            # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
            if (
                hook.__module__ == "torch.nn.utils.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                logging.warning("Removing weight_norm from %s", self.__class__.__name__)
                torch.nn.utils.remove_weight_norm(self.conv)
        return self

    def forward(self, x):
        """Forward Method.

        Args:
            x (Tensor): shape ``[batch, frame, feature]``.

        Returns:
            Tensor: The resulting feature. Shape ``[batch, frame, feature]``.
        """
        if self.use_residual:
            residual = x

        x = x.transpose(-2, -1)
        for conv in self.convs:
            x = conv(x)

            # remove extra padding
            if self.num_remove > 0:
                x = x[..., : -self.num_remove]

            x = torch.nn.functional.gelu(x)

            # manually normalize if the conv is not parameterized
            # with weight norm
            if self.weight_norm is None or self.weight_norm == "none":
                x = dim_1_layer_norm(x)

        x = x.transpose(-2, -1)

        if self.use_residual:
            x = x + residual
        return x

class RotaryPositionalEmbedding(torch.nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Rotary Position Embedding (RoPE) logic ported and adapted from:
    torchtune (https://github.com/pytorch/torchtune), BSD 3-Clause License.

    Original author(s): Meta AI / PyTorch Contributors

    Adapted for ESPNet to enable rotary position encoding
    for Moonshine Architecture

    If incorporated outside this file, please preserve this notice and
    include the BSD 3-Clause License from torchtune per original.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
            self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Args:
                x (torch.Tensor): input tensor with shape
                    ``[b, s, n_h, h_d]``
                    If your input is ``[b, s, h_d]`` shape, reshape to ``[b, s, 1, h_d]``
                    with x = x.unsqueeze(2)
                input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                    of each token. During training, this is used to indicate the positions
                    of each token relative to its sample when packed, shape [b, s].
                    During inference, this indicates the position of the current token.
                    If none, assume the index of the token is its position id. Default is None.

            Returns:
                torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

            Notation used for tensor shapes:
                - b: batch size
                - s: sequence length
                - n_h: num heads
                - h_d: head dim
            """
            assert x.ndim == 4, (
                f"Input tensor shape {x.shape} is incompatible. "
                "Expected shape: (batch, seq_len, num_heads, head_dim). "
                "If your input is (batch, seq_len, head_dim), add a singleton num_heads axis via "
                "x = x.unsqueeze(2) before calling this module."
            )
            # input tensor has shape [b, s, n_h, h_d]

            # h_d should be even for RoPE
            h_d = x.shape[-1]
            assert h_d % 2 == 0, (
                f"head_dim (last dimension={h_d}) must be even for RoPE. "
                "Found an odd dimension—check your attention block wiring."
            )
            # head_dim should match the self dim.
            assert h_d == self.dim, (
                f"Input head_dim ({h_d}) does not match RotaryPositionalEmbedding.dim ({self.dim})"
            )
            
            seq_len = x.size(1)
            if seq_len > self.cache.size(0):
                # Dynamically rebuild the cache to allow for longer sequence len
                self.build_rope_cache(seq_len)

            # extract the values based on whether input_pos is set or not
            rope_cache = (
                self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
            )
            # reshape input; the last dimension is used for computing the output.
            # Cast to float to match the reference implementation
            # tensor has shape [b, s, n_h, h_d // 2, 2]
            xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

            # reshape the cache for broadcasting
            # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
            # otherwise has shape [1, s, 1, h_d // 2, 2]
            rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

            # tensor has shape [b, s, n_h, h_d // 2, 2]
            x_out = torch.stack(
                [
                    xshaped[..., 0] * rope_cache[..., 0]
                    - xshaped[..., 1] * rope_cache[..., 1],
                    xshaped[..., 1] * rope_cache[..., 0]
                    + xshaped[..., 0] * rope_cache[..., 1],
                ],
                -1,
            )

            # tensor has shape [b, s, n_h, h_d]
            x_out = x_out.flatten(3)
            return x_out.type_as(x)
