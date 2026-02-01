# -----------------------------------------------------------------------------
# Moonshine: Speech Recognition for Live Transcription and Voice Commands
# Nat Jeffries, Evan King, Manjunath Kudlur, Guy Nicholson, James Wang, Pete Warden
# arXiv:2410.15608 (2024) https://arxiv.org/abs/2410.15608
#
# This implementation is based on the original Moonshine work,
# graciously released under the MIT License by the authors.
# -----------------------------------------------------------------------------

import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.specaug.specaug import SpecAug
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.embedding import RotaryPositionalEmbedding
from someplace.somewhere import Moonshine_custom_MHA # Why not use the standard MHA with RoPE applied to Q and K?
from espnet.nets.pytorch_backend.transformer.attention import (  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
    RoPEMultiHeadedAttention
)
@typechecked
class MoonShineEncoderLayer(torch.nn.Module):
    """
        Single Encoder Layer module. 

        Args:
            dim: int. input dim
            inner_dim: int. MHAttention output dims
            enc_ff_mult: int, optional. Projection multiplier
            enc_ff_swiglu: int,optional. Flag to set and use Swiglu FFN layer, Uses Gelu by default
            dtype: str, optional. The dtype to use for model computations and
                weights. Defaults to None.

        #TODO: Remove during clean up
        Layer of Encoder Recipe:

            Ingredients:
                1. Layer Norm (norm1)
                2. MHAWithRope (What the helll OH MY GAWD NO WAYIYIAY) (Comes in main encoder)
                3. Layer Norm (norm2)
                4. FFSwiglu or FFGelu (ff)

            Directions:
                1. x <- Input from frontEnd
                2. _x = x <- Keep a copy of x
                3. Norm X using LayerNorm... 
                4. x <- apply RoPE enhanced Attention with 1 head
                5. x = x + _x <- Residual connection
                6. _x = x <- new save
                7. Norm x with Layer norm2
                8. x = ff(x)
                9. x = x+ _x <- Add residual
        
        """
    def __init__(
        self,
        dim,
        inner_dim,
        n_head,
        enc_ff_mult=4,
        enc_ff_swiglu=False,
    ):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim, bias=False) 
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html -> Check Norm Shape, bias
        # from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm 
        # A bit Meta but not using ^ because by default affine with bias is enabled... (Maybe a toggle Hyperparam could be introduced)
        # But default Norm in Moonshine is affine with no center (beta) but scale (gamma)... refer given doc
        self.norm2 = torch.nn.LayerNorm(dim, bias=False) 
        self.attn = MultiHeadedAttention(dim,n_head,inner_dim) # Using standard MHA with RoPE applied to Q and K inside Encoder
        # Really like how E-Branchformer arranges this...
        self.enc_ff_swiglu = enc_ff_swiglu
        if enc_ff_swiglu:
            self.ff_in = torch.nn.Linear(dim, dim*enc_ff_mult*2) 
            self.ff_act = torch.nn.SiLU()
            self.ff_out = torch.nn.Linear(dim*enc_ff_mult, dim)
        else:
            self.ff_in = torch.nn.Linear(dim, dim*enc_ff_mult)
            self.ff_act = torch.nn.GELU()
            self.ff_out = torch.nn.Linear(dim*enc_ff_mult, dim)
    def forward(
            self,
            x_input,
            rot_pos_emb # My forward in the MHA needs this too
    ):
        """
        Args:
            x_input: Input Tensor of Shape [B, L, dim]
            rot_pos_emb: Input Tensor of Shape [B, L, dim]
        """
        _x = x_input
        x_input = self.norm1(x_input)
        x_input = self.attn(x_input,rot_pos_emb) # RoPE applied inside MHA
        x_input = x_input +_x
        _x = x_input 
        x_input = self.norm2(x_input)
        # Swiglu Or GELU branching
        if self.enc_ff_swiglu:
            x_input_2 = self.ff_in(x_input)
            x_input, gate = torch.split(x_input_2, x_input_2.size(-1)//2, dim=-1)
            x_input = x_input*self.ff_act(gate)
            x_input = self.ff_out(x_input)
        else:
            x_input = self.ff_in(x_input)
            x_input = self.ff_act(x_input)
            x_input = self.ff_out(x_input)
        x_input = _x + x_input
        return x_input

@typechecked
class MoonShineEncoder(AbsEncoder):
    """Linear encoder module. 

    Args:
        dim: int. input dim
        inner_dim: int. MHAttention output dims
        n_head: int. Number of heads in MHA
        enc_n_layers: int. Number of Encoder layers stacked
        enc_ff_mult: int, optional. Projection multiplier
        enc_ff_swiglu: int,optional. Flag to set and use Swiglu FFN layer, Uses Gelu by default
        dtype: str, optional. The dtype to use for model computations and
            weights. Defaults to None.
    #TODO: Remove during clean up
    We need to add the following things:
        1. Implement batch by supporting padding masks <Need to figure where this goes;mostly passed to forward>
        2. 
    """
    def __init__(
        self,
        dim,
        inner_dim,
        n_head,
        enc_n_layers,
        enc_ff_mult=4,
        enc_ff_swiglu=False,
    ):
        raise NotImplementedError