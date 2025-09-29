import contextlib
import logging
from typing import Optional, Tuple, Union

import humanfriendly
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend


class MoonshineFrontend(AbsFrontend):
    """Audio Preprocessor as implemented in Moonshine.
    
    This frontend applies a series of 1D convolutions with activations and 
    normalization to preprocess audio signals for ASR models.
    
    The preprocessing stack consists of:
    1. Conv1d (kernel_size=127, stride=64) + Tanh + GroupNorm
    2. Conv1d (kernel_size=7, stride=3) + GELU  
    3. Conv1d (kernel_size=3, stride=2) + GELU
    
    Args:
        dim (int): Dimension of output feature space (number of channels).
        
    Note:
        Input shape should be [batch, length, 1] (channels last).
        Output shape will be [batch, dim, conv_output_length].
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # First convolution block: large kernel for initial feature extraction
        conv1 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=dim,
            kernel_size=127,
            stride=64,
            bias=False
        )
        tanh_activation = torch.nn.Tanh()
        group_norm = torch.nn.GroupNorm(num_groups=1, num_channels=dim)

        # Second convolution block: smaller kernel for refinement  
        conv2 = torch.nn.Conv1d(
            in_channels=dim,
            out_channels=2 * dim,
            kernel_size=7,
            stride=3
        )
        gelu1 = torch.nn.GELU()
        
        # Third convolution block: final feature compression
        conv3 = torch.nn.Conv1d(
            in_channels=2 * dim,
            out_channels=dim,
            kernel_size=3,
            stride=2
        )
        gelu2 = torch.nn.GELU()
        
        self.preproc = torch.nn.Sequential(
            conv1,
            tanh_activation,
            group_norm,
            conv2,
            gelu1,
            conv3,
            gelu2,               
        )

    def output_size(self) -> int:
        """Return the number of output channels (feature dimension).
        
        Returns:
            int: Output feature dimension (same as input dim parameter).
            
        Note:
            This returns the channel dimension.
            The sequence length after convolutions depends on input length.
        """
        return self.dim
    
    @typechecked
    def forward(
        self, 
        input: torch.Tensor, 
        input_lengths: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass through the preprocessing network.
        
        Args:
            input (torch.Tensor): Input audio tensor of shape [batch, length, 1].
            input_lengths (torch.Tensor, optional): Length of each sequence 
                in the batch. Currently unused but kept for API compatibility.
        
        Returns:
            torch.Tensor: Processed features of shape [batch, dim, conv_length].
            
        Note:
            Input is expected in channels-last format [B, L, 1] but internally
            converted to channels-first [B, 1, L] for Conv1d processing.
        """
        # Convert from channels-last [B, L, 1] to channels-first [B, 1, L]
        assert input.ndim == 3, f"Expected shape of input to be [B, L, 1], Try reshaping or averaging channels.\n Expected ndims = 3, got ndims = {input.ndim}."
        assert input.shape[-1] == 1, f"Expected channels of input to be 1, Mono Audio of shape [B, L, 1]. Try reshaping if not working."
        input = input.permute(0, 2, 1)
        return self.preproc(input)
