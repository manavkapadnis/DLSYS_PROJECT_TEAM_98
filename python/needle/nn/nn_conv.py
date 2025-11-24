"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        # Initialize weight: (kernel_size, kernel_size, in_channels, out_channels)
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        self.weight = Parameter(init.kaiming_uniform(
            fan_in, fan_out,
            shape=(kernel_size, kernel_size, in_channels, out_channels),
            device=device, dtype=dtype
        ))
        
        # Initialize bias
        if bias:
            bound = 1.0 / np.sqrt(fan_in)
            self.bias = Parameter(init.rand(out_channels, 
                                           low=-bound, high=bound,
                                           device=device, dtype=dtype))
        else:
            self.bias = None
        
        # Calculate padding to maintain spatial dimensions (for stride=1)
        self.padding = kernel_size // 2
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Input x is in NCHW format, need to convert to NHWC
        # x shape: (N, C, H, W)
        x_nhwc = ops.transpose(ops.transpose(x, (1, 2)), (2, 3))
        # x_nhwc shape: (N, H, W, C)
        
        # Apply convolution (conv op expects NHWC)
        out = ops.conv(x_nhwc, self.weight, stride=self.stride, padding=self.padding)
        # out shape: (N, H_out, W_out, C_out)
        
        # Add bias if present
        if self.bias is not None:
            # Reshape bias to (1, 1, 1, C_out) for broadcasting
            bias_reshaped = ops.reshape(self.bias, (1, 1, 1, self.out_channels))
            out = out + ops.broadcast_to(bias_reshaped, out.shape)
        
        # Convert back to NCHW format
        # (N, H, W, C) -> (N, C, H, W)
        out_nchw = ops.transpose(ops.transpose(out, (2, 3)), (1, 2))
        
        return out_nchw
        ### END YOUR SOLUTION