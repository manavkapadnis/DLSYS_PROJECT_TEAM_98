import math
from .init_basic import *
from typing import Any

def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, shape=None, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    if shape is not None:
        # Use shape directly for multidimensional arrays
        a = gain * math.sqrt(6 / (fan_in + fan_out))
        return rand(*shape, low=-a, high=a, **kwargs)
    else:
        # Original behavior for 2D matrices
        a = gain * math.sqrt(6 / (fan_in + fan_out))
        return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, shape=None, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    if shape is not None:
        # Use shape directly for multidimensional arrays
        std = gain * math.sqrt(2 / (fan_in + fan_out))
        return randn(*shape, mean=0.0, std=std, **kwargs)
    else:
        # Original behavior for 2D matrices
        std = gain * math.sqrt(2 / (fan_in + fan_out))
        return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", shape=None, **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)  # Recommended gain for ReLU
    bound = gain * math.sqrt(3 / fan_in)
    
    if shape is not None:
        # Use shape directly for multidimensional arrays (e.g., convolution kernels)
        return rand(*shape, low=-bound, high=bound, **kwargs)
    else:
        # Original behavior for 2D matrices
        return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", shape=None, **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)  # Recommended gain for ReLU
    std = gain / math.sqrt(fan_in)
    
    if shape is not None:
        # Use shape directly for multidimensional arrays (e.g., convolution kernels)
        return randn(*shape, mean=0.0, std=std, **kwargs)
    else:
        # Original behavior for 2D matrices
        return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION
