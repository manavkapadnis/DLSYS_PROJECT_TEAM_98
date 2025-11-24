from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        # Use numpy for broadcasting
        import numpy as np
        Z_np = Z.numpy()
        
        # Find maximum along last axis for numerical stability
        max_Z = np.max(Z_np, axis=-1, keepdims=True)
        shifted = Z_np - max_Z
        
        # Compute LogSumExp along last axis
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
        logsumexp_result = max_Z + np.log(sum_exp)
        
        # LogSoftmax = z_i - LogSumExp(z)
        result = Z_np - logsumexp_result
        
        # Convert back to NDArray
        return array_api.array(result, device=Z.device)

    def gradient(self, out_grad: Tensor, node: Tensor):
        Z = node.inputs[0]
        
        # Compute softmax for gradient using Tensor ops
        logsumexp_val = logsumexp(Z, axes=(-1,))
        
        # Reshape to broadcast
        new_shape = tuple(list(Z.shape[:-1]) + [1])
        logsumexp_reshaped = reshape(logsumexp_val, new_shape)
        logsumexp_broadcast = broadcast_to(logsumexp_reshaped, Z.shape)
        
        # Softmax = exp(z_i - LogSumExp(z))
        softmax = exp(Z - logsumexp_broadcast)
        
        # Gradient: out_grad - softmax * sum(out_grad, axis=-1)
        sum_grad = summation(out_grad, axes=(-1,))
        sum_grad_reshaped = reshape(sum_grad, new_shape)
        sum_grad_broadcast = broadcast_to(sum_grad_reshaped, Z.shape)
        
        return out_grad - softmax * sum_grad_broadcast


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        # Convert to numpy for broadcasting operations
        import numpy as np
        Z_np = Z.numpy()
        
        # Find maximum along axes for numerical stability
        max_z = np.max(Z_np, axis=self.axes, keepdims=True)
        
        # Subtract max, exp, sum, and log
        shifted = Z_np - max_z
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=self.axes)
        result = np.log(sum_exp) + np.max(Z_np, axis=self.axes, keepdims=False)
        
        # Convert back to NDArray
        return array_api.array(result, device=Z.device)

    def gradient(self, out_grad: Tensor, node: Tensor):
        Z = node.inputs[0]
        input_shape = Z.shape
        
        # Determine the shape for broadcasting
        if self.axes is None:
            new_shape = tuple([1] * len(input_shape))
        else:
            new_shape = list(input_shape)
            for axis in self.axes:
                new_shape[axis] = 1
            new_shape = tuple(new_shape)
        
        # Reshape and broadcast using Tensor ops
        grad_reshaped = reshape(out_grad, new_shape)
        grad_broadcast = broadcast_to(grad_reshaped, input_shape)
        
        node_reshaped = reshape(node, new_shape)
        node_broadcast = broadcast_to(node_reshaped, input_shape)
        
        # Gradient: out_grad * exp(Z - logsumexp(Z))
        return grad_broadcast * exp(Z - node_broadcast)


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)