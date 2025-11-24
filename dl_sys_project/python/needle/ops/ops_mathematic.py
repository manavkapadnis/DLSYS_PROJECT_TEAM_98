"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        # If shapes don't match, broadcast manually
        if a.shape != b.shape:
            # Compute broadcast shape
            max_ndim = max(len(a.shape), len(b.shape))
            a_shape = (1,) * (max_ndim - len(a.shape)) + a.shape
            b_shape = (1,) * (max_ndim - len(b.shape)) + b.shape
            
            broadcast_shape = []
            for i in range(max_ndim):
                if a_shape[i] == b_shape[i]:
                    broadcast_shape.append(a_shape[i])
                elif a_shape[i] == 1:
                    broadcast_shape.append(b_shape[i])
                elif b_shape[i] == 1:
                    broadcast_shape.append(a_shape[i])
                else:
                    raise ValueError(f"Cannot broadcast shapes {a.shape} and {b.shape}")
            
            broadcast_shape = tuple(broadcast_shape)
            
            # Broadcast arrays
            if a.shape != broadcast_shape:
                a = a.broadcast_to(broadcast_shape)
            if b.shape != broadcast_shape:
                b = b.broadcast_to(broadcast_shape)
        
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        
        # Compute gradients
        grad_lhs = out_grad * rhs
        grad_rhs = out_grad * lhs
        
        # Handle broadcasting in gradients
        def reduce_to_shape(grad, original_shape):
            # If shapes match, no reduction needed
            if grad.shape == original_shape:
                return grad
            
            # Compute axes to sum over
            axes = []
            
            # Handle dimension mismatch
            ndim_diff = len(grad.shape) - len(original_shape)
            for i in range(ndim_diff):
                axes.append(i)
            
            # Handle size-1 dimensions
            for i, (grad_dim, orig_dim) in enumerate(zip(
                grad.shape[ndim_diff:], original_shape
            )):
                if orig_dim == 1 and grad_dim > 1:
                    axes.append(i + ndim_diff)
            
            # Sum and reshape
            if axes:
                grad = grad.sum(tuple(axes))
            
            return grad.reshape(original_shape) if grad.shape != original_shape else grad
        
        grad_lhs = reduce_to_shape(grad_lhs, lhs.shape)
        grad_rhs = reduce_to_shape(grad_rhs, rhs.shape)
        
        return grad_lhs, grad_rhs




def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** b
        # raise NotImplementedError()
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        # For z = a^b:
        # dz/da = b * a^(b-1)
        # dz/db = a^b * ln(a)
        
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a ** b) * log(a) 
        
        return grad_a, grad_b
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return out_grad * self.scalar * (node.inputs[0] ** (self.scalar - 1))       

        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / (rhs ** 2)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/ self.scalar
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            # Default: transpose last two axes only
            perm = list(range(a.ndim))
            if a.ndim >= 2:
                perm[-2], perm[-1] = perm[-1], perm[-2]
            return a.permute(tuple(perm))  # Use .permute() method
        else:
            # Transpose the specified axes
            perm = list(range(a.ndim))
            if len(self.axes) == 2:
                i, j = self.axes
                perm[i], perm[j] = perm[j], perm[i]
            return a.permute(tuple(perm))  # Use .permute() method

    def gradient(self, out_grad, node):
        if self.axes is None:
            return transpose(out_grad)
        else:
            # If forward swapped axes (i, j), backward swaps them back
            return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Compact the array if it's not compact
        if not a.is_compact():
            a = a.compact()
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION



def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        out_shape = out_grad.shape
        if input_shape == out_shape:
            return out_grad
        axis = []
        for i in range(len(out_shape) - len(input_shape)):
            axis.append(i)
        for i in range(len(input_shape)):
            if input_shape[i] == 1 and out_shape[i + len(out_shape) - len(input_shape)] != 1:
                axis.append(i + len(out_shape) - len(input_shape))
        if len(axis) == 0:
            return out_grad
        return summation(out_grad, tuple(axis)).reshape(input_shape)
    
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # Handle None (sum all), single axis, or tuple of axes
        if self.axes is None:
            return array_api.sum(a, axis=None)
        elif isinstance(self.axes, (tuple, list)):
            # Sum over multiple axes one at a time
            result = a
            # Sort axes in reverse order to maintain correct indices
            for axis in sorted(self.axes, reverse=True):
                result = array_api.sum(result, axis=axis)
            return result
        else:
            return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        
        if self.axes is None:
            # Summed over all axes
            return broadcast_to(reshape(out_grad, (1,) * len(input_shape)), input_shape)
        
        # Reshape to add back summed dimensions
        axes = [self.axes] if isinstance(self.axes, int) else list(self.axes)
        shape = list(input_shape)
        for axis in sorted(axes):
            shape[axis] = 1
        
        return broadcast_to(reshape(out_grad, shape), input_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b  # Use the NDArray __matmul__ operator directly

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        # grad_lhs = out_grad @ rhs.T
        # grad_rhs = lhs.T @ out_grad
        return (out_grad @ transpose(rhs), transpose(lhs) @ out_grad)


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)

class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Get the input data
        input_data = node.inputs[0].realize_cached_data()
        
        # Create mask where input > 0
        mask = (input_data > 0) * 1.0
        
        # Create Tensor from the float mask with same device as out_grad
        mask_tensor = Tensor(mask, device=out_grad.device, requires_grad=False)
        
        return out_grad * mask_tensor
        ### END YOUR SOLUTION




def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        # derivative of tanh is 1 - tanh^2
        return out_grad - out_grad * (node ** 2)



def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        self.axis = axis

    def compute(self, args: TensorTuple) -> NDArray:
        ### BEGIN YOUR SOLUTION
        arrays = [arg for arg in args]
        n = len(arrays)
        shape = list(arrays[0].shape)
        new_shape = shape[:self.axis] + [n] + shape[self.axis:]
        
        out = array_api.empty(new_shape, dtype=arrays[0].dtype, device=arrays[0].device)
        
        for i, arr in enumerate(arrays):
            slices = [slice(None)] * len(new_shape)
            slices[self.axis] = i
            out[tuple(slices)] = arr
        
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Split out_grad along the axis it was stacked on
        # This returns a tuple of tensors, each with the stacked dimension still present
        split_grads = split(out_grad, self.axis)
        
        # Each tensor in split_grads has shape like (1, ...) along the split axis
        # We need to remove that singleton dimension by reshaping
        result = []
        for grad in split_grads:
            # Remove the singleton dimension at self.axis
            # For example, if grad has shape (1, 6, 5) and axis=0, we want (6, 5)
            new_shape = list(grad.shape)
            new_shape.pop(self.axis)  # Remove dimension at axis
            result.append(grad.reshape(tuple(new_shape)))
        
        return make_tuple(*result)
        ### END YOUR SOLUTION




def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        # Get the size of the dimension we're splitting along
        n = A.shape[self.axis]
        
        # Create a list to hold the split arrays
        splits = []
        
        # Calculate the new shape (remove the split axis)
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        new_shape = tuple(new_shape)
        
        # Extract each slice along the axis
        for i in range(n):
            # Create slicing tuple
            slices = [slice(None)] * len(A.shape)
            slices[self.axis] = i
            
            # Extract the slice and reshape to remove singleton dimension
            split_array = A[tuple(slices)]
            splits.append(split_array)
        
        return tuple(splits)

    def gradient(self, out_grad, node):
        # Split's gradient is Stack - stack the gradients back along the same axis
        return stack(list(out_grad), self.axis)



def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Gradient of flip is flip (flipping twice gives back original)
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation
    
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Calculate new shape after dilation
        new_shape = list(a.shape)
        for axis in self.axes:
            # Check if axis is valid for the array shape
            if axis >= len(a.shape):
                # Skip invalid axes or raise an error
                continue
            new_shape[axis] = a.shape[axis] * (self.dilation + 1)
        
        # Create output array filled with zeros
        out = a.device.full(tuple(new_shape), 0.0, dtype="float32")
        
        # Create slicing tuple to place original values
        slices = []
        for i in range(len(a.shape)):
            if i in self.axes and i < len(a.shape):
                # For dilated axes, place values at every (dilation+1) position
                slices.append(slice(0, new_shape[i], self.dilation + 1))
            else:
                # For non-dilated axes, use all positions
                slices.append(slice(None))
        
        # Place the original array values
        out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Gradient of dilate is undilate
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Create slicing tuple to extract every (dilation+1)-th element
        slices = []
        for i in range(len(a.shape)):
            if i in self.axes:
                # For dilated axes, extract every (dilation+1) position
                slices.append(slice(0, a.shape[i], self.dilation + 1))
            else:
                # For non-dilated axes, use all positions
                slices.append(slice(None))
        
        return a[tuple(slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Gradient of undilate is dilate
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
      ### BEGIN YOUR SOLUTION
      # A: input (N, H, W, C_in) in NHWC format
      # B: weight (K, K, C_in, C_out)
      
      # Apply padding if needed
      if self.padding > 0:
          pad_axes = ((0, 0), (self.padding, self.padding), 
                    (self.padding, self.padding), (0, 0))
          A = A.pad(pad_axes)
      
      # Ensure A is compact after padding (padding might make it non-compact)
      if not A.is_compact():
          A = A.compact()
      
      N, H, W, C_in = A.shape
      K, _, _, C_out = B.shape
      
      # Output dimensions
      H_out = (H - K) // self.stride + 1
      W_out = (W - K) // self.stride + 1
      
      # Use im2col approach
      Ns, Hs, Ws, Cs = A.strides
      
      # Create strided view for patches
      A_col = A.as_strided(
          shape=(N, H_out, W_out, K, K, C_in),
          strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)
      ).compact()
      
      # Reshape for matrix multiplication
      A_col = A_col.reshape((N * H_out * W_out, K * K * C_in))
      
      # CRITICAL: Compact B before reshaping to handle non-compact arrays from gradient
      B = B.compact()
      B_col = B.reshape((K * K * C_in, C_out))
      
      # Matrix multiplication
      out = A_col @ B_col
      
      # Reshape to output format
      return out.reshape((N, H_out, W_out, C_out))
      ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
      ### BEGIN YOUR SOLUTION
      X, W = node.inputs
      K = W.shape[0]
      
      # Gradient w.r.t. X
      # Handle strided convolutions by dilating out_grad
      if self.stride > 1:
          out_grad_dilated = dilate(out_grad, (1, 2), self.stride - 1)
      else:
          out_grad_dilated = out_grad
      
      # Flip weight kernel: (K, K, C_in, C_out) -> flip over axes 0,1
      W_flipped = flip(W, (0, 1))
      
      # Transpose to (K, K, C_out, C_in) for convolution with out_grad
      W_flipped_transposed = transpose(W_flipped, (2, 3))
      
      # Compute X_grad with appropriate padding
      # padding needed: K - 1 - self.padding
      X_grad = conv(out_grad_dilated, W_flipped_transposed, 
                    stride=1, padding=K - 1 - self.padding)
      
      # Gradient w.r.t. W
      # Need to compute conv where batches are accumulated
      # Transform X: (N, H, W, C_in) -> (C_in, H, W, N)
      # Transform out_grad: (N, H', W', C_out) -> (H', W', N, C_out)
      
      # Permute X to move channels to batch dimension
      X_t = transpose(X, (0, 3))  # (C_in, H, W, N) - swaps axes 0 and 3
      
      # Permute out_grad_dilated to prepare for convolution
      # (N, H, W, C_out) -> (H, W, N, C_out)
      out_grad_t = transpose(out_grad_dilated, (0, 1))  # swap N and H: (H, N, W, C_out)
      out_grad_t = transpose(out_grad_t, (1, 2))        # swap N and W: (H, W, N, C_out)
      
      # Convolve: result is (C_in, K, K, C_out)
      W_grad = conv(X_t, out_grad_t, stride=1, padding=self.padding)
      
      # Permute back to (K, K, C_in, C_out)
      W_grad = transpose(W_grad, (0, 1))  # (K, C_in, K, C_out)
      W_grad = transpose(W_grad, (1, 2))  # (K, K, C_in, C_out)
      
      return X_grad, W_grad
      ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


class BlockSparseAttention(TensorOp):
    def __init__(self, sparse_blocks: List[int], block_size: int):
        self.sparse_blocks = sparse_blocks
        self.block_size = block_size

    def compute(self, q: NDArray, k: NDArray, v: NDArray) -> NDArray:
        # q, k, v are (batch, heads, seq_len, head_dim)
        # Output is same shape
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Create output array
        out = array_api.empty(q.shape, dtype=q.dtype, device=q.device)
        
        # Call backend
        # Note: sparse_blocks is a list of ints
        q.device.block_sparse_attention(
            q.compact()._handle,
            k.compact()._handle,
            v.compact()._handle,
            out._handle,
            self.sparse_blocks,
            batch_size,
            num_heads,
            seq_len,
            head_dim
        )
        
        return out

    def gradient(self, out_grad, node):
        """
        Compute gradients for block sparse attention.
        
        For attention: O = softmax(Q @ K^T / sqrt(d)) @ V
        Uses split operation instead of indexing since Tensor is not subscriptable.
        """
        q, k, v = node.inputs
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Scale factor
        scale = head_dim ** 0.5
        
        # Reshape to 3D for easier handling: merge batch and heads
        # (batch, heads, seq, head_dim) -> (batch*heads, seq, head_dim)
        q_reshaped = reshape(q, (batch_size * num_heads, seq_len, head_dim))
        k_reshaped = reshape(k, (batch_size * num_heads, seq_len, head_dim))
        v_reshaped = reshape(v, (batch_size * num_heads, seq_len, head_dim))
        out_grad_reshaped = reshape(out_grad, (batch_size * num_heads, seq_len, head_dim))
        
        # Split along the batch*heads dimension to get individual 2D tensors
        q_splits = split(q_reshaped, axis=0)  # tuple of (seq, head_dim) tensors
        k_splits = split(k_reshaped, axis=0)
        v_splits = split(v_reshaped, axis=0)
        out_grad_splits = split(out_grad_reshaped, axis=0)
        
        # Process each head
        attention_weights_list = []
        scores_list = []
        
        for i in range(batch_size * num_heads):
            # Get 2D slices - these are now (1, seq, head_dim), need to squeeze
            q_3d = q_splits[i]  # (1, seq, head_dim)
            k_3d = k_splits[i]
            
            # Reshape to remove the first dimension
            q_2d = reshape(q_3d, (seq_len, head_dim))
            k_2d = reshape(k_3d, (seq_len, head_dim))
            
            # Compute scores: (seq, head_dim) @ (head_dim, seq) = (seq, seq)
            k_2d_t = transpose(k_2d)  # (head_dim, seq)
            scores_2d = q_2d @ k_2d_t / scale  # (seq, seq)
            scores_list.append(scores_2d)
            
            # Apply softmax - need to handle broadcasting manually
            scores_exp = exp(scores_2d)
            scores_sum = summation(scores_exp, axes=(1,))  # (seq,)
            scores_sum = reshape(scores_sum, (seq_len, 1))
            # Broadcast scores_sum to (seq, seq) for division
            scores_sum_broadcasted = broadcast_to(scores_sum, (seq_len, seq_len))
            attention_2d = scores_exp / scores_sum_broadcasted  # (seq, seq)
            attention_weights_list.append(attention_2d)
        
        # Stack attention weights
        attention_weights = stack(attention_weights_list, axis=0)  # (batch*heads, seq, seq)
        
        # Split attention weights for gradient computation
        attention_splits = split(attention_weights, axis=0)
        
        # Compute gradients
        grad_v_list = []
        grad_q_list = []
        grad_k_list = []
        
        for i in range(batch_size * num_heads):
            # Get 2D tensors
            attention_3d = attention_splits[i]  # (1, seq, seq)
            attention_2d = reshape(attention_3d, (seq_len, seq_len))
            
            out_grad_3d = out_grad_splits[i]  # (1, seq, head_dim)
            out_grad_2d = reshape(out_grad_3d, (seq_len, head_dim))
            
            v_3d = v_splits[i]  # (1, seq, head_dim)
            v_2d = reshape(v_3d, (seq_len, head_dim))
            
            q_3d = q_splits[i]  # (1, seq, head_dim)
            q_2d = reshape(q_3d, (seq_len, head_dim))
            
            k_3d = k_splits[i]  # (1, seq, head_dim)
            k_2d = reshape(k_3d, (seq_len, head_dim))
            
            # Gradient w.r.t. V: A^T @ dL/dO
            attention_2d_t = transpose(attention_2d)  # (seq, seq)
            grad_v_2d = attention_2d_t @ out_grad_2d  # (seq, head_dim)
            grad_v_list.append(grad_v_2d)
            
            # Gradient w.r.t. attention weights: dL/dO @ V^T
            v_2d_t = transpose(v_2d)  # (head_dim, seq)
            grad_attention_2d = out_grad_2d @ v_2d_t  # (seq, seq)
            
            # Gradient through softmax
            # dL/dS = A * (dL/dA - sum(dL/dA * A, axis=-1, keepdims=True))
            grad_attention_weighted = grad_attention_2d * attention_2d
            grad_attention_sum = summation(grad_attention_weighted, axes=(1,))  # (seq,)
            grad_attention_sum = reshape(grad_attention_sum, (seq_len, 1))
            # Broadcast for subtraction
            grad_attention_sum_broadcasted = broadcast_to(grad_attention_sum, (seq_len, seq_len))
            grad_scores_2d = attention_2d * (grad_attention_2d - grad_attention_sum_broadcasted)
            
            # Scale by 1/sqrt(d)
            grad_scores_2d = grad_scores_2d / scale
            
            # Gradient w.r.t. Q: dL/dS @ K
            grad_q_2d = grad_scores_2d @ k_2d  # (seq, head_dim)
            grad_q_list.append(grad_q_2d)
            
            # Gradient w.r.t. K: dL/dS^T @ Q
            grad_scores_2d_t = transpose(grad_scores_2d)  # (seq, seq)
            grad_k_2d = grad_scores_2d_t @ q_2d  # (seq, head_dim)
            grad_k_list.append(grad_k_2d)
        
        # Stack all gradients
        grad_v = stack(grad_v_list, axis=0)  # (batch*heads, seq, head_dim)
        grad_q = stack(grad_q_list, axis=0)  # (batch*heads, seq, head_dim)
        grad_k = stack(grad_k_list, axis=0)  # (batch*heads, seq, head_dim)
        
        # Reshape back to 4D
        grad_v = reshape(grad_v, (batch_size, num_heads, seq_len, head_dim))
        grad_q = reshape(grad_q, (batch_size, num_heads, seq_len, head_dim))
        grad_k = reshape(grad_k, (batch_size, num_heads, seq_len, head_dim))
        
        return grad_q, grad_k, grad_v


def block_sparse_attention(q, k, v, sparse_blocks, block_size):
    return BlockSparseAttention(sparse_blocks, block_size)(q, k, v)