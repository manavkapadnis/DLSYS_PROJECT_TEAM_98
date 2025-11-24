"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # Initialize weight first (required order for tests)
        # Weight shape: (in_features, out_features)
        # Use Kaiming uniform with fan_in = in_features
        self.weight = Parameter(init.kaiming_uniform(
            fan_in=in_features, 
            fan_out=out_features, 
            device=device, 
            dtype=dtype
        ))
        
        # Initialize bias if needed
        if bias:
            # Use Kaiming uniform with fan_in = out_features (as specified)
            # kaiming_uniform(out_features, 1) gives shape (out_features, 1)
            # We need to transpose to get (1, out_features) as expected by tests
            bias_init = init.kaiming_uniform(
                fan_in=out_features,
                fan_out=1,
                device=device,
                dtype=dtype
            )
            # Transpose from (out_features, 1) to (1, out_features)
            self.bias = Parameter(ops.transpose(bias_init))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # Matrix multiplication: X @ weight
        out = X @ self.weight
        
        # Add bias if it exists (explicit broadcasting required)
        if self.bias is not None:
            # bias shape: (1, out_features)
            # out shape: (N, out_features)
            # Need to broadcast bias to match out shape
            out = out + ops.broadcast_to(self.bias, out.shape)
        
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # ReLU(x) = max(0, x) - element-wise maximum between 0 and input
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        out = x
        for module in self.modules:
            out = module(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Need to import the functions at the top of the file or within the method
        from ..ops.ops_logarithmic import logsumexp
        from ..ops.ops_mathematic import summation
        
        # logits: (batch_size, num_classes) - raw predictions  
        # y: (batch_size,) - integer class labels (not one-hot)
        
        # Softmax loss formula: ℓ_softmax(z,y) = log(∑_i exp(z_i)) - z_y
        # This is equivalent to: LogSumExp(z) - z_y
        
        # Step 1: Compute LogSumExp along the class dimension (axis=1)
        # This gives log(∑_i exp(z_i)) for each sample
        logsumexp_result = logsumexp(logits, axes=(1,))  # shape: (batch_size,)
        
        # Step 2: Extract the logits corresponding to the true labels z_y
        # Convert integer labels to one-hot encoding
        batch_size, num_classes = logits.shape
        y_one_hot = init.one_hot(num_classes, y, dtype=logits.dtype, device=logits.device)
        # y_one_hot shape: (batch_size, num_classes)
        
        # Extract logits[i, y[i]] for each sample i using element-wise multiplication and sum
        selected_logits = summation(logits * y_one_hot, axes=(1,))  # shape: (batch_size,)
        
        # Step 3: Compute the loss = LogSumExp(logits) - logits[true_labels]
        loss = logsumexp_result - selected_logits  # shape: (batch_size,)
        
        # Step 4: Return average loss over the batch
        return summation(loss) / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        # Running statistics (not parameters, so use .data)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype).data
        self.running_var = init.ones(dim, device=device, dtype=dtype).data
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x shape: (batch_size, features)
        batch_size = x.shape[0]
        
        if self.training:
            # Training mode: use batch statistics and update running statistics
            # Compute batch mean and variance across batch dimension (axis=0)
            batch_mean = ops.summation(x, axes=(0,)) / batch_size  # shape: (features,)
            
            # Compute batch variance: E[(x - mean)^2]
            mean_broadcast = ops.broadcast_to(ops.reshape(batch_mean, (1, -1)), x.shape)
            centered = x - mean_broadcast
            batch_var = ops.summation(centered ** 2, axes=(0,)) / batch_size  # shape: (features,)
            
            # Update running statistics: new = (1 - momentum) * old + momentum * observed
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data
            
            # Use batch statistics for normalization
            mean_for_norm = batch_mean
            var_for_norm = batch_var
        else:
            # Evaluation mode: use running statistics
            mean_for_norm = Tensor(self.running_mean, device=x.device, dtype=x.dtype)
            var_for_norm = Tensor(self.running_var, device=x.device, dtype=x.dtype)
        
        # Normalize: (x - mean) / sqrt(var + eps)
        mean_broadcast = ops.broadcast_to(ops.reshape(mean_for_norm, (1, -1)), x.shape)
        var_broadcast = ops.broadcast_to(ops.reshape(var_for_norm, (1, -1)), x.shape)
        
        normalized = (x - mean_broadcast) / ops.power_scalar(var_broadcast + self.eps, 0.5)
        
        # Apply scale and shift: weight * normalized + bias
        weight_broadcast = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
        bias_broadcast = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)
        
        return weight_broadcast * normalized + bias_broadcast
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # Initialize learnable parameters: weight (scale) and bias (shift)
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Handle both 2D (batch_size, features) and 3D (batch_size, seq_len, features) inputs
        # Normalize over the last dimension (features)
        
        # For 2D: x is (batch_size, features), normalize over axis=1
        # For 3D: x is (batch_size, seq_len, features), normalize over axis=2
        norm_axis = len(x.shape) - 1  # Last axis
        
        # Compute mean over feature dimension
        mean = ops.summation(x, axes=(norm_axis,)) / self.dim
        # Reshape to add back the feature dimension as 1 for broadcasting
        mean_shape = list(x.shape)
        mean_shape[norm_axis] = 1
        mean = ops.reshape(mean, tuple(mean_shape))
        mean_broadcast = ops.broadcast_to(mean, x.shape)
        
        # Compute variance: E[(x - mean)^2]
        centered = x - mean_broadcast
        var = ops.summation(centered ** 2, axes=(norm_axis,)) / self.dim
        var = ops.reshape(var, tuple(mean_shape))
        var_broadcast = ops.broadcast_to(var, x.shape)
        
        # Normalize: (x - mean) / sqrt(var + eps)
        normalized = centered / ops.power_scalar(var_broadcast + self.eps, 0.5)
        
        # Apply scale and shift: weight * normalized + bias
        # Reshape weight and bias to broadcast correctly
        weight_shape = [1] * len(x.shape)
        weight_shape[norm_axis] = self.dim
        weight_broadcast = ops.broadcast_to(ops.reshape(self.weight, tuple(weight_shape)), x.shape)
        bias_broadcast = ops.broadcast_to(ops.reshape(self.bias, tuple(weight_shape)), x.shape)
        
        return weight_broadcast * normalized + bias_broadcast
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # Generate random mask: 1 with probability (1-p), 0 with probability p
            mask = init.randb(*x.shape, p=(1-self.p), device=x.device, dtype=x.dtype)
            
            # Scale by 1/(1-p) to maintain expected value (inverted dropout)
            # This ensures no scaling needed during evaluation
            return x * mask / (1 - self.p)
        else:
            # Evaluation mode: identity function (no dropout)
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Residual/skip connection: F(x) + x
        return self.fn(x) + x
        ### END YOUR SOLUTION

class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Flatten all dimensions except batch dimension
        # X shape: (N, ...)
        batch_size = X.shape[0]
        return X.reshape((batch_size, -1))
        ### END YOUR SOLUTION
