"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.data
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * param.data

            if param not in self.u:
                z = np.zeros_like(param.data.numpy(), dtype=param.data.numpy().dtype)
                self.u[param] = ndl.Tensor(z, device=param.device, dtype=param.dtype).data

            # --- FIXED: EMA version ---
            self.u[param] = self.momentum * self.u[param] + (1 - self.momentum) * grad

            update = param.data - self.lr * self.u[param]
            update = ndl.Tensor(update.numpy(), device=param.device, dtype=param.dtype).data
            param.data = update
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        # Increment time step
        self.t += 1

        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.data
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * param.data

            # init state
            if param not in self.m:
                z = np.zeros_like(param.data.numpy(), dtype=param.data.numpy().dtype)
                self.m[param] = ndl.Tensor(z, device=param.device, dtype=param.dtype).data
                self.v[param] = ndl.Tensor(z, device=param.device, dtype=param.dtype).data

            # m_t = β1 m_{t-1} + (1-β1) g
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            # v_t = β2 v_{t-1} + (1-β2) g^2
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad * grad)

            # bias correction
            m_hat = self.m[param] / (1 - (self.beta1 ** self.t))
            v_hat = self.v[param] / (1 - (self.beta2 ** self.t))

            # safe sqrt with dtype/device
            v_hat_np = v_hat.numpy()
            sqrt_v = ndl.Tensor(np.sqrt(v_hat_np), device=param.device, dtype=param.dtype).data

            denom = sqrt_v + self.eps

            update = param.data - self.lr * m_hat / denom

            # force cast back to correct dtype/device
            update = ndl.Tensor(update.numpy(), device=param.device, dtype=param.dtype).data
            param.data = update
        ### END YOUR SOLUTION

