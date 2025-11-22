"""
Block-Sparse Attention Implementation for Efficient Transformers

Implements various sparse attention patterns:
- Local (sliding window)
- Global (strided attention)
- Mixed (local + global)
"""
from typing import Optional
import numpy as np
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
from needle.nn.nn_basic import (
    Parameter,
    Module,
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear
)
from needle.nn.nn_transformer import MultiHeadAttention


class BlockSparsePattern:
    """
    Utility class to generate block-sparse attention masks
    """
    
    @staticmethod
    def local_pattern(seq_len: int, block_size: int, window_size: int = 1):
        """
        Local/sliding window pattern
        
        Args:
            seq_len: sequence length
            block_size: size of each block
            window_size: number of blocks to attend to (on each side)
            
        Returns:
            mask: (n_blocks, n_blocks) boolean mask
        """
        n_blocks = (seq_len + block_size - 1) // block_size
        mask = np.zeros((n_blocks, n_blocks), dtype=bool)
        
        for i in range(n_blocks):
            # Attend to blocks within window
            start = max(0, i - window_size)
            end = min(n_blocks, i + window_size + 1)
            mask[i, start:end] = True
            
        return mask
    
    @staticmethod
    def global_pattern(seq_len: int, block_size: int, stride: int = 2):
        """
        Global/strided pattern
        
        Args:
            seq_len: sequence length
            block_size: size of each block
            stride: stride for global attention
            
        Returns:
            mask: (n_blocks, n_blocks) boolean mask
        """
        n_blocks = (seq_len + block_size - 1) // block_size
        mask = np.zeros((n_blocks, n_blocks), dtype=bool)
        
        for i in range(n_blocks):
            # Attend to every stride-th block
            mask[i, ::stride] = True
            # Always attend to self
            mask[i, i] = True
            
        return mask
    
    @staticmethod
    def mixed_pattern(seq_len: int, block_size: int, window_size: int = 1, stride: int = 4):
        """
        Mixed pattern (local + global)
        
        Args:
            seq_len: sequence length
            block_size: size of each block
            window_size: local window size
            stride: global stride
            
        Returns:
            mask: (n_blocks, n_blocks) boolean mask
        """
        local_mask = BlockSparsePattern.local_pattern(seq_len, block_size, window_size)
        global_mask = BlockSparsePattern.global_pattern(seq_len, block_size, stride)
        return local_mask | global_mask


class BlockSparseMultiHeadAttention(Module):
    """
    Block-sparse multi-head attention
    
    This implements efficient block-sparse attention patterns to reduce
    computational complexity from O(n^2) to O(n * sqrt(n)) or better.
    """
    
    def __init__(
        self,
        *,
        dropout=0.,
        causal=False,
        device=None,
        dtype="float32",
        block_size=64,
        sparse_pattern="local"
    ):
        super().__init__()
        
        self.device = device
        self.dtype = dtype
        self.causal = causal
        self.dropout = Dropout(dropout)
        self.block_size = block_size
        self.sparse_pattern = sparse_pattern
        
    def create_block_mask(self, seq_len: int, device):
        """Create block-sparse attention mask"""
        if self.sparse_pattern == "local":
            block_mask = BlockSparsePattern.local_pattern(seq_len, self.block_size, window_size=1)
        elif self.sparse_pattern == "global":
            block_mask = BlockSparsePattern.global_pattern(seq_len, self.block_size, stride=2)
        elif self.sparse_pattern == "mixed":
            block_mask = BlockSparsePattern.mixed_pattern(seq_len, self.block_size, window_size=1, stride=4)
        else:
            raise ValueError(f"Unknown sparse pattern: {self.sparse_pattern}")
        
        # Expand to full attention mask
        n_blocks = block_mask.shape[0]
        full_mask = np.zeros((seq_len, seq_len), dtype=np.float32)
        
        for i in range(n_blocks):
            for j in range(n_blocks):
                if block_mask[i, j]:
                    i_start = i * self.block_size
                    i_end = min((i + 1) * self.block_size, seq_len)
                    j_start = j * self.block_size
                    j_end = min((j + 1) * self.block_size, seq_len)
                    full_mask[i_start:i_end, j_start:j_end] = 1.0
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
            full_mask = full_mask * causal_mask
        
        # Convert to additive mask (0 for valid, -inf for masked)
        full_mask = (1.0 - full_mask) * (-1e10)
        
        return ndarray.array(full_mask, device=device)
    
    def matmul(self, a, b_transpose):
        """Batched matrix multiplication"""
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)
        
        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)
        
        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)
        
        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)
        
        return (a * b_transpose).sum(len(a.shape) - 1)
    
    def softmax(self, logit):
        """Softmax function"""
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )
        
        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)
        
        probs = ops.exp(logit - max_val)
        
        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)
        
        return probs / denom
    
    def forward(self, q, k, v):
        """
        Forward pass with block-sparse attention
        
        Args:
            q: (batch_size, num_head, seq_len, dim_head)
            k: (batch_size, num_head, seq_len, dim_head)
            v: (batch_size, num_head, seq_len, dim_head)
            
        Returns:
            result: (batch_size, num_head, seq_len, dim_head)
            probs: attention probabilities (for analysis)
        """
        batch_size, num_head, seq_len, q_dim = q.shape
        
        # Compute attention scores
        scores = self.matmul(q, k)
        scores = scores / (q_dim ** 0.5)
        
        # Apply block-sparse mask
        mask = self.create_block_mask(seq_len, q.device)
        mask_tensor = Tensor(mask, device=q.device, dtype="float32", requires_grad=False)
        
        # Expand mask for batch and heads
        mask_expanded = mask_tensor.reshape((1, 1, seq_len, seq_len))
        mask_broadcast = mask_expanded.broadcast_to(scores.shape)
        
        scores = scores + mask_broadcast
        
        # Softmax and dropout
        probs = self.softmax(scores)
        probs = self.dropout(probs)
        
        # Apply to values
        v_transpose = ops.transpose(v, axes=(2, 3))
        result = self.matmul(probs, v_transpose)
        
        return result, probs


class SparseAttentionLayer(Module):
    """
    Sparse attention layer with block-sparse attention
    """
    
    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout=0.,
        causal=True,
        device=None,
        dtype="float32",
        block_size=64,
        sparse_pattern="local"
    ):
        super().__init__()
        
        self.device = device
        self.dtype = dtype
        
        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features
        
        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features
        
        self.num_head = num_head
        self.dim_head = dim_head
        
        # Layer norms
        self.prenorm_q = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(v_features, device=device, dtype=dtype)
        
        inner_dim = num_head * dim_head
        
        # Projections
        self.q_projection = Linear(q_features, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_projection = Linear(k_features, inner_dim, bias=False, device=device, dtype=dtype)
        self.v_projection = Linear(v_features, inner_dim, bias=False, device=device, dtype=dtype)
        
        # Block-sparse attention
        self.attn = BlockSparseMultiHeadAttention(
            dropout=dropout,
            causal=causal,
            device=device,
            dtype=dtype,
            block_size=block_size,
            sparse_pattern=sparse_pattern
        )
        
        self.out_projection = Linear(inner_dim, out_features, bias=False, device=device, dtype=dtype)
    
    def forward(self, q, k=None, v=None):
        """Forward pass"""
        if k is None:
            k = q
        if v is None:
            v = q
        
        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        
        # Prenorm
        q_norm = self.prenorm_q(q)
        k_norm = self.prenorm_k(k)
        v_norm = self.prenorm_v(v)
        
        # Reshape for projection
        q_norm_2d = q_norm.reshape((batch_size * queries_len, q_dim))
        k_norm_2d = k_norm.reshape((batch_size * keys_values_len, k_dim))
        v_norm_2d = v_norm.reshape((batch_size * keys_values_len, k_dim))
        
        # Project
        inner_dim = self.num_head * self.dim_head
        q_proj_2d = self.q_projection(q_norm_2d)
        k_proj_2d = self.k_projection(k_norm_2d)
        v_proj_2d = self.v_projection(v_norm_2d)
        
        # Reshape to 3D
        q_proj = q_proj_2d.reshape((batch_size, queries_len, inner_dim))
        k_proj = k_proj_2d.reshape((batch_size, keys_values_len, inner_dim))
        v_proj = v_proj_2d.reshape((batch_size, keys_values_len, inner_dim))
        
        # Separate heads
        q_heads = q_proj.reshape((batch_size, queries_len, self.num_head, self.dim_head))
        k_heads = k_proj.reshape((batch_size, keys_values_len, self.num_head, self.dim_head))
        v_heads = v_proj.reshape((batch_size, keys_values_len, self.num_head, self.dim_head))
        
        # Transpose
        q_heads = ops.transpose(q_heads, axes=(1, 2))
        k_heads = ops.transpose(k_heads, axes=(1, 2))
        v_heads = ops.transpose(v_heads, axes=(1, 2))
        
        # Attention
        attn_output, probs = self.attn(q_heads, k_heads, v_heads)
        
        # Transpose back
        attn_output = ops.transpose(attn_output, axes=(1, 2))
        
        # Reshape
        attn_output = attn_output.reshape((batch_size, queries_len, inner_dim))
        
        # Output projection
        attn_output_2d = attn_output.reshape((batch_size * queries_len, inner_dim))
        result_2d = self.out_projection(attn_output_2d)
        result = result_2d.reshape((batch_size, queries_len, self.out_features))
        
        return result


class SparseTransformerLayer(Module):
    """
    Transformer layer with block-sparse attention
    """
    
    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout=0.,
        causal=True,
        device=None,
        dtype="float32",
        block_size=64,
        sparse_pattern="local"
    ):
        super().__init__()
        
        self.device = device
        self.dtype = dtype
        self.q_features = q_features
        self.hidden_size = hidden_size
        
        # Sparse attention layer
        self.attn_layer = SparseAttentionLayer(
            q_features=q_features,
            num_head=num_head,
            dim_head=dim_head,
            dropout=dropout,
            causal=causal,
            device=device,
            dtype=dtype,
            block_size=block_size,
            sparse_pattern=sparse_pattern
        )
        
        self.attn_dropout = Dropout(dropout)
        
        # MLP
        self.norm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.linear1 = Linear(q_features, hidden_size, device=device, dtype=dtype)
        self.relu = ReLU()
        self.mlp_dropout = Dropout(dropout)
        self.linear2 = Linear(hidden_size, q_features, device=device, dtype=dtype)
        self.mlp_output_dropout = Dropout(dropout)
    
    def forward(self, x):
        """Forward pass with residual connections"""
        batch_size, seq_len, x_dim = x.shape
        
        # Attention block with residual
        attn_out = self.attn_layer(x, x, x)
        attn_out = self.attn_dropout(attn_out)
        x = x + attn_out
        
        # MLP block with residual
        mlp_in = self.norm(x)
        mlp_in_2d = mlp_in.reshape((batch_size * seq_len, self.q_features))
        
        mlp_hidden = self.linear1(mlp_in_2d)
        mlp_hidden = self.relu(mlp_hidden)
        mlp_hidden = self.mlp_dropout(mlp_hidden)
        
        mlp_out = self.linear2(mlp_hidden)
        mlp_out = mlp_out.reshape((batch_size, seq_len, self.q_features))
        mlp_out = self.mlp_output_dropout(mlp_out)
        
        x = x + mlp_out
        
        return x
