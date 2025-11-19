from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        Input: i, j: the shape of the mask to be created
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        """
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
        """
        The softmax function; 
        """
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

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        # Step 1: Compute Q @ K^T using batched matmul
        # matmul(a, b) computes a @ b^T, so pass k directly
        # q: (B, H, T, D), k: (B, H, T, D) -> scores: (B, H, T, T)
        scores = self.matmul(q, k)
        
        # Step 2: Scale by 1/sqrt(D)
        scores = scores / (q_dim ** 0.5)
        
        # Step 3: Apply causal mask if needed
        if self.causal:
            # create_causal_mask returns shape (1, 1, queries_len, keys_values_len)
            mask = self.create_causal_mask(queries_len, keys_values_len, q.device)
            # Convert to Tensor and broadcast to match scores shape (B, H, T, T)
            mask_tensor = Tensor(mask, device=q.device, dtype="float32", requires_grad=False)
            # Broadcast mask to scores shape
            mask_broadcast = mask_tensor.broadcast_to(scores.shape)
            scores = scores + mask_broadcast
        
        # Step 4: Apply softmax
        probs = self.softmax(scores)
        
        # Step 5: Apply dropout to attention probabilities
        probs = self.dropout(probs)
        
        # Step 6: Multiply by V
        # We want probs @ v
        # matmul(a, b) computes a @ b^T
        # So matmul(probs, v^T) = probs @ (v^T)^T = probs @ v
        # Transpose v: (B, H, T, D) -> (B, H, D, T)
        v_transpose = ops.transpose(v, axes=(2, 3))
        # Compute probs @ v: (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        result = self.matmul(probs, v_transpose)
        
        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
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

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(self, q, k=None, v=None):
        """Forward pass of the self-attention layer."""
        if k is None:
            k = q
        if v is None:
            v = q
        
        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape
        
        ### BEGIN YOUR SOLUTION
        # Step 1: Apply prenorm
        q_norm = self.prenorm_q(q)  # (B, queries_len, q_features)
        k_norm = self.prenorm_k(k)  # (B, keys_values_len, k_features)
        v_norm = self.prenorm_v(v)  # (B, keys_values_len, v_features)

        # Reshape to 2D for linear projection: (B, T, D) -> (B*T, D)
        q_norm_2d = q_norm.reshape((batch_size * queries_len, q_dim))
        k_norm_2d = k_norm.reshape((batch_size * keys_values_len, k_dim))
        v_norm_2d = v_norm.reshape((batch_size * keys_values_len, v_dim))

        # Step 2: Project (now works with 2D inputs)
        inner_dim = self.num_head * self.dim_head
        q_proj_2d = self.q_projection(q_norm_2d)  # (B*queries_len, inner_dim)
        k_proj_2d = self.k_projection(k_norm_2d)  # (B*keys_values_len, inner_dim)
        v_proj_2d = self.v_projection(v_norm_2d)  # (B*keys_values_len, inner_dim)

        # Reshape back to 3D
        q_proj = q_proj_2d.reshape((batch_size, queries_len, inner_dim))
        k_proj = k_proj_2d.reshape((batch_size, keys_values_len, inner_dim))
        v_proj = v_proj_2d.reshape((batch_size, keys_values_len, inner_dim))

        # Step 3: Reshape to separate heads (B, T, inner_dim) -> (B, T, num_head, dim_head)
        q_heads = q_proj.reshape((batch_size, queries_len, self.num_head, self.dim_head))
        k_heads = k_proj.reshape((batch_size, keys_values_len, self.num_head, self.dim_head))
        v_heads = v_proj.reshape((batch_size, keys_values_len, self.num_head, self.dim_head))

        # Transpose to (B, num_head, T, dim_head)
        q_heads = ops.transpose(q_heads, axes=(1, 2))
        k_heads = ops.transpose(k_heads, axes=(1, 2))
        v_heads = ops.transpose(v_heads, axes=(1, 2))

        # Step 4: Apply multi-head attention
        attn_output, probs = self.attn(q_heads, k_heads, v_heads)  # (B, num_head, queries_len, dim_head)

        # Step 5: Transpose back (B, num_head, queries_len, dim_head) -> (B, queries_len, num_head, dim_head)
        attn_output = ops.transpose(attn_output, axes=(1, 2))

        # Reshape to (B, queries_len, inner_dim)
        attn_output = attn_output.reshape((batch_size, queries_len, inner_dim))

        # Step 6: Output projection (reshape to 2D first)
        attn_output_2d = attn_output.reshape((batch_size * queries_len, inner_dim))
        result_2d = self.out_projection(attn_output_2d)
        result = result_2d.reshape((batch_size, queries_len, self.out_features))

        ### END YOUR SOLUTION
                
        return result






class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.q_features = q_features
        self.hidden_size = hidden_size
        
        # Attention layer
        self.attn_layer = AttentionLayer(
            q_features=q_features,
            num_head=num_head,
            dim_head=dim_head,
            dropout=dropout,
            causal=causal,
            device=device,
            dtype=dtype
        )
        
        # Dropout for attention output
        self.attn_dropout = Dropout(dropout)
        
        # MLP (feedforward network)
        self.norm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.linear1 = Linear(q_features, hidden_size, device=device, dtype=dtype)
        self.relu = ReLU()
        self.mlp_dropout = Dropout(dropout)
        self.linear2 = Linear(hidden_size, q_features, device=device, dtype=dtype)
        self.mlp_output_dropout = Dropout(dropout)
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        # First residual block: x = x + Dropout(Attention(x))
        attn_out = self.attn_layer(x, x, x)  # Self-attention
        attn_out = self.attn_dropout(attn_out)
        x = x + attn_out
        
        # Second residual block: x = x + Dropout(Linear_2(Dropout(ReLU(Linear_1(LayerNorm1d(x))))))
        # Apply LayerNorm
        mlp_in = self.norm(x)  # (batch_size, seq_len, q_features)
        
        # Reshape to 2D for Linear layers
        mlp_in_2d = mlp_in.reshape((batch_size * seq_len, self.q_features))
        
        # Linear_1
        mlp_hidden = self.linear1(mlp_in_2d)  # (batch_size * seq_len, hidden_size)
        
        # ReLU
        mlp_hidden = self.relu(mlp_hidden)
        
        # Dropout
        mlp_hidden = self.mlp_dropout(mlp_hidden)
        
        # Linear_2
        mlp_out = self.linear2(mlp_hidden)  # (batch_size * seq_len, q_features)
        
        # Reshape back to 3D
        mlp_out = mlp_out.reshape((batch_size, seq_len, self.q_features))
        
        # Final dropout and residual connection
        mlp_out = self.mlp_output_dropout(mlp_out)
        x = x + mlp_out
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first
        self.embedding_size = embedding_size
        self.sequence_len = sequence_len

        ### BEGIN YOUR SOLUTION
        # Learnable positional embeddings
        self.pos_embedding = Embedding(
            num_embeddings=sequence_len,
            embedding_dim=embedding_size,
            device=device,
            dtype=dtype
        )
        
        # Stack of transformer layers
        self.layers = []
        for _ in range(num_layers):
            layer = TransformerLayer(
                q_features=embedding_size,
                num_head=num_head,
                dim_head=dim_head,
                hidden_size=hidden_size,
                dropout=dropout,
                causal=causal,
                device=device,
                dtype=dtype
            )
            self.layers.append(layer)
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        # x shape: (batch_size, seq_len, embedding_size)
        batch_size, seq_len, _ = x.shape
        
        # Create position indices: shape (seq_len, batch_size)
        # Each row represents a position (0, 1, 2, ..., seq_len-1)
        # Each column represents a batch element (all have same positions)
        positions = np.arange(seq_len).reshape(seq_len, 1)  # (seq_len, 1)
        positions = np.tile(positions, (1, batch_size))  # (seq_len, batch_size)
        
        # Convert to Tensor (Embedding expects integer indices)
        pos_tensor = Tensor(positions, device=self.device, dtype=self.dtype)
        
        # Get positional embeddings: (seq_len, batch_size, embedding_size)
        pos_emb = self.pos_embedding(pos_tensor)
        
        # Transpose to match x shape: (batch_size, seq_len, embedding_size)
        pos_emb = ops.transpose(pos_emb, axes=(0, 1))
        
        # Add positional embeddings to input
        x = x + pos_emb
        
        # Pass through all transformer layers
        for layer in self.layers:
            x = layer(x)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)