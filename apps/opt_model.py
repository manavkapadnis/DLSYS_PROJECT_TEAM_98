"""
OPT-125M Language Model Implementation
Based on Meta AI's OPT (Open Pre-trained Transformer) architecture
Paper: https://arxiv.org/abs/2205.01068
"""
import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import numpy as np
from needle import ops


class OPTConfig:
    """Configuration for OPT-125M model"""
    def __init__(
        self,
        vocab_size=10000,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,  # 4 * d_model for OPT
        max_seq_len=256,
        dropout=0.1,
        device=None,
        dtype="float32",
        use_sparse_attention=False,
        sparse_block_size=64,
        sparse_pattern="local"
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.device = device
        self.dtype = dtype
        self.use_sparse_attention = use_sparse_attention
        self.sparse_block_size = sparse_block_size
        self.sparse_pattern = sparse_pattern
        
        # Derived parameters
        self.dim_head = d_model // num_heads
        
    def get_total_params(self):
        """Calculate total number of parameters"""
        # Token embedding: vocab_size * d_model
        token_emb_params = self.vocab_size * self.d_model
        
        # Position embedding: max_seq_len * d_model
        pos_emb_params = self.max_seq_len * self.d_model
        
        # Transformer layers
        # Each layer: 4 * d_model^2 (QKV + output proj) + 2 * d_model * d_ff (FFN)
        # + layer norms: 2 * d_model per layer
        layer_params = (
            4 * self.d_model * self.d_model +  # Attention weights
            2 * self.d_model * self.d_ff +      # FFN weights
            4 * self.d_model                     # Layer norm params (2 norms * 2 params each)
        )
        transformer_params = self.num_layers * layer_params
        
        # Final layer norm: 2 * d_model
        final_norm_params = 2 * self.d_model
        
        # LM head shares weights with token embedding (weight tying)
        # So no additional parameters
        
        total = token_emb_params + pos_emb_params + transformer_params + final_norm_params
        
        return total


class OPTLM(nn.Module):
    """
    OPT-125M Language Model
    
    Architecture based on Meta AI's OPT-125M:
    - 12 transformer layers
    - 768 hidden dimension
    - 12 attention heads
    - 3072 feedforward dimension
    - ReLU activation in FFN
    - Pre-layer normalization
    - Learned positional embeddings
    """
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            device=config.device,
            dtype=config.dtype
        )
        
        # Positional embedding (learned, starts from position 2 in OPT)
        # OPT uses offset of 2 for historical reasons
        self.pos_embedding = nn.Embedding(
            config.max_seq_len + 2,  # Add offset
            config.d_model,
            device=config.device,
            dtype=config.dtype
        )
        
        # Embedding dropout
        self.emb_dropout = nn.Dropout(config.dropout)
        
        # Transformer layers
        if config.use_sparse_attention:
            from needle.nn.nn_sparse_attention import SparseTransformerLayer
            self.layers = [
                SparseTransformerLayer(
                    q_features=config.d_model,
                    num_head=config.num_heads,
                    dim_head=config.dim_head,
                    hidden_size=config.d_ff,
                    dropout=config.dropout,
                    causal=True,
                    device=config.device,
                    dtype=config.dtype,
                    block_size=config.sparse_block_size,
                    sparse_pattern=config.sparse_pattern
                )
                for _ in range(config.num_layers)
            ]
        else:
            self.layers = [
                nn.TransformerLayer(
                    q_features=config.d_model,
                    num_head=config.num_heads,
                    dim_head=config.dim_head,
                    hidden_size=config.d_ff,
                    dropout=config.dropout,
                    causal=True,
                    device=config.device,
                    dtype=config.dtype
                )
                for _ in range(config.num_layers)
            ]
        
        # Final layer norm
        self.final_norm = nn.LayerNorm1d(
            config.d_model,
            device=config.device,
            dtype=config.dtype
        )
        
        # Output projection (language modeling head)
        # OPT uses weight tying - shares weights with token embedding
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
            device=config.device,
            dtype=config.dtype
        )
        
    def forward(self, input_ids, targets=None):
        """
        Forward pass
        
        Args:
            input_ids: (batch_size, seq_len) token indices
            targets: (batch_size, seq_len) target token indices (optional)
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar loss if targets provided, else None
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position indices (with offset of 2 for OPT)
        positions = np.arange(seq_len) + 2  # OPT positional offset
        positions = positions.reshape(1, seq_len)
        positions = np.tile(positions, (batch_size, 1))
        pos_tensor = ndl.Tensor(positions, device=self.config.device, dtype=self.config.dtype)
        
        # Get embeddings
        # input_ids shape: (batch_size, seq_len)
        # Need to transpose for Embedding layer which expects (seq_len, batch_size)
        token_emb = self.token_embedding(input_ids.reshape((seq_len, batch_size)))
        pos_emb = self.pos_embedding(pos_tensor.reshape((seq_len, batch_size)))
        
        # Transpose to (batch_size, seq_len, d_model)
        token_emb = ndl.ops.transpose(token_emb, axes=(0, 1))
        pos_emb = ndl.ops.transpose(pos_emb, axes=(0, 1))
        
        # Combine embeddings
        x = token_emb + pos_emb
        x = self.emb_dropout(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Reshape for linear layer: (batch_size * seq_len, d_model)
        x_flat = x.reshape((batch_size * seq_len, self.config.d_model))
        
        # Language modeling head
        logits = self.lm_head(x_flat)
        
        # Reshape back: (batch_size, seq_len, vocab_size)
        logits = logits.reshape((batch_size, seq_len, self.config.vocab_size))
        
        loss = None
        if targets is not None:
            # Compute cross-entropy loss
            loss = nn.SoftmaxLoss()(
                logits.reshape((batch_size * seq_len, self.config.vocab_size)),
                targets.reshape((batch_size * seq_len,))
            )
        
        return logits, loss
    
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        """
        Generate text autoregressively
        
        Args:
            input_ids: (batch_size, seq_len) initial token indices
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: if set, only sample from top k tokens
            
        Returns:
            generated: (batch_size, seq_len + max_new_tokens) token indices
        """
        for _ in range(max_new_tokens):
            # Get predictions
            logits, _ = self.forward(input_ids)
            
            # Get logits for last token
            batch_size = logits.shape[0]
            seq_len = logits.shape[1]
            vocab_size = logits.shape[2]
            
            # Extract last token logits
            logits_flat = logits.reshape((batch_size * seq_len, vocab_size))
            
            # Get indices for last token of each sequence
            last_indices = []
            for b in range(batch_size):
                last_indices.append((b + 1) * seq_len - 1)
            
            # Extract logits for last tokens
            logits_last_np = logits_flat.numpy()
            logits_last_list = [logits_last_np[idx] for idx in last_indices]
            logits_last = ndl.Tensor(
                np.array(logits_last_list), 
                device=self.config.device
            )
            
            # Apply temperature
            if temperature != 1.0:
                logits_last = logits_last * (1.0 / temperature)
            
            # Apply top-k filtering if specified
            if top_k is not None:
                logits_np = logits_last.numpy()
                for i in range(logits_np.shape[0]):
                    top_k_indices = np.argsort(logits_np[i])[-top_k:]
                    mask = np.full_like(logits_np[i], -1e10)
                    mask[top_k_indices] = 0
                    logits_np[i] = logits_np[i] + mask
                logits_last = ndl.Tensor(logits_np, device=self.config.device)
            
            # Compute softmax probabilities
            max_logits = ndl.Tensor(
                np.max(logits_last.numpy(), axis=1, keepdims=True),
                device=self.config.device
            )
            logits_shifted = logits_last - ops.broadcast_to(
                max_logits, 
                logits_last.shape
            )
            
            probs = ops.exp(logits_shifted)
            probs_sum = ops.summation(probs, axes=(1,))
            probs_sum = probs_sum.reshape((batch_size, 1))
            probs = probs / ops.broadcast_to(probs_sum, probs.shape)
            
            # Sample next token
            probs_np = probs.numpy()
            next_tokens = []
            for i in range(batch_size):
                p = probs_np[i]
                p = p / p.sum()  # Renormalize
                next_token = np.random.choice(self.config.vocab_size, p=p)
                next_tokens.append(next_token)
            
            next_tokens = np.array(next_tokens).reshape(-1, 1)
            next_tokens_tensor = ndl.Tensor(
                next_tokens, 
                device=self.config.device
            )
            
            # Concatenate to sequence
            input_np = input_ids.numpy()
            next_np = next_tokens_tensor.numpy()
            new_input = np.concatenate([input_np, next_np], axis=1)
            input_ids = ndl.Tensor(new_input, device=self.config.device)
        
        return input_ids


def create_opt_125m(
    vocab_size=10000,
    max_seq_len=256,
    use_sparse_attention=False,
    device=None,
    dtype="float32"
):
    """
    Factory function to create OPT-125M model
    
    Args:
        vocab_size: size of vocabulary
        max_seq_len: maximum sequence length
        use_sparse_attention: whether to use block-sparse attention
        device: device to place model on
        dtype: data type
        
    Returns:
        model: OPTLM instance
        config: OPTConfig instance
    """
    config = OPTConfig(
        vocab_size=vocab_size,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,  # 4x hidden size for OPT
        max_seq_len=max_seq_len,
        dropout=0.1,
        device=device,
        dtype=dtype,
        use_sparse_attention=use_sparse_attention,
        sparse_block_size=64,
        sparse_pattern="local"
    )
    
    model = OPTLM(config)
    
    print(f"Created OPT-125M model with ~{config.get_total_params() / 1e6:.1f}M parameters")
    print(f"Sparse attention: {use_sparse_attention}")
    
    return model, config


if __name__ == "__main__":
    # Test model creation
    device = ndl.cpu()
    model, config = create_opt_125m(use_sparse_attention=False, device=device)
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = ndl.Tensor(
        np.random.randint(0, config.vocab_size, (batch_size, seq_len)),
        device=device
    )
    
    logits, _ = model(input_ids)
    print(f"Output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {config.vocab_size})")
    print("\nModel test successful!")
