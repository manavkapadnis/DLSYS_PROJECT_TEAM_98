"""
Pythia-70M Language Model Implementation
Based on EleutherAI's Pythia architecture
"""
import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import numpy as np


class PythiaConfig:
    """Configuration for Pythia-70M model"""
    def __init__(
        self,
        vocab_size=10000,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=256,
        dropout=0.1,
        device=None,
        dtype="float32",
        use_sparse_attention=False,
        sparse_block_size=64,
        sparse_pattern="local"  # "local", "global", or "mixed"
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
        # Embedding: vocab_size * d_model
        embedding_params = self.vocab_size * self.d_model
        
        # Transformer layers
        # Each layer: 4 * d_model^2 (QKV + output proj) + 2 * d_model * d_ff (FFN)
        layer_params = (4 * self.d_model * self.d_model + 2 * self.d_model * self.d_ff)
        transformer_params = self.num_layers * layer_params
        
        # Layer norms and position embeddings
        misc_params = self.max_seq_len * self.d_model + self.num_layers * 2 * self.d_model
        
        return embedding_params + transformer_params + misc_params


class PythiaLM(nn.Module):
    """
    Pythia-70M Language Model
    
    Architecture matches EleutherAI's Pythia-70M:
    - 6 transformer layers
    - 512 hidden dimension
    - 8 attention heads
    - 2048 feedforward dimension
    """
    def __init__(self, config: PythiaConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            device=config.device,
            dtype=config.dtype
        )
        
        # Positional embedding (learned)
        self.pos_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model,
            device=config.device,
            dtype=config.dtype
        )
        
        # Dropout for embeddings
        self.emb_dropout = nn.Dropout(config.dropout)
        
        # Transformer layers
        if config.use_sparse_attention:
            from python.needle.nn.nn_sparse_attention import SparseTransformerLayer
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
        
        # Create position indices
        positions = np.arange(seq_len).reshape(1, seq_len)
        positions = np.tile(positions, (batch_size, 1))
        pos_tensor = ndl.Tensor(positions, device=self.config.device, dtype=self.config.dtype)
        
        # Get embeddings
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
            logits_last = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply temperature
            logits_last = logits_last / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                logits_np = logits_last.numpy()
                top_k_indices = np.argsort(logits_np, axis=-1)[:, -top_k:]
                mask = np.zeros_like(logits_np)
                for i in range(logits_np.shape[0]):
                    mask[i, top_k_indices[i]] = 1
                logits_np = logits_np * mask + (1 - mask) * (-1e10)
                logits_last = ndl.Tensor(logits_np, device=self.config.device)
            
            # Sample from distribution
            probs = ndl.ops.exp(logits_last)
            probs_sum = ndl.ops.summation(probs, axes=(1,))
            probs_sum = probs_sum.reshape((probs.shape[0], 1))
            probs = probs / ndl.ops.broadcast_to(probs_sum, probs.shape)
            
            # Sample next token (using numpy for simplicity)
            probs_np = probs.numpy()
            next_tokens = []
            for i in range(probs_np.shape[0]):
                next_token = np.random.choice(self.config.vocab_size, p=probs_np[i])
                next_tokens.append(next_token)
            
            next_tokens = np.array(next_tokens).reshape(-1, 1)
            next_tokens_tensor = ndl.Tensor(next_tokens, device=self.config.device)
            
            # Append to sequence
            input_ids = ndl.ops.concat([input_ids, next_tokens_tensor], axis=1)
        
        return input_ids


def create_pythia_70m(
    vocab_size=10000,
    max_seq_len=256,
    use_sparse_attention=False,
    device=None,
    dtype="float32"
):
    """
    Factory function to create Pythia-70M model
    
    Args:
        vocab_size: size of vocabulary
        max_seq_len: maximum sequence length
        use_sparse_attention: whether to use block-sparse attention
        device: device to place model on
        dtype: data type
        
    Returns:
        model: PythiaLM instance
        config: PythiaConfig instance
    """
    config = PythiaConfig(
        vocab_size=vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=max_seq_len,
        dropout=0.1,
        device=device,
        dtype=dtype,
        use_sparse_attention=use_sparse_attention,
        sparse_block_size=64,
        sparse_pattern="local"
    )
    
    model = PythiaLM(config)
    
    print(f"Created Pythia-70M model with ~{config.get_total_params() / 1e6:.1f}M parameters")
    print(f"Sparse attention: {use_sparse_attention}")
    
    return model, config


if __name__ == "__main__":
    # Test model creation
    device = ndl.cpu()
    model, config = create_pythia_70m(use_sparse_attention=False, device=device)
    
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
