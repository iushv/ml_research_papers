"""
Transformer Block Implementations

This module provides Transformer encoder blocks for both:
1. Standard Transformer (baseline)
2. Forgetting Transformer (FoX)

Each block consists of:
- Multi-Head (Forgetting) Attention
- Feed-Forward Network
- Layer Normalization
- Residual Connections
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from standard_attention import MultiHeadAttention
    from forgetting_attention import MultiHeadForgettingAttention
except ImportError:
    from .standard_attention import MultiHeadAttention
    from .forgetting_attention import MultiHeadForgettingAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = GELU(x·W1 + b1)·W2 + b2
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (default: 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class StandardTransformerBlock(nn.Module):
    """
    Standard Transformer Encoder Block.
    
    Architecture:
        x → LayerNorm → MultiHeadAttention → Dropout → + → 
        ↑                                               ↓
        ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
        ↓
        → LayerNorm → FeedForward → Dropout → + → output
        ↑                                      ↓
        ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention_weights: Whether to return attention weights
            
        Returns:
            output: Output tensor (batch, seq_len, d_model)
            attention_weights: Optional attention weights
        """
        # Pre-norm attention
        normed = self.norm1(x)
        attn_output, attn_weights = self.attention(
            normed, normed, normed,
            mask=mask,
            return_attention_weights=return_attention_weights
        )
        x = x + self.dropout1(attn_output)
        
        # Pre-norm feed-forward
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout2(ff_output)
        
        return x, attn_weights


class ForgettingTransformerBlock(nn.Module):
    """
    Forgetting Transformer (FoX) Encoder Block.
    
    Same architecture as StandardTransformerBlock but uses
    MultiHeadForgettingAttention with forget gates.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadForgettingAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
        return_forget_gate: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention_weights: Whether to return attention weights
            return_forget_gate: Whether to return forget gate values
            
        Returns:
            output: Output tensor (batch, seq_len, d_model)
            attention_weights: Optional attention weights
            forget_gate: Optional forget gate values
        """
        # Pre-norm attention
        normed = self.norm1(x)
        attn_output, attn_weights, forget_gate = self.attention(
            normed, normed, normed,
            mask=mask,
            return_attention_weights=return_attention_weights,
            return_forget_gate=return_forget_gate
        )
        x = x + self.dropout1(attn_output)
        
        # Pre-norm feed-forward
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout2(ff_output)
        
        return x, attn_weights, forget_gate


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer encoder blocks.
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        use_forgetting: bool = False
    ):
        """
        Args:
            num_layers: Number of encoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            use_forgetting: Whether to use Forgetting Attention
        """
        super().__init__()
        
        self.use_forgetting = use_forgetting
        
        BlockClass = ForgettingTransformerBlock if use_forgetting else StandardTransformerBlock
        
        self.layers = nn.ModuleList([
            BlockClass(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through all encoder layers.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention_weights: Whether to return attention weights
            
        Returns:
            output: Output tensor (batch, seq_len, d_model)
            all_attention_weights: List of attention weights from each layer
        """
        all_attention_weights = []
        
        for layer in self.layers:
            if self.use_forgetting:
                x, attn_weights, _ = layer(
                    x, mask=mask,
                    return_attention_weights=return_attention_weights
                )
            else:
                x, attn_weights = layer(
                    x, mask=mask,
                    return_attention_weights=return_attention_weights
                )
            
            if attn_weights is not None:
                all_attention_weights.append(attn_weights)
        
        x = self.final_norm(x)
        
        return x, all_attention_weights


class LanguageModel(nn.Module):
    """
    Simple Transformer Language Model for evaluation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_forgetting: bool = False
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            use_forgetting: Whether to use Forgetting Transformer
        """
        super().__init__()
        
        self.d_model = d_model
        self.use_forgetting = use_forgetting
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding (learnable)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            use_forgetting=use_forgetting
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Tie weights
        self.output_proj.weight = self.embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            attention_weights: List of attention weights from each layer
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        x = self.embedding(input_ids) * (self.d_model ** 0.5)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        causal_mask = causal_mask.unsqueeze(0)  # (1, seq_len, seq_len)
        
        # Encode
        x, attention_weights = self.encoder(
            x, mask=causal_mask,
            return_attention_weights=return_attention_weights
        )
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits, attention_weights
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Unit Tests
# ============================================================================

def run_tests():
    """Run unit tests for Transformer blocks."""
    print("=" * 60)
    print("Testing Transformer Blocks")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 16
    d_model = 64
    num_heads = 4
    vocab_size = 1000
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Test 1: Standard Transformer Block
    print("\n[Test 1] StandardTransformerBlock...")
    std_block = StandardTransformerBlock(d_model, num_heads).to(device)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    output, attn = std_block(x, return_attention_weights=True)
    assert output.shape == x.shape, f"Wrong output shape: {output.shape}"
    print("✅ Passed!")
    
    # Test 2: Forgetting Transformer Block
    print("\n[Test 2] ForgettingTransformerBlock...")
    fox_block = ForgettingTransformerBlock(d_model, num_heads).to(device)
    output, attn, forget = fox_block(x, return_attention_weights=True, return_forget_gate=True)
    assert output.shape == x.shape, f"Wrong output shape: {output.shape}"
    assert forget is not None, "Forget gate not returned"
    print("✅ Passed!")
    
    # Test 3: Language Model (Standard)
    print("\n[Test 3] Language Model (Standard)...")
    std_lm = LanguageModel(vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, num_layers=2, use_forgetting=False).to(device)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    logits, _ = std_lm(input_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Wrong logits shape: {logits.shape}"
    print(f"   Parameters: {std_lm.count_parameters():,}")
    print("✅ Passed!")
    
    # Test 4: Language Model (Forgetting)
    print("\n[Test 4] Language Model (Forgetting)...")
    fox_lm = LanguageModel(vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, num_layers=2, use_forgetting=True).to(device)
    logits, _ = fox_lm(input_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Wrong logits shape: {logits.shape}"
    print(f"   Parameters: {fox_lm.count_parameters():,}")
    print("✅ Passed!")
    
    # Test 5: Gradient flow through both models
    print("\n[Test 5] Gradient flow...")
    for name, model in [("Standard", std_lm), ("Forgetting", fox_lm)]:
        model.zero_grad()
        logits, _ = model(input_ids)
        loss = logits.sum()
        loss.backward()
        
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert grad_norm > 0, f"{name} model has zero gradients"
        print(f"   {name} grad norm: {grad_norm:.4f}")
    print("✅ Passed!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    run_tests()
