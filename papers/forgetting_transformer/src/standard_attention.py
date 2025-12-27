"""
Standard Scaled Dot-Product Attention Implementation

This module implements the baseline attention mechanism as described in
"Attention Is All You Need" (Vaswani et al., 2017).

Mathematical Foundation:
    Attention(Q, K, V) = softmax(QK^T / √d_k) · V

Where:
    - Q: Query matrix of shape (batch, seq_len, d_model)
    - K: Key matrix of shape (batch, seq_len, d_model)  
    - V: Value matrix of shape (batch, seq_len, d_model)
    - d_k: Dimension of keys (used for scaling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    Computes attention weights and applies them to values:
        Attention(Q, K, V) = softmax(QK^T / √d_k) · V
    """
    
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch, seq_len, d_k)
            key: Key tensor of shape (batch, seq_len, d_k)
            value: Value tensor of shape (batch, seq_len, d_v)
            mask: Optional attention mask (1 = attend, 0 = mask)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            output: Attention output of shape (batch, seq_len, d_v)
            attention_weights: Optional attention weights (batch, seq_len, seq_len)
        """
        d_k = query.size(-1)
        
        # Compute attention scores: QK^T / √d_k
        # Shape: (batch, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided (set masked positions to -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)
        
        if return_attention_weights:
            return output, attention_weights
        return output, None


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Splits the model dimension into multiple heads, applies attention
    independently, then concatenates and projects the results.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
    where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear projections
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor of shape (batch, seq_len, d_model)
            key: Key tensor of shape (batch, seq_len, d_model)
            value: Value tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention_weights: Whether to return attention weights
            
        Returns:
            output: Attention output of shape (batch, seq_len, d_model)
            attention_weights: Optional attention weights
        """
        batch_size = query.size(0)
        
        # 1. Linear projections: (batch, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. Reshape for multi-head: (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply attention
        # Reshape for batch processing: (batch * num_heads, seq_len, d_k)
        Q = Q.contiguous().view(batch_size * self.num_heads, -1, self.d_k)
        K = K.contiguous().view(batch_size * self.num_heads, -1, self.d_k)
        V = V.contiguous().view(batch_size * self.num_heads, -1, self.d_k)
        
        if mask is not None:
            # Expand mask for all heads
            # Input mask shape: (batch, seq, seq) or (1, seq, seq)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq, seq)
            mask = mask.expand(batch_size, self.num_heads, -1, -1)
            mask = mask.contiguous().view(batch_size * self.num_heads, mask.size(-2), mask.size(-1))
        
        attn_output, attn_weights = self.attention(
            Q, K, V, mask=mask, return_attention_weights=return_attention_weights
        )
        
        # 4. Reshape back: (batch, seq_len, d_model)
        attn_output = attn_output.view(batch_size, self.num_heads, -1, self.d_k)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # 5. Final linear projection
        output = self.W_o(attn_output)
        
        # Reshape attention weights if needed
        if attn_weights is not None:
            attn_weights = attn_weights.view(batch_size, self.num_heads, -1, attn_weights.size(-1))
        
        return output, attn_weights


# ============================================================================
# Unit Tests
# ============================================================================

def run_tests():
    """Run unit tests for standard attention implementation."""
    print("=" * 60)
    print("Testing Standard Attention Implementation")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 4
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Test 1: ScaledDotProductAttention output shape
    print("\n[Test 1] ScaledDotProductAttention output shape...")
    attention = ScaledDotProductAttention().to(device)
    Q = torch.randn(batch_size, seq_len, d_model // num_heads, device=device)
    K = torch.randn(batch_size, seq_len, d_model // num_heads, device=device)
    V = torch.randn(batch_size, seq_len, d_model // num_heads, device=device)
    
    output, weights = attention(Q, K, V, return_attention_weights=True)
    assert output.shape == (batch_size, seq_len, d_model // num_heads), f"Wrong output shape: {output.shape}"
    assert weights.shape == (batch_size, seq_len, seq_len), f"Wrong weights shape: {weights.shape}"
    print("✅ Passed!")
    
    # Test 2: Attention weights sum to 1
    print("\n[Test 2] Attention weights sum to 1...")
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), "Weights don't sum to 1"
    print("✅ Passed!")
    
    # Test 3: MultiHeadAttention output shape
    print("\n[Test 3] MultiHeadAttention output shape...")
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads).to(device)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    output, weights = mha(x, x, x, return_attention_weights=True)
    assert output.shape == (batch_size, seq_len, d_model), f"Wrong MHA output shape: {output.shape}"
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len), f"Wrong MHA weights shape: {weights.shape}"
    print("✅ Passed!")
    
    # Test 4: Gradient flow
    print("\n[Test 4] Gradient flow...")
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    output, _ = mha(x, x, x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "No gradient computed"
    assert not torch.isnan(x.grad).any(), "NaN in gradients"
    print("✅ Passed!")
    
    # Test 5: Causal mask
    print("\n[Test 5] Causal masking...")
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
    output, weights = attention(Q, K, V, mask=causal_mask, return_attention_weights=True)
    
    # Check upper triangular is zero (masked out)
    upper_tri = torch.triu(weights[0], diagonal=1)
    assert torch.allclose(upper_tri, torch.zeros_like(upper_tri), atol=1e-6), "Causal mask not working"
    print("✅ Passed!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
