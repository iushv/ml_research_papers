"""
Forgetting Attention Implementation (FoX)

This module implements the Forgetting Transformer attention mechanism from:
"Forgetting Transformer: Softmax Attention with a Forget Gate"
by Zhixuan Lin, Evgenii Nikishin, Xu Owen He, Aaron Courville (2025)

Mathematical Foundation:
    
    Standard Attention:
        Attention(Q, K, V) = softmax(QK^T / √d_k) · V
    
    Forgetting Attention:
        ForgetAttention(Q, K, V) = softmax(S ⊙ F) · V
        
    Where:
        S = QK^T / √d_k                    (standard attention scores)
        F = σ(W_f · [q_i; k_j] + b_f)      (data-dependent forget gate)
        
    The forget gate F down-weights attention scores in a data-dependent manner,
    allowing the model to selectively "forget" less relevant past information.

Key Innovation:
    - Data-dependent forgetting (unlike fixed position-based decay)
    - Maintains compatibility with FlashAttention
    - Achieves O(1) memory complexity in recurrent formulation
    - No positional embeddings required
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ForgetGate(nn.Module):
    """
    Data-Dependent Forget Gate.
    
    Computes a gating factor for each query-key pair:
        F[i,j] = σ(W_q · q_i + W_k · k_j + b)
    
    This allows the model to selectively down-weight certain attention
    connections based on the content of the query and key.
    """
    
    def __init__(self, d_k: int, bias: bool = True):
        """
        Args:
            d_k: Dimension of query/key vectors
            bias: Whether to use bias
        """
        super().__init__()
        
        # Separate projections for query and key contributions to forget gate
        # This is more efficient than concatenating [q; k] and doing one projection
        self.W_q = nn.Linear(d_k, 1, bias=False)
        self.W_k = nn.Linear(d_k, 1, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize forget gate weights.
        
        Initialize with small values so initial forget gate is close to 0.5
        (neither fully forgetting nor fully remembering).
        """
        nn.init.normal_(self.W_q.weight, std=0.02)
        nn.init.normal_(self.W_k.weight, std=0.02)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute forget gate values.
        
        Args:
            query: Query tensor of shape (batch, seq_len_q, d_k)
            key: Key tensor of shape (batch, seq_len_k, d_k)
            
        Returns:
            forget_gate: Tensor of shape (batch, seq_len_q, seq_len_k)
                         with values in [0, 1]
        """
        # Compute query contribution: (batch, seq_len_q, 1)
        q_contrib = self.W_q(query)
        
        # Compute key contribution: (batch, seq_len_k, 1)
        k_contrib = self.W_k(key)
        
        # Combine via broadcasting: (batch, seq_len_q, seq_len_k)
        # q_contrib: (batch, seq_len_q, 1) + k_contrib.T: (batch, 1, seq_len_k)
        gate_logits = q_contrib + k_contrib.transpose(-2, -1)
        
        if self.bias is not None:
            gate_logits = gate_logits + self.bias
        
        # Apply sigmoid to get values in [0, 1]
        forget_gate = torch.sigmoid(gate_logits)
        
        return forget_gate


class ForgettingAttention(nn.Module):
    """
    Forgetting Attention mechanism (FoX).
    
    Integrates a data-dependent forget gate into softmax attention:
        ForgetAttention(Q, K, V) = softmax(S ⊙ F) · V
        
    Where S is the standard attention scores and F is the forget gate.
    
    The forget gate allows selective "forgetting" of past tokens,
    enabling better length extrapolation and memory efficiency.
    """
    
    def __init__(self, d_k: int, dropout: float = 0.0):
        """
        Args:
            d_k: Dimension of query/key vectors
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_k = d_k
        self.forget_gate = ForgetGate(d_k)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
        return_forget_gate: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for Forgetting Attention.
        
        Args:
            query: Query tensor of shape (batch, seq_len, d_k)
            key: Key tensor of shape (batch, seq_len, d_k)
            value: Value tensor of shape (batch, seq_len, d_v)
            mask: Optional attention mask (1 = attend, 0 = mask)
            return_attention_weights: Whether to return attention weights
            return_forget_gate: Whether to return forget gate values
            
        Returns:
            output: Attention output of shape (batch, seq_len, d_v)
            attention_weights: Optional attention weights (batch, seq_len, seq_len)
            forget_gate: Optional forget gate values (batch, seq_len, seq_len)
        """
        # Step 1: Compute standard attention scores
        # S = QK^T / √d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Step 2: Compute forget gate
        # F = σ(W_q · q + W_k · k + b)
        forget_values = self.forget_gate(query, key)
        
        # Step 3: Apply forget gate to scores
        # This down-weights scores where forget gate is low
        gated_scores = scores * forget_values
        
        # Step 4: Apply mask if provided
        if mask is not None:
            gated_scores = gated_scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 5: Apply softmax
        attention_weights = F.softmax(gated_scores, dim=-1)
        
        # Handle NaN from all-masked rows
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Step 6: Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Step 7: Apply attention weights to values
        output = torch.matmul(attention_weights, value)
        
        # Prepare optional returns
        attn_out = attention_weights if return_attention_weights else None
        forget_out = forget_values if return_forget_gate else None
        
        return output, attn_out, forget_out


class MultiHeadForgettingAttention(nn.Module):
    """
    Multi-Head Forgetting Attention mechanism.
    
    Extends the standard multi-head attention with forget gates,
    allowing each head to learn different forgetting patterns.
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
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Forgetting Attention for each head
        self.attention = ForgettingAttention(d_k=self.d_k, dropout=dropout)
        
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
        return_attention_weights: bool = False,
        return_forget_gate: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for multi-head forgetting attention.
        
        Args:
            query: Query tensor of shape (batch, seq_len, d_model)
            key: Key tensor of shape (batch, seq_len, d_model)
            value: Value tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention_weights: Whether to return attention weights
            return_forget_gate: Whether to return forget gate values
            
        Returns:
            output: Attention output of shape (batch, seq_len, d_model)
            attention_weights: Optional attention weights
            forget_gate: Optional forget gate values
        """
        batch_size = query.size(0)
        
        # 1. Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. Reshape for multi-head: (batch * num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        Q = Q.contiguous().view(batch_size * self.num_heads, -1, self.d_k)
        K = K.contiguous().view(batch_size * self.num_heads, -1, self.d_k)
        V = V.contiguous().view(batch_size * self.num_heads, -1, self.d_k)
        
        # Handle mask
        if mask is not None:
            # Input mask shape: (batch, seq, seq) or (1, seq, seq)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq, seq)
            mask = mask.expand(batch_size, self.num_heads, -1, -1)
            mask = mask.contiguous().view(batch_size * self.num_heads, mask.size(-2), mask.size(-1))
        
        # 3. Apply forgetting attention
        attn_output, attn_weights, forget_values = self.attention(
            Q, K, V,
            mask=mask,
            return_attention_weights=return_attention_weights,
            return_forget_gate=return_forget_gate
        )
        
        # 4. Reshape back: (batch, seq_len, d_model)
        attn_output = attn_output.view(batch_size, self.num_heads, -1, self.d_k)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # 5. Final linear projection
        output = self.W_o(attn_output)
        
        # Reshape optional outputs
        if attn_weights is not None:
            attn_weights = attn_weights.view(batch_size, self.num_heads, -1, attn_weights.size(-1))
        if forget_values is not None:
            forget_values = forget_values.view(batch_size, self.num_heads, -1, forget_values.size(-1))
        
        return output, attn_weights, forget_values


# ============================================================================
# Unit Tests
# ============================================================================

def run_tests():
    """Run unit tests for Forgetting Attention implementation."""
    print("=" * 60)
    print("Testing Forgetting Attention Implementation (FoX)")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 4
    d_k = d_model // num_heads
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Test 1: ForgetGate output shape and bounds
    print("\n[Test 1] ForgetGate output shape and bounds...")
    forget_gate = ForgetGate(d_k=d_k).to(device)
    q = torch.randn(batch_size, seq_len, d_k, device=device)
    k = torch.randn(batch_size, seq_len, d_k, device=device)
    
    gate_values = forget_gate(q, k)
    assert gate_values.shape == (batch_size, seq_len, seq_len), f"Wrong shape: {gate_values.shape}"
    assert gate_values.min() >= 0, "Forget gate values below 0"
    assert gate_values.max() <= 1, "Forget gate values above 1"
    print(f"   Gate value range: [{gate_values.min():.4f}, {gate_values.max():.4f}]")
    print("✅ Passed!")
    
    # Test 2: ForgettingAttention output shape
    print("\n[Test 2] ForgettingAttention output shape...")
    fox_attention = ForgettingAttention(d_k=d_k).to(device)
    v = torch.randn(batch_size, seq_len, d_k, device=device)
    
    output, weights, forget_out = fox_attention(
        q, k, v,
        return_attention_weights=True,
        return_forget_gate=True
    )
    assert output.shape == (batch_size, seq_len, d_k), f"Wrong output shape: {output.shape}"
    assert weights.shape == (batch_size, seq_len, seq_len), f"Wrong weights shape: {weights.shape}"
    assert forget_out.shape == (batch_size, seq_len, seq_len), f"Wrong forget shape: {forget_out.shape}"
    print("✅ Passed!")
    
    # Test 3: Attention weights sum to 1 (approximately)
    print("\n[Test 3] Attention weights sum to 1...")
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-4), "Weights don't sum to 1"
    print("✅ Passed!")
    
    # Test 4: MultiHeadForgettingAttention output shape
    print("\n[Test 4] MultiHeadForgettingAttention output shape...")
    mhfa = MultiHeadForgettingAttention(d_model=d_model, num_heads=num_heads).to(device)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    output, weights, forget_out = mhfa(
        x, x, x,
        return_attention_weights=True,
        return_forget_gate=True
    )
    assert output.shape == (batch_size, seq_len, d_model), f"Wrong MHFA output shape: {output.shape}"
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len), f"Wrong MHFA weights shape: {weights.shape}"
    assert forget_out.shape == (batch_size, num_heads, seq_len, seq_len), f"Wrong MHFA forget shape: {forget_out.shape}"
    print("✅ Passed!")
    
    # Test 5: Gradient flow
    print("\n[Test 5] Gradient flow...")
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    output, _, _ = mhfa(x, x, x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "No gradient computed"
    assert not torch.isnan(x.grad).any(), "NaN in gradients"
    
    # Check forget gate gradients
    for name, param in mhfa.attention.forget_gate.named_parameters():
        assert param.grad is not None, f"No gradient for forget gate {name}"
        assert not torch.isnan(param.grad).any(), f"NaN in forget gate gradient {name}"
    print("✅ Passed!")
    
    # Test 6: Causal mask
    print("\n[Test 6] Causal masking...")
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
    output, weights, _ = fox_attention(q, k, v, mask=causal_mask, return_attention_weights=True)
    
    upper_tri = torch.triu(weights[0], diagonal=1)
    assert torch.allclose(upper_tri, torch.zeros_like(upper_tri), atol=1e-6), "Causal mask not working"
    print("✅ Passed!")
    
    print("\n[Test 7] Verify forget gate affects output...")
    from standard_attention import ScaledDotProductAttention
    std_attention = ScaledDotProductAttention().to(device)
    
    std_output, _ = std_attention(q, k, v)
    fox_output, _, _ = fox_attention(q, k, v)
    
    # Outputs should be different (forget gate modifies scores)
    diff = (std_output - fox_output).abs().mean()
    assert diff > 1e-5, "Forget gate not affecting output!"
    print(f"   Mean difference from standard attention: {diff:.6f}")
    print("✅ Passed!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
    
    # Bonus: Print forget gate statistics
    print("\n📊 Forget Gate Statistics:")
    print(f"   Mean gate value: {forget_out.mean():.4f}")
    print(f"   Std gate value:  {forget_out.std():.4f}")
    print(f"   Min gate value:  {forget_out.min():.4f}")
    print(f"   Max gate value:  {forget_out.max():.4f}")


if __name__ == "__main__":
    run_tests()
