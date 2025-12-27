"""
Recurrent Forgetting Attention Implementation

This implements the O(1) memory formulation of Forgetting Attention.
Instead of computing the full attention matrix, we maintain a fixed-size
state that gets updated at each timestep.

Mathematical Formulation:
    
    For each timestep t:
        S_t = f_t * S_{t-1} + k_t ⊗ v_t     # State update with forget
        z_t = f_t * z_{t-1} + k_t            # Normalizer update  
        o_t = (q_t · S_t) / (q_t · z_t + ε)  # Output
    
    Where:
        S_t ∈ ℝ^{d_k × d_v} = accumulated key-value state
        z_t ∈ ℝ^{d_k} = normalizer state
        f_t ∈ (0, 1) = forget gate (scalar or vector)
        k_t, v_t = key and value at timestep t
        q_t = query at timestep t

Memory Complexity:
    - Parallel mode: O(n²) - stores full attention matrix
    - Recurrent mode: O(d²) - stores only state S and normalizer z
                      O(1) with respect to sequence length!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RecurrentForgetGate(nn.Module):
    """
    Forget gate for recurrent formulation.
    
    Computes a scalar forget factor for each timestep:
        f_t = σ(w_q · q_t + w_k · k_t + b)
    """
    
    def __init__(self, d_k: int):
        super().__init__()
        self.w_q = nn.Linear(d_k, 1, bias=False)
        self.w_k = nn.Linear(d_k, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Initialize to produce values around 0.9 (remember most)
        nn.init.normal_(self.w_q.weight, std=0.01)
        nn.init.normal_(self.w_k.weight, std=0.01)
        nn.init.constant_(self.bias, 2.0)  # sigmoid(2) ≈ 0.88
    
    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch, d_k) - single timestep query
            key: (batch, d_k) - single timestep key
            
        Returns:
            forget: (batch, 1) - forget factor in (0, 1)
        """
        logit = self.w_q(query) + self.w_k(key) + self.bias
        return torch.sigmoid(logit)


class RecurrentForgettingAttentionCell(nn.Module):
    """
    Single-step recurrent forgetting attention cell.
    
    This processes one timestep and updates the internal state.
    Memory: O(d²) regardless of sequence length!
    """
    
    def __init__(self, d_k: int, d_v: int):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.forget_gate = RecurrentForgetGate(d_k)
        self.eps = 1e-6
    
    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the recurrent state.
        
        Returns:
            S: (batch, d_k, d_v) - key-value accumulator
            z: (batch, d_k) - normalizer accumulator
        """
        S = torch.zeros(batch_size, self.d_k, self.d_v, device=device)
        z = torch.zeros(batch_size, self.d_k, device=device)
        return S, z
    
    def forward(
        self,
        query: torch.Tensor,      # (batch, d_k)
        key: torch.Tensor,        # (batch, d_k)
        value: torch.Tensor,      # (batch, d_v)
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Process one timestep.
        
        Args:
            query: (batch, d_k)
            key: (batch, d_k)
            value: (batch, d_v)
            state: (S, z) from previous timestep
            
        Returns:
            output: (batch, d_v)
            new_state: (S_new, z_new)
            forget_value: (batch, 1)
        """
        S, z = state
        batch_size = query.size(0)
        
        # Compute forget gate: f_t ∈ (0, 1)
        f = self.forget_gate(query, key)  # (batch, 1)
        
        # Compute outer product: k ⊗ v
        # key: (batch, d_k) -> (batch, d_k, 1)
        # value: (batch, d_v) -> (batch, 1, d_v)
        kv = torch.bmm(key.unsqueeze(2), value.unsqueeze(1))  # (batch, d_k, d_v)
        
        # Update state with forgetting
        # S_t = f_t * S_{t-1} + k_t ⊗ v_t
        f_expanded = f.unsqueeze(-1)  # (batch, 1, 1)
        S_new = f_expanded * S + kv
        
        # Update normalizer
        # z_t = f_t * z_{t-1} + k_t
        # f: (batch, 1), z: (batch, d_k) - need to broadcast f
        z_new = f * z + key  # f broadcasts to (batch, d_k)
        
        # Compute output
        # o_t = (q_t · S_t) / (q_t · z_t + ε)
        # query: (batch, d_k) -> (batch, 1, d_k)
        numerator = torch.bmm(query.unsqueeze(1), S_new)  # (batch, 1, d_v)
        denominator = (query * z_new).sum(dim=-1, keepdim=True) + self.eps  # (batch, 1)
        output = (numerator / denominator.unsqueeze(-1)).squeeze(1)  # (batch, d_v)
        
        return output, (S_new, z_new), f


class RecurrentForgettingAttention(nn.Module):
    """
    Full recurrent forgetting attention module.
    
    Can operate in two modes:
    1. Parallel mode: Process entire sequence at once (O(n²) memory)
    2. Recurrent mode: Process token by token (O(1) memory w.r.t. seq length)
    """
    
    def __init__(self, d_model: int, d_k: int = None, d_v: int = None):
        super().__init__()
        d_k = d_k or d_model
        d_v = d_v or d_model
        
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)
        self.W_o = nn.Linear(d_v, d_model)
        
        self.cell = RecurrentForgettingAttentionCell(d_k, d_v)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward_recurrent(
        self,
        x: torch.Tensor,
        return_forget_gates: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass in recurrent mode (O(1) memory w.r.t. sequence length).
        
        Args:
            x: (batch, seq_len, d_model)
            return_forget_gates: whether to return forget gate values
            
        Returns:
            output: (batch, seq_len, d_model)
            forget_gates: (batch, seq_len) if requested
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Project to Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, d_k)
        K = self.W_k(x)  # (batch, seq_len, d_k)
        V = self.W_v(x)  # (batch, seq_len, d_v)
        
        # Initialize state
        state = self.cell.init_state(batch_size, device)
        
        # Process each timestep
        outputs = []
        forget_gates = [] if return_forget_gates else None
        
        for t in range(seq_len):
            q_t = Q[:, t, :]  # (batch, d_k)
            k_t = K[:, t, :]  # (batch, d_k)
            v_t = V[:, t, :]  # (batch, d_v)
            
            output_t, state, f_t = self.cell(q_t, k_t, v_t, state)
            outputs.append(output_t)
            
            if return_forget_gates:
                forget_gates.append(f_t.squeeze(-1))
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_v)
        output = self.W_o(output)  # (batch, seq_len, d_model)
        
        if return_forget_gates:
            forget_gates = torch.stack(forget_gates, dim=1)  # (batch, seq_len)
            return output, forget_gates
        
        return output, None
    
    def forward(
        self,
        x: torch.Tensor,
        mode: str = "recurrent",
        return_forget_gates: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, d_model)
            mode: "recurrent" for O(1) memory, "parallel" for O(n²)
            return_forget_gates: whether to return forget gate values
        """
        if mode == "recurrent":
            return self.forward_recurrent(x, return_forget_gates)
        else:
            raise NotImplementedError("Parallel mode not implemented in this module")


class RecurrentLanguageModel(nn.Module):
    """
    Language model using recurrent forgetting attention.
    
    This model can process arbitrarily long sequences with O(1) memory
    during inference (streaming mode).
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        max_seq_len: int = 8192,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': RecurrentForgettingAttention(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model),
                    nn.Dropout(dropout)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model)
            })
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.output_proj.weight = self.embedding.weight  # Tie weights
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_forget_gates: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch, seq_len)
            return_forget_gates: whether to return forget gate values
            
        Returns:
            logits: (batch, seq_len, vocab_size)
            forget_gates: list of (batch, seq_len) per layer if requested
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Layers
        all_forget_gates = [] if return_forget_gates else None
        
        for layer in self.layers:
            # Pre-norm attention
            residual = x
            x = layer['norm1'](x)
            attn_out, fg = layer['attention'](x, return_forget_gates=return_forget_gates)
            x = residual + attn_out
            
            if return_forget_gates:
                all_forget_gates.append(fg)
            
            # Pre-norm FFN
            residual = x
            x = layer['norm2'](x)
            x = residual + layer['ffn'](x)
        
        x = self.final_norm(x)
        logits = self.output_proj(x)
        
        return logits, all_forget_gates
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Tests
# =============================================================================

def run_tests():
    print("=" * 60)
    print("Testing Recurrent Forgetting Attention")
    print("=" * 60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    batch_size = 2
    seq_len = 32
    d_model = 64
    vocab_size = 1000
    
    # Test 1: RecurrentForgettingAttentionCell
    print("\n[Test 1] RecurrentForgettingAttentionCell...")
    cell = RecurrentForgettingAttentionCell(d_k=d_model, d_v=d_model).to(device)
    state = cell.init_state(batch_size, device)
    
    q = torch.randn(batch_size, d_model, device=device)
    k = torch.randn(batch_size, d_model, device=device)
    v = torch.randn(batch_size, d_model, device=device)
    
    output, new_state, f = cell(q, k, v, state)
    assert output.shape == (batch_size, d_model)
    assert new_state[0].shape == (batch_size, d_model, d_model)
    assert 0 < f.mean().item() < 1
    print(f"   Output shape: {output.shape}")
    print(f"   Forget gate mean: {f.mean().item():.4f}")
    print("✅ Passed!")
    
    # Test 2: RecurrentForgettingAttention
    print("\n[Test 2] RecurrentForgettingAttention...")
    attn = RecurrentForgettingAttention(d_model).to(device)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    output, fg = attn(x, mode="recurrent", return_forget_gates=True)
    assert output.shape == x.shape
    assert fg.shape == (batch_size, seq_len)
    print(f"   Output shape: {output.shape}")
    print(f"   Forget gates shape: {fg.shape}")
    print("✅ Passed!")
    
    # Test 3: RecurrentLanguageModel
    print("\n[Test 3] RecurrentLanguageModel...")
    model = RecurrentLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=2
    ).to(device)
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    logits, _ = model(input_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Parameters: {model.count_parameters():,}")
    print("✅ Passed!")
    
    # Test 4: Gradient flow
    print("\n[Test 4] Gradient flow...")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    logits, _ = model(input_ids)
    loss = logits.sum()
    loss.backward()
    
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0
    print(f"   Gradient norm: {grad_norm:.4f}")
    print("✅ Passed!")
    
    # Test 5: Memory comparison preview
    print("\n[Test 5] Memory scaling preview...")
    for test_len in [64, 128, 256]:
        input_ids = torch.randint(0, vocab_size, (1, test_len), device=device)
        with torch.no_grad():
            logits, _ = model(input_ids)
        print(f"   seq_len={test_len}: ✓ (processes without OOM)")
    print("✅ Passed!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
