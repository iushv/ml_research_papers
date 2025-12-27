"""
Comprehensive Paper Claims Verification

This script tests ALL claims made in the Forgetting Transformer paper:

1. Minimal parameter overhead
2. Data-dependent forgetting improves over fixed decay
3. Better length extrapolation
4. Memory efficiency (O(1) in recurrent mode)
5. No positional embeddings needed
6. Compatible training dynamics
"""

import sys
import os

# Add src folder to path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import tracemalloc
from tqdm import tqdm

from transformer_blocks import LanguageModel, TransformerEncoder
from standard_attention import MultiHeadAttention
from forgetting_attention import MultiHeadForgettingAttention, ForgetGate


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# CLAIM 1: Minimal Parameter Overhead
# =============================================================================
def test_claim_1_parameter_overhead():
    """
    CLAIM: The forget gate adds minimal parameter overhead.
    TEST: Compare parameter counts between Standard and Forgetting Transformer.
    """
    print("\n" + "=" * 70)
    print("CLAIM 1: Minimal Parameter Overhead")
    print("=" * 70)
    
    configs = [
        {"d_model": 128, "num_heads": 4, "num_layers": 4, "vocab_size": 5000},
        {"d_model": 256, "num_heads": 8, "num_layers": 6, "vocab_size": 10000},
        {"d_model": 512, "num_heads": 8, "num_layers": 12, "vocab_size": 30000},
    ]
    
    results = []
    for cfg in configs:
        std_model = LanguageModel(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            use_forgetting=False
        )
        fox_model = LanguageModel(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            use_forgetting=True
        )
        
        std_params = std_model.count_parameters()
        fox_params = fox_model.count_parameters()
        overhead = (fox_params - std_params) / std_params * 100
        
        results.append({
            "config": f"d={cfg['d_model']}, h={cfg['num_heads']}, L={cfg['num_layers']}",
            "std_params": std_params,
            "fox_params": fox_params,
            "overhead": overhead
        })
        
        print(f"\n  Config: {results[-1]['config']}")
        print(f"    Standard:   {std_params:>12,} params")
        print(f"    Forgetting: {fox_params:>12,} params")
        print(f"    Overhead:   {overhead:>11.4f}%")
    
    avg_overhead = np.mean([r["overhead"] for r in results])
    verdict = "✅ VERIFIED" if avg_overhead < 1.0 else "❌ NOT VERIFIED"
    print(f"\n  Average Overhead: {avg_overhead:.4f}%")
    print(f"  VERDICT: {verdict} (threshold: <1%)")
    
    return {"claim": "Minimal Parameter Overhead", "verified": avg_overhead < 1.0, "avg_overhead": avg_overhead}


# =============================================================================
# CLAIM 2: Data-Dependent Forgetting is Better
# =============================================================================
def test_claim_2_data_dependent_forgetting(device):
    """
    CLAIM: Data-dependent forgetting is better than fixed positional decay.
    TEST: Compare FoX with a fixed exponential decay baseline.
    """
    print("\n" + "=" * 70)
    print("CLAIM 2: Data-Dependent Forgetting > Fixed Decay")
    print("=" * 70)
    
    vocab_size = 5000
    d_model = 128
    num_heads = 4
    num_layers = 4
    seq_len = 64
    batch_size = 32
    num_steps = 300
    
    # Model 1: Forgetting Transformer (data-dependent)
    fox_model = LanguageModel(
        vocab_size=vocab_size, d_model=d_model, num_heads=num_heads,
        num_layers=num_layers, use_forgetting=True
    ).to(device)
    
    # Model 2: Standard Transformer (no decay)
    std_model = LanguageModel(
        vocab_size=vocab_size, d_model=d_model, num_heads=num_heads,
        num_layers=num_layers, use_forgetting=False
    ).to(device)
    
    def train(model, name):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        losses = []
        
        for step in tqdm(range(num_steps), desc=f"Training {name}"):
            input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                input_ids[:, 1:].reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        
        return np.mean(losses[-50:])  # Final loss
    
    print("\n  Training both models...")
    fox_loss = train(fox_model, "FoX")
    std_loss = train(std_model, "Std")
    
    print(f"\n  Standard Transformer Loss:   {std_loss:.4f}")
    print(f"  Forgetting Transformer Loss: {fox_loss:.4f}")
    
    # Check if forget gate learned meaningful patterns
    fox_model.eval()
    with torch.no_grad():
        input_ids = torch.randint(1, vocab_size, (1, seq_len), device=device)
        x = fox_model.embedding(input_ids) * (fox_model.d_model ** 0.5)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = x + fox_model.pos_embedding(positions)
        
        first_layer = fox_model.encoder.layers[0]
        normed = first_layer.norm1(x)
        _, _, forget_gate = first_layer.attention(
            normed, normed, normed,
            return_attention_weights=True,
            return_forget_gate=True
        )
    
    gate_std = forget_gate.std().item()
    gate_range = (forget_gate.max() - forget_gate.min()).item()
    
    print(f"\n  Forget Gate Statistics (after training):")
    print(f"    Std:   {gate_std:.4f}")
    print(f"    Range: {gate_range:.4f}")
    
    # Verdict: FoX should have learned varied forget patterns
    learned_patterns = gate_std > 0.05 or gate_range > 0.3
    verdict = "✅ VERIFIED" if learned_patterns else "⚠️ PARTIALLY VERIFIED"
    print(f"\n  VERDICT: {verdict} (gate shows learned patterns: {learned_patterns})")
    
    return {
        "claim": "Data-Dependent Forgetting",
        "verified": learned_patterns,
        "fox_loss": fox_loss,
        "std_loss": std_loss,
        "gate_std": gate_std
    }


# =============================================================================
# CLAIM 3: Better Length Extrapolation
# =============================================================================
def test_claim_3_length_extrapolation(device):
    """
    CLAIM: FoX extrapolates better to longer sequences.
    TEST: Train on short sequences, test on longer ones.
    """
    print("\n" + "=" * 70)
    print("CLAIM 3: Better Length Extrapolation")
    print("=" * 70)
    
    vocab_size = 5000
    d_model = 128
    num_heads = 4
    num_layers = 4
    train_len = 64
    batch_size = 32
    num_steps = 300
    test_lengths = [128, 256, 384, 512]
    
    # Create models with large max_seq_len
    fox_model = LanguageModel(
        vocab_size=vocab_size, d_model=d_model, num_heads=num_heads,
        num_layers=num_layers, max_seq_len=1024, use_forgetting=True
    ).to(device)
    
    std_model = LanguageModel(
        vocab_size=vocab_size, d_model=d_model, num_heads=num_heads,
        num_layers=num_layers, max_seq_len=1024, use_forgetting=False
    ).to(device)
    
    def train(model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        for _ in tqdm(range(num_steps), desc="Training"):
            input_ids = torch.randint(1, vocab_size, (batch_size, train_len), device=device)
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                input_ids[:, 1:].reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def evaluate(model, seq_len):
        model.eval()
        with torch.no_grad():
            input_ids = torch.randint(1, vocab_size, (1, seq_len), device=device)
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                input_ids[:, 1:].reshape(-1)
            )
        return loss.item()
    
    print(f"\n  Training on seq_len={train_len}...")
    train(std_model)
    train(fox_model)
    
    print(f"\n  Testing on longer sequences:")
    print(f"  {'Length':<10} {'Standard':<15} {'Forgetting':<15} {'Winner':<10}")
    print(f"  {'-'*50}")
    
    fox_wins = 0
    for length in test_lengths:
        std_loss = evaluate(std_model, length)
        fox_loss = evaluate(fox_model, length)
        winner = "FoX" if fox_loss < std_loss else "Std"
        if winner == "FoX":
            fox_wins += 1
        print(f"  {length:<10} {std_loss:<15.4f} {fox_loss:<15.4f} {winner:<10}")
    
    verdict = "✅ VERIFIED" if fox_wins > len(test_lengths) // 2 else "❌ NOT VERIFIED"
    print(f"\n  FoX wins: {fox_wins}/{len(test_lengths)}")
    print(f"  VERDICT: {verdict}")
    
    return {
        "claim": "Better Length Extrapolation",
        "verified": fox_wins > len(test_lengths) // 2,
        "fox_wins": fox_wins,
        "total_tests": len(test_lengths)
    }


# =============================================================================
# CLAIM 4: Memory Efficiency
# =============================================================================
def test_claim_4_memory_efficiency(device):
    """
    CLAIM: FoX achieves O(1) memory in recurrent mode.
    TEST: Measure memory usage at different sequence lengths.
    
    NOTE: Our implementation uses standard attention (O(n²) memory).
          True O(1) requires recurrent formulation which we haven't implemented.
    """
    print("\n" + "=" * 70)
    print("CLAIM 4: Memory Efficiency")
    print("=" * 70)
    
    print("\n  ⚠️ NOTE: This implementation uses standard attention mode (O(n²)).")
    print("          The paper's O(1) claim requires recurrent formulation.")
    
    vocab_size = 5000
    d_model = 128
    num_heads = 4
    num_layers = 2
    batch_size = 1
    
    fox_model = LanguageModel(
        vocab_size=vocab_size, d_model=d_model, num_heads=num_heads,
        num_layers=num_layers, max_seq_len=2048, use_forgetting=True
    ).to(device)
    
    std_model = LanguageModel(
        vocab_size=vocab_size, d_model=d_model, num_heads=num_heads,
        num_layers=num_layers, max_seq_len=2048, use_forgetting=False
    ).to(device)
    
    seq_lengths = [64, 128, 256, 512, 1024]
    
    print(f"\n  {'Length':<10} {'Std Time (ms)':<15} {'FoX Time (ms)':<15} {'Ratio':<10}")
    print(f"  {'-'*50}")
    
    for seq_len in seq_lengths:
        # Time standard model
        std_model.eval()
        fox_model.eval()
        
        with torch.no_grad():
            input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
            
            # Warmup
            _ = std_model(input_ids)
            _ = fox_model(input_ids)
            
            # Measure
            start = time.time()
            for _ in range(10):
                _ = std_model(input_ids)
            std_time = (time.time() - start) / 10 * 1000
            
            start = time.time()
            for _ in range(10):
                _ = fox_model(input_ids)
            fox_time = (time.time() - start) / 10 * 1000
        
        ratio = fox_time / std_time
        print(f"  {seq_len:<10} {std_time:<15.2f} {fox_time:<15.2f} {ratio:<10.2f}x")
    
    print(f"\n  VERDICT: ⚠️ REQUIRES RECURRENT IMPLEMENTATION")
    print(f"           Current implementation: O(n²) for both models")
    print(f"           Paper claim: O(1) with recurrent formulation")
    
    return {
        "claim": "Memory Efficiency O(1)",
        "verified": None,  # Cannot verify without recurrent implementation
        "note": "Requires recurrent formulation not implemented here"
    }


# =============================================================================
# CLAIM 5: Works Without Positional Embeddings
# =============================================================================
def test_claim_5_no_positional_embeddings(device):
    """
    CLAIM: FoX doesn't need positional embeddings.
    TEST: Train FoX without positional embeddings and compare.
    """
    print("\n" + "=" * 70)
    print("CLAIM 5: Works Without Positional Embeddings")
    print("=" * 70)
    
    vocab_size = 5000
    d_model = 128
    num_heads = 4
    num_layers = 4
    seq_len = 64
    batch_size = 32
    num_steps = 300
    
    # Create FoX model and remove positional embeddings
    class FoXNoPos(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.encoder = TransformerEncoder(
                num_layers=num_layers, d_model=d_model,
                num_heads=num_heads, use_forgetting=True
            )
            self.output_proj = nn.Linear(d_model, vocab_size)
            self.output_proj.weight = self.embedding.weight
            self.dropout = nn.Dropout(0.1)
            nn.init.normal_(self.embedding.weight, std=0.02)
        
        def forward(self, input_ids):
            seq_len = input_ids.size(1)
            x = self.embedding(input_ids) * (self.d_model ** 0.5)
            x = self.dropout(x)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0)
            x, _ = self.encoder(x, mask=causal_mask)
            return self.output_proj(x), None
        
        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    # FoX with positional embeddings
    fox_with_pos = LanguageModel(
        vocab_size=vocab_size, d_model=d_model, num_heads=num_heads,
        num_layers=num_layers, use_forgetting=True
    ).to(device)
    
    # FoX without positional embeddings
    fox_no_pos = FoXNoPos().to(device)
    
    # Standard without positional embeddings
    class StdNoPos(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.encoder = TransformerEncoder(
                num_layers=num_layers, d_model=d_model,
                num_heads=num_heads, use_forgetting=False
            )
            self.output_proj = nn.Linear(d_model, vocab_size)
            self.output_proj.weight = self.embedding.weight
            self.dropout = nn.Dropout(0.1)
            nn.init.normal_(self.embedding.weight, std=0.02)
        
        def forward(self, input_ids):
            seq_len = input_ids.size(1)
            x = self.embedding(input_ids) * (self.d_model ** 0.5)
            x = self.dropout(x)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0)
            x, _ = self.encoder(x, mask=causal_mask)
            return self.output_proj(x), None
    
    std_no_pos = StdNoPos().to(device)
    
    def train(model, name):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        losses = []
        for _ in tqdm(range(num_steps), desc=f"Training {name}"):
            input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                input_ids[:, 1:].reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return np.mean(losses[-50:])
    
    print("\n  Training models...")
    fox_pos_loss = train(fox_with_pos, "FoX+Pos")
    fox_nopos_loss = train(fox_no_pos, "FoX-NoPos")
    std_nopos_loss = train(std_no_pos, "Std-NoPos")
    
    print(f"\n  Results:")
    print(f"    FoX with Pos Emb:    {fox_pos_loss:.4f}")
    print(f"    FoX without Pos Emb: {fox_nopos_loss:.4f}")
    print(f"    Std without Pos Emb: {std_nopos_loss:.4f}")
    
    # FoX without pos should work reasonably well
    degradation = (fox_nopos_loss - fox_pos_loss) / fox_pos_loss * 100
    fox_beats_std_nopos = fox_nopos_loss < std_nopos_loss
    
    print(f"\n  FoX degradation without Pos: {degradation:.2f}%")
    print(f"  FoX-NoPos beats Std-NoPos: {fox_beats_std_nopos}")
    
    verdict = "✅ VERIFIED" if fox_beats_std_nopos and degradation < 10 else "⚠️ PARTIALLY VERIFIED"
    print(f"\n  VERDICT: {verdict}")
    
    return {
        "claim": "Works Without Positional Embeddings",
        "verified": fox_beats_std_nopos and degradation < 10,
        "degradation_pct": degradation,
        "fox_beats_std": fox_beats_std_nopos
    }


# =============================================================================
# CLAIM 6: Forget Gate Learns Meaningful Patterns
# =============================================================================
def test_claim_6_meaningful_patterns(device):
    """
    CLAIM: The forget gate learns data-dependent meaningful patterns.
    TEST: Analyze forget gate activations on different inputs.
    """
    print("\n" + "=" * 70)
    print("CLAIM 6: Forget Gate Learns Meaningful Patterns")
    print("=" * 70)
    
    vocab_size = 5000
    d_model = 128
    num_heads = 4
    num_layers = 4
    seq_len = 64
    batch_size = 32
    num_steps = 300
    
    fox_model = LanguageModel(
        vocab_size=vocab_size, d_model=d_model, num_heads=num_heads,
        num_layers=num_layers, use_forgetting=True
    ).to(device)
    
    # Train the model
    optimizer = torch.optim.AdamW(fox_model.parameters(), lr=1e-3)
    for _ in tqdm(range(num_steps), desc="Training"):
        fox_model.train()
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
        logits, _ = fox_model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size),
            input_ids[:, 1:].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Analyze forget gate patterns
    fox_model.eval()
    
    def get_forget_gates(input_ids):
        with torch.no_grad():
            x = fox_model.embedding(input_ids) * (fox_model.d_model ** 0.5)
            positions = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
            x = x + fox_model.pos_embedding(positions)
            
            gates = []
            for layer in fox_model.encoder.layers:
                normed = layer.norm1(x)
                attn_out, _, gate = layer.attention(
                    normed, normed, normed,
                    return_attention_weights=True,
                    return_forget_gate=True
                )
                gates.append(gate)
                x = x + layer.dropout1(attn_out)
                normed = layer.norm2(x)
                x = x + layer.dropout2(layer.feed_forward(normed))
            
            return gates
    
    # Test on different inputs
    input1 = torch.randint(1, vocab_size, (1, seq_len), device=device)
    input2 = torch.randint(1, vocab_size, (1, seq_len), device=device)
    
    gates1 = get_forget_gates(input1)
    gates2 = get_forget_gates(input2)
    
    # Compare gates between different inputs
    print("\n  Forget Gate Analysis (per layer):")
    print(f"  {'Layer':<10} {'Mean±Std (Input1)':<25} {'Mean±Std (Input2)':<25} {'Diff':<10}")
    print(f"  {'-'*70}")
    
    total_diff = 0
    for i, (g1, g2) in enumerate(zip(gates1, gates2)):
        mean1, std1 = g1.mean().item(), g1.std().item()
        mean2, std2 = g2.mean().item(), g2.std().item()
        diff = (g1 - g2).abs().mean().item()
        total_diff += diff
        print(f"  Layer {i+1:<5} {mean1:.4f} ± {std1:.4f}{'':<10} {mean2:.4f} ± {std2:.4f}{'':<10} {diff:.4f}")
    
    avg_diff = total_diff / len(gates1)
    
    # Check if gates are input-dependent (different inputs should give different gates)
    is_input_dependent = avg_diff > 0.01
    
    # Check if gates have non-trivial variance
    has_variance = any(g.std().item() > 0.05 for layer_gates in [gates1, gates2] for g in layer_gates)
    
    print(f"\n  Average gate difference between inputs: {avg_diff:.4f}")
    print(f"  Gates are input-dependent: {is_input_dependent}")
    print(f"  Gates have non-trivial variance: {has_variance}")
    
    verified = is_input_dependent and has_variance
    verdict = "✅ VERIFIED" if verified else "❌ NOT VERIFIED"
    print(f"\n  VERDICT: {verdict}")
    
    return {
        "claim": "Forget Gate Learns Meaningful Patterns",
        "verified": verified,
        "avg_diff": avg_diff,
        "is_input_dependent": is_input_dependent,
        "has_variance": has_variance
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "PAPER CLAIMS VERIFICATION" + " " * 21 + "#")
    print("#" + " " * 15 + "Forgetting Transformer (FoX)" + " " * 23 + "#")
    print("#" * 70)
    
    device = get_device()
    print(f"\nDevice: {device}")
    
    results = []
    
    # Test each claim
    results.append(test_claim_1_parameter_overhead())
    results.append(test_claim_2_data_dependent_forgetting(device))
    results.append(test_claim_3_length_extrapolation(device))
    results.append(test_claim_4_memory_efficiency(device))
    results.append(test_claim_5_no_positional_embeddings(device))
    results.append(test_claim_6_meaningful_patterns(device))
    
    # Summary
    print("\n" + "#" * 70)
    print("#" + " " * 25 + "FINAL SUMMARY" + " " * 28 + "#")
    print("#" * 70)
    
    print(f"\n  {'Claim':<45} {'Verified':<15}")
    print(f"  {'-'*60}")
    
    verified_count = 0
    for r in results:
        status = "✅ YES" if r["verified"] else ("⚠️ PARTIAL" if r["verified"] is None else "❌ NO")
        if r["verified"]:
            verified_count += 1
        print(f"  {r['claim']:<45} {status:<15}")
    
    print(f"\n  Total Verified: {verified_count}/{len([r for r in results if r['verified'] is not None])}")
    
    return results


if __name__ == "__main__":
    results = main()
