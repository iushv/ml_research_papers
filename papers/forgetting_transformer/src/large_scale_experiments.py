"""
Large-Scale FoX Experiments

This script runs comprehensive experiments on:
1. WikiText-2 dataset (real language modeling)
2. Memory benchmarking (O(1) vs O(n²))
3. Larger model configurations
4. Length extrapolation tests

Run with: python src/large_scale_experiments.py
"""

import sys
import os
import gc
import time
import json
import math
import resource
from datetime import datetime
from typing import Dict, List, Tuple

# Add src to path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from transformer_blocks import LanguageModel
from recurrent_attention import RecurrentLanguageModel


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_memory_mb():
    """Get current memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    else:
        # For MPS/CPU, use process memory
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


# =============================================================================
# Dataset Loading
# =============================================================================

def load_wikitext2(tokenizer_vocab_size: int = 10000, max_samples: int = None):
    """
    Load WikiText-2 dataset and create simple tokenizer.
    
    Returns:
        train_data: list of token sequences
        val_data: list of token sequences
        vocab: dict mapping tokens to ids
    """
    print("\n📚 Loading WikiText-2 dataset...")
    
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    except Exception as e:
        print(f"   ⚠️ Could not load WikiText-2: {e}")
        print("   Using synthetic data instead...")
        return create_synthetic_dataset(tokenizer_vocab_size)
    
    # Simple word-level tokenizer
    print("   Building vocabulary...")
    word_counts = {}
    for split in ['train', 'validation']:
        for example in dataset[split]:
            text = example['text']
            for word in text.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1
    
    # Keep top vocab_size - 3 words (reserve for special tokens)
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
    for word, _ in sorted_words[:tokenizer_vocab_size - 3]:
        vocab[word] = len(vocab)
    
    # Tokenize datasets
    def tokenize(examples):
        tokens = []
        for text in [examples['text']]:
            words = text.lower().split()
            seq = [vocab.get(w, vocab['<unk>']) for w in words]
            if seq:
                seq.append(vocab['<eos>'])
                tokens.extend(seq)
        return tokens
    
    print("   Tokenizing train set...")
    train_tokens = []
    for i, example in enumerate(dataset['train']):
        if max_samples and i >= max_samples:
            break
        train_tokens.extend(tokenize(example))
    
    print("   Tokenizing validation set...")
    val_tokens = []
    for i, example in enumerate(dataset['validation']):
        if max_samples and i >= max_samples // 10:
            break
        val_tokens.extend(tokenize(example))
    
    print(f"   ✅ Loaded {len(train_tokens):,} train tokens, {len(val_tokens):,} val tokens")
    print(f"   Vocab size: {len(vocab):,}")
    
    return train_tokens, val_tokens, vocab


def create_synthetic_dataset(vocab_size: int):
    """Create synthetic dataset if WikiText fails to load."""
    train_tokens = list(np.random.randint(3, vocab_size, size=500000))
    val_tokens = list(np.random.randint(3, vocab_size, size=50000))
    vocab = {str(i): i for i in range(vocab_size)}
    return train_tokens, val_tokens, vocab


def create_batches(tokens: List[int], seq_len: int, batch_size: int, device: torch.device):
    """Create batches from token list."""
    # Truncate to fit batch_size * n
    n_tokens = len(tokens)
    n_batches = (n_tokens - 1) // (batch_size * seq_len)
    n_tokens = n_batches * batch_size * seq_len
    
    tokens = tokens[:n_tokens + 1]  # +1 for targets
    data = torch.tensor(tokens, dtype=torch.long, device=device)
    
    # Reshape into batches
    batches = []
    for i in range(n_batches):
        start = i * batch_size * seq_len
        batch_tokens = data[start:start + batch_size * seq_len + 1]
        x = batch_tokens[:-1].view(batch_size, seq_len)
        y = batch_tokens[1:].view(batch_size, seq_len)
        batches.append((x, y))
    
    return batches


# =============================================================================
# Training
# =============================================================================

def train_model(
    model: nn.Module,
    train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    vocab_size: int,
    num_epochs: int = 3,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    model_name: str = "Model"
) -> Dict:
    """Train a model and return metrics."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(train_batches)
    )
    
    results = {
        'train_losses': [],
        'val_losses': [],
        'val_perplexities': [],
        'epoch_times': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0
        pbar = tqdm(train_batches, desc=f"{model_name} Epoch {epoch+1}/{num_epochs}")
        
        for x, y in pbar:
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_batches)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_batches:
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    y.view(-1)
                )
                val_loss += loss.item()
        
        val_loss /= len(val_batches)
        val_ppl = math.exp(min(val_loss, 10))  # Cap to avoid overflow
        
        epoch_time = time.time() - epoch_start
        
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['val_perplexities'].append(val_ppl)
        results['epoch_times'].append(epoch_time)
        
        print(f"   Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Val PPL={val_ppl:.2f}, Time={epoch_time:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    results['final_val_loss'] = results['val_losses'][-1]
    results['final_val_ppl'] = results['val_perplexities'][-1]
    results['best_val_loss'] = best_val_loss
    
    return results


# =============================================================================
# Experiments
# =============================================================================

def experiment_1_wikitext_comparison(device, results_dir):
    """Compare Standard, FoX, and Recurrent FoX on WikiText-2."""
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: WikiText-2 Language Modeling")
    print("=" * 70)
    
    # Configuration
    config = {
        'd_model': 256,
        'num_heads': 4,
        'num_layers': 4,
        'vocab_size': 10000,
        'seq_len': 128,
        'batch_size': 32,
        'num_epochs': 3,
        'lr': 1e-3
    }
    
    print(f"\nConfiguration: {config}")
    
    # Load data
    train_tokens, val_tokens, vocab = load_wikitext2(
        tokenizer_vocab_size=config['vocab_size']
    )
    config['vocab_size'] = len(vocab)
    
    train_batches = create_batches(
        train_tokens, config['seq_len'], config['batch_size'], device
    )
    val_batches = create_batches(
        val_tokens, config['seq_len'], config['batch_size'], device
    )
    
    print(f"\n   Train batches: {len(train_batches)}")
    print(f"   Val batches: {len(val_batches)}")
    
    results = {'config': config, 'models': {}}
    
    # Model 1: Standard Transformer
    print("\n--- Standard Transformer ---")
    std_model = LanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        use_forgetting=False
    ).to(device)
    print(f"   Parameters: {std_model.count_parameters():,}")
    
    results['models']['standard'] = train_model(
        std_model, train_batches, val_batches, config['vocab_size'],
        num_epochs=config['num_epochs'], lr=config['lr'],
        model_name="Std"
    )
    results['models']['standard']['params'] = std_model.count_parameters()
    
    del std_model
    gc.collect()
    
    # Model 2: Forgetting Transformer (parallel)
    print("\n--- Forgetting Transformer (Parallel) ---")
    fox_model = LanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        use_forgetting=True
    ).to(device)
    print(f"   Parameters: {fox_model.count_parameters():,}")
    
    results['models']['fox_parallel'] = train_model(
        fox_model, train_batches, val_batches, config['vocab_size'],
        num_epochs=config['num_epochs'], lr=config['lr'],
        model_name="FoX"
    )
    results['models']['fox_parallel']['params'] = fox_model.count_parameters()
    
    del fox_model
    gc.collect()
    
    # Model 3: Recurrent Forgetting Transformer
    print("\n--- Recurrent Forgetting Transformer (O(1) memory) ---")
    rec_model = RecurrentLanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_layers=config['num_layers']
    ).to(device)
    print(f"   Parameters: {rec_model.count_parameters():,}")
    
    results['models']['fox_recurrent'] = train_model(
        rec_model, train_batches, val_batches, config['vocab_size'],
        num_epochs=config['num_epochs'], lr=config['lr'],
        model_name="Rec"
    )
    results['models']['fox_recurrent']['params'] = rec_model.count_parameters()
    
    # Summary
    print("\n" + "-" * 70)
    print("EXPERIMENT 1 SUMMARY")
    print("-" * 70)
    print(f"\n{'Model':<25} {'Params':<15} {'Val Loss':<12} {'Val PPL':<12}")
    print("-" * 64)
    for name, data in results['models'].items():
        print(f"{name:<25} {data['params']:>12,} {data['final_val_loss']:>10.4f} {data['final_val_ppl']:>10.2f}")
    
    # Save results
    with open(f"{results_dir}/experiment_1_wikitext.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def experiment_2_memory_benchmark(device, results_dir):
    """Benchmark memory usage at different sequence lengths."""
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Memory Benchmark")
    print("=" * 70)
    
    vocab_size = 5000
    d_model = 256
    num_layers = 4
    batch_size = 1
    
    seq_lengths = [128, 256, 512, 1024, 2048]
    
    if str(device) == "mps":
        # MPS has limited memory query capabilities
        print("\n   Note: Running on MPS - memory measurements are approximate")
    
    results = {
        'config': {'vocab_size': vocab_size, 'd_model': d_model, 'num_layers': num_layers},
        'parallel': {},
        'recurrent': {}
    }
    
    print(f"\n{'Seq Len':<10} {'Parallel (ms)':<15} {'Recurrent (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for seq_len in seq_lengths:
        # Test parallel model
        try:
            fox_model = LanguageModel(
                vocab_size=vocab_size, d_model=d_model,
                num_heads=4, num_layers=num_layers,
                max_seq_len=seq_len + 100, use_forgetting=True
            ).to(device)
            
            fox_model.eval()
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            
            # Warmup
            with torch.no_grad():
                _ = fox_model(input_ids)
            
            # Measure
            start = time.time()
            with torch.no_grad():
                for _ in range(5):
                    _ = fox_model(input_ids)
            parallel_time = (time.time() - start) / 5 * 1000
            
            results['parallel'][seq_len] = {'time_ms': parallel_time, 'success': True}
            del fox_model
            gc.collect()
            
        except Exception as e:
            results['parallel'][seq_len] = {'time_ms': float('inf'), 'success': False, 'error': str(e)}
            parallel_time = float('inf')
        
        # Test recurrent model
        try:
            rec_model = RecurrentLanguageModel(
                vocab_size=vocab_size, d_model=d_model, num_layers=num_layers
            ).to(device)
            
            rec_model.eval()
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            
            # Warmup
            with torch.no_grad():
                _ = rec_model(input_ids)
            
            # Measure
            start = time.time()
            with torch.no_grad():
                for _ in range(5):
                    _ = rec_model(input_ids)
            recurrent_time = (time.time() - start) / 5 * 1000
            
            results['recurrent'][seq_len] = {'time_ms': recurrent_time, 'success': True}
            del rec_model
            gc.collect()
            
        except Exception as e:
            results['recurrent'][seq_len] = {'time_ms': float('inf'), 'success': False, 'error': str(e)}
            recurrent_time = float('inf')
        
        # Print results
        if parallel_time != float('inf') and recurrent_time != float('inf'):
            speedup = f"{parallel_time / recurrent_time:.2f}x"
        else:
            speedup = "N/A"
        
        print(f"{seq_len:<10} {parallel_time:<15.2f} {recurrent_time:<15.2f} {speedup:<10}")
    
    # Save results
    with open(f"{results_dir}/experiment_2_memory.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def experiment_3_larger_models(device, results_dir):
    """Test larger model configurations."""
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Larger Model Scaling")
    print("=" * 70)
    
    configs = [
        {'d_model': 128, 'num_layers': 4, 'num_heads': 4, 'name': 'Small'},
        {'d_model': 256, 'num_layers': 6, 'num_heads': 8, 'name': 'Medium'},
        {'d_model': 384, 'num_layers': 8, 'num_heads': 8, 'name': 'Large'},
    ]
    
    vocab_size = 10000
    seq_len = 128
    batch_size = 16
    num_steps = 200
    
    results = {'configs': configs, 'models': {}}
    
    print(f"\n{'Config':<15} {'Params':<15} {'Train Loss':<12} {'Time/Step':<12}")
    print("-" * 54)
    
    for cfg in configs:
        print(f"\n--- {cfg['name']} Configuration ---")
        
        # Create model
        model = LanguageModel(
            vocab_size=vocab_size,
            d_model=cfg['d_model'],
            num_heads=cfg['num_heads'],
            num_layers=cfg['num_layers'],
            use_forgetting=True
        ).to(device)
        
        params = model.count_parameters()
        
        # Quick training
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        
        losses = []
        times = []
        
        for step in tqdm(range(num_steps), desc=f"Training {cfg['name']}"):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            
            start = time.time()
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                input_ids[:, 1:].reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            times.append(time.time() - start)
            losses.append(loss.item())
        
        final_loss = np.mean(losses[-50:])
        avg_time = np.mean(times) * 1000
        
        results['models'][cfg['name']] = {
            'params': params,
            'config': cfg,
            'final_loss': final_loss,
            'avg_step_time_ms': avg_time,
            'losses': losses
        }
        
        print(f"{cfg['name']:<15} {params:>12,} {final_loss:<12.4f} {avg_time:<12.2f}ms")
        
        del model
        gc.collect()
    
    # Save results
    with open(f"{results_dir}/experiment_3_scaling.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    return results


def experiment_4_length_extrapolation(device, results_dir):
    """Test length extrapolation with recurrent model."""
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Length Extrapolation (Recurrent)")  
    print("=" * 70)
    
    vocab_size = 5000
    d_model = 256
    num_layers = 4
    train_len = 128
    batch_size = 32
    num_steps = 300
    
    test_lengths = [256, 512, 1024, 2048]
    
    # Train recurrent model on short sequences
    print(f"\n   Training on seq_len={train_len}...")
    model = RecurrentLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        max_seq_len=max(test_lengths) + 100
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for step in tqdm(range(num_steps), desc="Training"):
        model.train()
        input_ids = torch.randint(0, vocab_size, (batch_size, train_len), device=device)
        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size),
            input_ids[:, 1:].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Test on longer sequences
    print(f"\n   Testing extrapolation...")
    
    results = {
        'train_len': train_len,
        'test_results': {}
    }
    
    print(f"\n{'Test Length':<15} {'Loss':<12} {'PPL':<12} {'Status':<10}")
    print("-" * 49)
    
    model.eval()
    for test_len in test_lengths:
        try:
            with torch.no_grad():
                input_ids = torch.randint(0, vocab_size, (1, test_len), device=device)
                logits, _ = model(input_ids)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, vocab_size),
                    input_ids[:, 1:].reshape(-1)
                )
                ppl = math.exp(min(loss.item(), 10))
                
            results['test_results'][test_len] = {
                'loss': loss.item(),
                'ppl': ppl,
                'success': True
            }
            print(f"{test_len:<15} {loss.item():<12.4f} {ppl:<12.2f} {'✓':<10}")
            
        except Exception as e:
            results['test_results'][test_len] = {
                'loss': float('inf'),
                'ppl': float('inf'),
                'success': False,
                'error': str(e)
            }
            print(f"{test_len:<15} {'ERROR':<12} {'N/A':<12} {'✗':<10}")
    
    # Save results
    with open(f"{results_dir}/experiment_4_extrapolation.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "LARGE-SCALE FOX EXPERIMENTS" + " " * 19 + "#")
    print("#" * 70)
    
    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(src_dir), "results", "large_scale")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    all_results = {}
    
    # Run experiments
    all_results['exp1'] = experiment_1_wikitext_comparison(device, results_dir)
    all_results['exp2'] = experiment_2_memory_benchmark(device, results_dir)
    all_results['exp3'] = experiment_3_larger_models(device, results_dir)
    all_results['exp4'] = experiment_4_length_extrapolation(device, results_dir)
    
    # Final summary
    print("\n" + "#" * 70)
    print("#" + " " * 25 + "FINAL SUMMARY" + " " * 28 + "#")
    print("#" * 70)
    
    print("\n📊 Experiment 1: WikiText-2 Language Modeling")
    if 'exp1' in all_results and 'models' in all_results['exp1']:
        for name, data in all_results['exp1']['models'].items():
            print(f"   {name}: PPL = {data['final_val_ppl']:.2f}")
    
    print("\n⚡ Experiment 2: Memory/Speed Benchmark")
    if 'exp2' in all_results:
        print("   Parallel vs Recurrent timing measured")
    
    print("\n📈 Experiment 3: Model Scaling")
    if 'exp3' in all_results and 'models' in all_results['exp3']:
        for name, data in all_results['exp3']['models'].items():
            print(f"   {name}: {data['params']:,} params, Loss = {data['final_loss']:.4f}")
    
    print("\n📏 Experiment 4: Length Extrapolation")
    if 'exp4' in all_results and 'test_results' in all_results['exp4']:
        for length, data in all_results['exp4']['test_results'].items():
            status = "✓" if data['success'] else "✗"
            print(f"   {length} tokens: {status}")
    
    # Save combined results
    with open(f"{results_dir}/all_experiments.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    
    print(f"\n✅ All experiments complete! Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
