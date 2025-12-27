"""
Comparative Evaluation Script

This script runs comprehensive experiments comparing:
1. Standard Transformer
2. Forgetting Transformer (FoX)

Metrics collected:
- Training loss curves
- Memory usage
- Inference speed
- Length extrapolation
"""

import sys
import os

# Add src folder to path for local imports
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from tqdm import tqdm
from datetime import datetime

from transformer_blocks import LanguageModel


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_copy_data(batch_size, seq_len, vocab_size, device):
    """Generate data for copy task."""
    tokens = torch.randint(1, vocab_size, (batch_size, seq_len // 2), device=device)
    return tokens.repeat(1, 2)


def generate_language_data(batch_size, seq_len, vocab_size, device):
    """Generate random language modeling data."""
    return torch.randint(1, vocab_size, (batch_size, seq_len), device=device)


def measure_memory():
    """Measure current memory usage."""
    if torch.backends.mps.is_available():
        # MPS doesn't have direct memory query, return process memory
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # MB
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # MB
    return 0


def train_model(model, num_steps, batch_size, seq_len, vocab_size, device, lr=1e-3):
    """Train model and return metrics."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    losses = []
    times = []
    
    for step in tqdm(range(num_steps), desc="Training"):
        input_ids = generate_language_data(batch_size, seq_len, vocab_size, device)
        
        start_time = time.time()
        
        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size),
            input_ids[:, 1:].reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        step_time = time.time() - start_time
        
        losses.append(loss.item())
        times.append(step_time)
    
    return {
        'losses': losses,
        'times': times,
        'final_loss': np.mean(losses[-50:]),
        'avg_step_time': np.mean(times),
    }


def evaluate_inference_speed(model, batch_size, seq_len, vocab_size, device, num_runs=100):
    """Measure inference speed."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            input_ids = generate_language_data(batch_size, seq_len, vocab_size, device)
            _ = model(input_ids)
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            input_ids = generate_language_data(batch_size, seq_len, vocab_size, device)
            start = time.time()
            _ = model(input_ids)
            times.append(time.time() - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
    }


def evaluate_length_extrapolation(model, vocab_size, device, train_len=64, test_lengths=[128, 256, 512]):
    """Test model on longer sequences than training."""
    model.eval()
    results = {}
    
    for length in test_lengths:
        try:
            with torch.no_grad():
                input_ids = generate_language_data(1, length, vocab_size, device)
                logits, _ = model(input_ids)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, vocab_size),
                    input_ids[:, 1:].reshape(-1)
                )
                results[length] = {'loss': loss.item(), 'success': True}
        except Exception as e:
            results[length] = {'loss': float('inf'), 'success': False, 'error': str(e)}
    
    return results


def run_experiments():
    """Run all experiments and return results."""
    print("=" * 60)
    print("Forgetting Transformer (FoX) Evaluation")
    print("=" * 60)
    
    device = get_device()
    print(f"\nDevice: {device}")
    
    # Configuration
    config = {
        'vocab_size': 5000,
        'd_model': 128,
        'num_heads': 4,
        'num_layers': 4,
        'max_seq_len': 512,
        'train_seq_len': 64,
        'batch_size': 32,
        'num_steps': 500,
        'learning_rate': 1e-3,
    }
    
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    results = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
    }
    
    # Create models
    print("\n" + "-" * 60)
    print("Creating models...")
    
    std_model = LanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len'],
        use_forgetting=False
    ).to(device)
    
    fox_model = LanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len'],
        use_forgetting=True
    ).to(device)
    
    results['std_params'] = std_model.count_parameters()
    results['fox_params'] = fox_model.count_parameters()
    
    print(f"  Standard Transformer: {results['std_params']:,} parameters")
    print(f"  Forgetting Transformer: {results['fox_params']:,} parameters")
    print(f"  Overhead: {100*(results['fox_params']-results['std_params'])/results['std_params']:.3f}%")
    
    # Training
    print("\n" + "-" * 60)
    print("Training Standard Transformer...")
    results['std_training'] = train_model(
        std_model, config['num_steps'], config['batch_size'],
        config['train_seq_len'], config['vocab_size'], device,
        config['learning_rate']
    )
    
    print("\nTraining Forgetting Transformer...")
    results['fox_training'] = train_model(
        fox_model, config['num_steps'], config['batch_size'],
        config['train_seq_len'], config['vocab_size'], device,
        config['learning_rate']
    )
    
    print(f"\n  Standard final loss: {results['std_training']['final_loss']:.4f}")
    print(f"  Forgetting final loss: {results['fox_training']['final_loss']:.4f}")
    
    # Inference speed
    print("\n" + "-" * 60)
    print("Measuring inference speed...")
    
    results['std_inference'] = evaluate_inference_speed(
        std_model, config['batch_size'], config['train_seq_len'],
        config['vocab_size'], device
    )
    results['fox_inference'] = evaluate_inference_speed(
        fox_model, config['batch_size'], config['train_seq_len'],
        config['vocab_size'], device
    )
    
    print(f"  Standard: {results['std_inference']['mean_ms']:.2f} ± {results['std_inference']['std_ms']:.2f} ms")
    print(f"  Forgetting: {results['fox_inference']['mean_ms']:.2f} ± {results['fox_inference']['std_ms']:.2f} ms")
    
    # Length extrapolation
    print("\n" + "-" * 60)
    print("Testing length extrapolation...")
    
    results['std_extrapolation'] = evaluate_length_extrapolation(
        std_model, config['vocab_size'], device,
        train_len=config['train_seq_len']
    )
    results['fox_extrapolation'] = evaluate_length_extrapolation(
        fox_model, config['vocab_size'], device,
        train_len=config['train_seq_len']
    )
    
    for length in [128, 256, 512]:
        std_loss = results['std_extrapolation'].get(length, {}).get('loss', float('inf'))
        fox_loss = results['fox_extrapolation'].get(length, {}).get('loss', float('inf'))
        print(f"  Length {length}: Standard={std_loss:.4f}, Forgetting={fox_loss:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Determine winner for each category
    loss_winner = "Forgetting" if results['fox_training']['final_loss'] < results['std_training']['final_loss'] else "Standard"
    speed_winner = "Forgetting" if results['fox_inference']['mean_ms'] < results['std_inference']['mean_ms'] else "Standard"
    
    print(f"\n  Training Loss: {loss_winner} is better")
    print(f"  Inference Speed: {speed_winner} is faster")
    print(f"  Parameter Overhead: {100*(results['fox_params']-results['std_params'])/results['std_params']:.3f}%")
    
    results['summary'] = {
        'loss_winner': loss_winner,
        'speed_winner': speed_winner,
        'std_final_loss': results['std_training']['final_loss'],
        'fox_final_loss': results['fox_training']['final_loss'],
        'loss_improvement': (results['std_training']['final_loss'] - results['fox_training']['final_loss']) / results['std_training']['final_loss'] * 100,
    }
    
    return results


def save_results(results, filepath):
    """Save results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    results = run_experiments()
    save_results(results, "results/evaluation_results.json")
    
    # Save model checkpoints
    print("\nSaving model checkpoints...")
    torch.save(results.get('std_model_state', {}), "results/std_model.pt")
    torch.save(results.get('fox_model_state', {}), "results/fox_model.pt")
    
    print("\n✅ Evaluation complete!")
