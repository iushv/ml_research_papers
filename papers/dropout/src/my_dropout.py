"""
================================================================================
DROPOUT: A Simple Way to Prevent Neural Networks from Overfitting
================================================================================

Paper: Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov (2014)
Link: https://jmlr.org/papers/v15/srivastava14a.html

This file contains a from-scratch implementation of Dropout with detailed
learning notes for beginners.

================================================================================
LEARNING NOTES
================================================================================

1. WHAT IS DROPOUT?
   - A regularization technique to prevent overfitting
   - During training, randomly "drop" (set to 0) neurons with probability p
   - During inference, use all neurons (no dropping)

2. WHY DOES IT WORK?
   - Prevents co-adaptation: neurons can't rely on specific other neurons
   - Ensemble effect: like training many different networks
   - Forces redundancy: network must learn robust features

3. THE MATH:
   Training:   output = input * mask / (1 - p)
   Inference:  output = input  (unchanged)
   
   Where:
   - mask is a binary tensor (0 or 1 for each element)
   - p is the dropout probability (typically 0.5)
   - Division by (1-p) maintains expected value

4. INVERTED DROPOUT:
   We scale during training (divide by 1-p) instead of during inference.
   This is called "inverted dropout" and is the standard approach because:
   - No changes needed at inference time
   - Inference is the same regardless of dropout rate

================================================================================
"""

import torch
import torch.nn as nn


class Dropout(nn.Module):
    """
    Custom Dropout implementation from scratch.
    
    This implements "inverted dropout" where we scale during training
    to maintain the expected value of activations.
    
    Example:
        >>> dropout = Dropout(p=0.5)
        >>> x = torch.ones(10) * 2.0
        >>> dropout.train()
        >>> print(dropout(x))  # Some values 4.0, some 0.0 (random)
        >>> dropout.eval()
        >>> print(dropout(x))  # All values 2.0 (unchanged)
    """
    
    def __init__(self, p: float = 0.5):
        """
        Initialize Dropout layer.
        
        Args:
            p: Probability of dropping a neuron. Default 0.5.
               Common values: 0.5 for hidden layers, 0.2-0.3 for input layer
        
        Raises:
            ValueError: If p is not in [0, 1)
        """
        super().__init__()
        
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout to input tensor.
        
        LEARNING NOTE: The training/eval mode switch
        =============================================
        - self.training is True when model.train() was called
        - self.training is False when model.eval() was called
        - This is inherited from nn.Module automatically!
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            - Training: Tensor with random neurons dropped and scaled
            - Inference: Input unchanged
        """
        # During inference, return input unchanged
        if not self.training:
            return x
        
        # During training, create random binary mask
        # torch.rand_like(x) creates random values in [0, 1) with same shape as x
        # (random > p) gives True where we KEEP neurons, False where we DROP
        mask = torch.rand_like(x) > self.p
        
        # Apply mask and scale
        # The mask is boolean, but PyTorch handles multiplication correctly
        # Division by (1-p) maintains expected value: E[output] = input
        return x * mask / (1 - self.p)
    
    def extra_repr(self) -> str:
        """String representation for printing the layer."""
        return f'p={self.p}'


# =============================================================================
# TESTS AND DEMONSTRATIONS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DROPOUT IMPLEMENTATION - LEARNING DEMO")
    print("=" * 60)
    
    # Create input tensor
    x = torch.ones(10) * 2.0
    print(f"\nInput: {x.tolist()}")
    print(f"Input mean: {x.mean().item():.2f}")
    
    # Create dropout layer
    dropout = Dropout(p=0.5)
    print(f"\nDropout layer: {dropout}")
    
    # Training mode - dropout active
    print("\n--- TRAINING MODE (dropout active) ---")
    dropout.train()
    for i in range(3):
        output = dropout(x)
        print(f"  Sample {i+1}: {output.tolist()}")
        print(f"  Mean: {output.mean().item():.2f} (should be ~2.0 due to scaling)")
    
    # Inference mode - dropout disabled
    print("\n--- INFERENCE MODE (dropout disabled) ---")
    dropout.eval()
    for i in range(3):
        output = dropout(x)
        print(f"  Sample {i+1}: {output.tolist()}")
        print(f"  Mean: {output.mean().item():.2f} (always exactly 2.0)")
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY:")
    print("- Training: Random dropping + scaling preserves expected value")
    print("- Inference: No changes, all neurons active")
    print("=" * 60)