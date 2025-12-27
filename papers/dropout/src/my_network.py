"""
================================================================================
NEURAL NETWORK WITH DROPOUT - FROM SCRATCH IMPLEMENTATION
================================================================================

This file teaches you how to:
1. Build a Multi-Layer Perceptron (MLP) from scratch
2. Properly integrate Dropout layers
3. Understand the data flow through a neural network

================================================================================
LEARNING NOTES: BUILDING A NEURAL NETWORK
================================================================================

1. LAYERS IN A NEURAL NETWORK:
   - Linear (Dense) layer: y = Wx + b
   - Activation function: Introduces non-linearity (ReLU, Sigmoid, etc.)
   - Dropout: Regularization to prevent overfitting

2. TYPICAL MLP ARCHITECTURE:
   Input → Linear → ReLU → (Dropout) → Linear → ReLU → (Dropout) → Linear → Output
   
3. KEY RULES FOR DROPOUT PLACEMENT:
   ✅ Apply AFTER activation functions (e.g., after ReLU)
   ✅ Apply to hidden layers
   ❌ DON'T apply to the output layer (would randomly zero predictions!)
   ❌ DON'T apply before the first layer (input dropout is different)

4. UNDERSTANDING nn.Linear:
   nn.Linear(in_features, out_features)
   - in_features: size of input to this layer
   - out_features: size of output from this layer
   
   Example for 3-layer network with input=10, hidden=32, output=2:
   fc1: 10 → 32  (input_dim → hidden_dim)
   fc2: 32 → 32  (hidden_dim → hidden_dim)
   fc3: 32 → 2   (hidden_dim → output_dim)

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from my_dropout import Dropout


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with optional Dropout.
    
    Architecture:
        Input (input_dim)
            ↓
        Linear (input_dim → hidden_dim)
            ↓
        ReLU
            ↓
        [Dropout] (if use_dropout=True)
            ↓
        Linear (hidden_dim → hidden_dim)
            ↓
        ReLU
            ↓
        [Dropout] (if use_dropout=True)
            ↓
        Linear (hidden_dim → output_dim)
            ↓
        Output (output_dim)
    
    Example:
        >>> model = MLP(input_dim=10, hidden_dim=64, output_dim=2, use_dropout=True)
        >>> x = torch.randn(32, 10)  # Batch of 32 samples, 10 features each
        >>> output = model(x)         # Shape: (32, 2)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_dropout: bool = False,
        dropout_p: float = 0.5
    ):
        """
        Initialize the MLP.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of neurons in hidden layers
            output_dim: Number of output classes/values
            use_dropout: Whether to use dropout (default: False)
            dropout_p: Dropout probability (default: 0.5)
        """
        super().__init__()
        
        # Store configuration
        self.use_dropout = use_dropout
        
        # LEARNING NOTE: Defining Layers
        # ================================
        # We define all layers in __init__ so PyTorch can:
        # 1. Track their parameters for optimization
        # 2. Move them to GPU when model.to('cuda') is called
        # 3. Save/load them properly
        
        # Layer 1: Input → Hidden
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        
        # Layer 2: Hidden → Hidden
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        
        # Layer 3: Hidden → Output
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        
        # Dropout layers (using our custom implementation!)
        if use_dropout:
            self.dropout1 = Dropout(p=dropout_p)
            self.dropout2 = Dropout(p=dropout_p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        LEARNING NOTE: The Forward Pass
        =================================
        Data flows through the network in this order:
        1. Linear transformation (matrix multiplication + bias)
        2. Activation function (introduces non-linearity)
        3. Dropout (regularization, only during training)
        4. Repeat for each layer...
        5. Final linear layer (no activation for classification with CrossEntropyLoss)
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Layer 1: Linear → ReLU → (Dropout)
        x = self.fc1(x)          # Linear transformation
        x = F.relu(x)            # ReLU activation
        if self.use_dropout:
            x = self.dropout1(x) # Dropout (only active during training)
        
        # Layer 2: Linear → ReLU → (Dropout)
        x = self.fc2(x)
        x = F.relu(x)
        if self.use_dropout:
            x = self.dropout2(x)
        
        # Layer 3: Linear (no activation, no dropout)
        # LEARNING NOTE: Why no activation on output?
        # For classification with CrossEntropyLoss, the loss function
        # includes softmax internally, so we output raw logits.
        x = self.fc3(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MLP WITH DROPOUT - DEMONSTRATION")
    print("=" * 60)
    
    # Create sample input
    batch_size = 4
    input_dim = 10
    hidden_dim = 32
    output_dim = 2
    
    x = torch.randn(batch_size, input_dim)
    print(f"\nInput shape: {x.shape}")
    
    # Model WITHOUT dropout
    print("\n--- Model WITHOUT Dropout ---")
    model_no_dropout = MLP(input_dim, hidden_dim, output_dim, use_dropout=False)
    print(f"Parameters: {model_no_dropout.count_parameters():,}")
    print(f"Output shape: {model_no_dropout(x).shape}")
    
    # Model WITH dropout
    print("\n--- Model WITH Dropout ---")
    model_dropout = MLP(input_dim, hidden_dim, output_dim, use_dropout=True, dropout_p=0.5)
    print(f"Parameters: {model_dropout.count_parameters():,}")
    
    # Training mode - outputs vary
    model_dropout.train()
    print("\nTraining mode (dropout active):")
    print(f"  Output 1: {model_dropout(x)[0].tolist()}")
    print(f"  Output 2: {model_dropout(x)[0].tolist()}")
    
    # Inference mode - outputs consistent
    model_dropout.eval()
    print("\nInference mode (dropout disabled):")
    print(f"  Output 1: {model_dropout(x)[0].tolist()}")
    print(f"  Output 2: {model_dropout(x)[0].tolist()}")
    
    print("\n" + "=" * 60)
