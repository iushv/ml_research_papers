"""
================================================================================
TRAINING LOOP FROM SCRATCH - COMPLETE GUIDE
================================================================================

This file teaches the complete training pipeline for neural networks:
1. Data preparation
2. Model creation
3. Training loop (the heart of deep learning!)
4. Evaluation
5. Comparing models with/without regularization

================================================================================
LEARNING NOTES: THE TRAINING LOOP
================================================================================

The training loop is the core of deep learning. Every iteration has 5 steps:

    1. optimizer.zero_grad()    # Clear old gradients
    2. outputs = model(inputs)  # Forward pass
    3. loss = criterion(outputs, labels)  # Compute loss
    4. loss.backward()          # Backward pass (compute gradients)
    5. optimizer.step()         # Update weights

WHY EACH STEP MATTERS:

1. zero_grad(): Gradients ACCUMULATE by default in PyTorch!
   Without this, gradients from previous iterations add up.
   
2. Forward pass: Data flows through the network, producing predictions.

3. Loss computation: Measures how wrong our predictions are.
   - CrossEntropyLoss: For classification (includes softmax)
   - MSELoss: For regression
   
4. backward(): PyTorch automatically computes gradients using 
   the chain rule (backpropagation). This is the "magic" of autograd!
   
5. step(): Updates weights using: w = w - lr * gradient

================================================================================
LEARNING NOTES: TRAIN VS EVAL MODE
================================================================================

CRITICAL: Always set the correct mode!

    model.train()  # Before training
    - Dropout is ACTIVE (random dropping)
    - BatchNorm uses batch statistics
    
    model.eval()   # Before testing/inference
    - Dropout is DISABLED (use all neurons)
    - BatchNorm uses saved running statistics

Forgetting model.eval() is a common bug that causes:
- Inconsistent predictions
- Lower accuracy than expected

================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from my_network import MLP


def create_dataset(n_train: int = 100, n_test: int = 1000, seed: int = 42):
    """
    Create a simple 2-class classification dataset.
    
    We use a small training set to make overfitting easy to demonstrate.
    The test set is larger to get reliable accuracy estimates.
    
    Args:
        n_train: Number of training samples (small = easy to overfit)
        n_test: Number of test samples
        seed: Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test: Training and test data
    """
    torch.manual_seed(seed)
    
    # Class 0: Gaussian blob centered at (-1, -1)
    X0_train = torch.randn(n_train // 2, 2) + torch.tensor([-1.0, -1.0])
    X0_test = torch.randn(n_test // 2, 2) + torch.tensor([-1.0, -1.0])
    
    # Class 1: Gaussian blob centered at (1, 1)
    X1_train = torch.randn(n_train // 2, 2) + torch.tensor([1.0, 1.0])
    X1_test = torch.randn(n_test // 2, 2) + torch.tensor([1.0, 1.0])
    
    # Combine into single tensors
    X_train = torch.cat([X0_train, X1_train])
    y_train = torch.cat([torch.zeros(n_train // 2), torch.ones(n_train // 2)]).long()
    
    X_test = torch.cat([X0_test, X1_test])
    y_test = torch.cat([torch.zeros(n_test // 2), torch.ones(n_test // 2)]).long()
    
    return X_train, y_train, X_test, y_test


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.01,
    verbose: bool = True
):
    """
    Train a model and track accuracy over time.
    
    This function demonstrates the complete training pipeline with detailed
    comments explaining each step.
    
    Args:
        model: The neural network to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Whether to print progress
        
    Returns:
        train_accs: List of training accuracies per epoch
        test_accs: List of test accuracies per epoch
    """
    # ==========================================================================
    # SETUP: Create optimizer and loss function
    # ==========================================================================
    
    # LEARNING NOTE: The Optimizer
    # Adam is a popular choice because it:
    # - Adapts learning rate per parameter
    # - Includes momentum for faster convergence
    # - Works well with default hyperparameters
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # LEARNING NOTE: The Loss Function
    # CrossEntropyLoss for classification:
    # - Combines LogSoftmax + NLLLoss
    # - Input: raw logits (no softmax needed!)
    # - Target: class indices (not one-hot)
    criterion = nn.CrossEntropyLoss()
    
    # Track metrics
    train_accs = []
    test_accs = []
    
    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    
    for epoch in range(epochs):
        # ----------------------------------------------------------------------
        # 1. TRAINING STEP
        # ----------------------------------------------------------------------
        
        # Set model to training mode (activates dropout)
        model.train()
        
        # Step 1: Clear gradients from previous iteration
        # WHY: Gradients accumulate by default. Without this, we'd be
        # summing gradients from all previous epochs!
        optimizer.zero_grad()
        
        # Step 2: Forward pass - compute predictions
        outputs = model(X_train)
        
        # Step 3: Compute loss - how wrong are we?
        loss = criterion(outputs, y_train)
        
        # Step 4: Backward pass - compute gradients
        # PyTorch automatically computes ∂loss/∂weight for ALL weights
        # using the chain rule (backpropagation)
        loss.backward()
        
        # Step 5: Update weights
        # For each weight: w = w - lr * gradient
        optimizer.step()
        
        # Compute training accuracy
        _, predicted = outputs.max(dim=1)  # Get class with highest score
        train_acc = (predicted == y_train).float().mean().item()
        
        # ----------------------------------------------------------------------
        # 2. EVALUATION STEP
        # ----------------------------------------------------------------------
        
        # Set model to evaluation mode (disables dropout)
        model.eval()
        
        # LEARNING NOTE: torch.no_grad()
        # During evaluation, we don't need gradients, so we disable them:
        # - Saves memory (no gradient tensors stored)
        # - Faster computation
        # - Prevents accidental gradient computation
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = outputs.max(dim=1)
            test_acc = (predicted == y_test).float().mean().item()
        
        # Save metrics
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # Print progress
        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}: Train={train_acc:.1%}, Test={test_acc:.1%}, Loss={loss.item():.4f}")
    
    return train_accs, test_accs


def run_experiment():
    """
    Run the complete experiment comparing models with and without dropout.
    
    This demonstrates how dropout prevents overfitting:
    - Model without dropout: Memorizes training data (overfits)
    - Model with dropout: Generalizes better to test data
    """
    print("=" * 60)
    print("DROPOUT EXPERIMENT: PROVING IT PREVENTS OVERFITTING")
    print("=" * 60)
    
    # Create dataset (small training set = easy to overfit)
    X_train, y_train, X_test, y_test = create_dataset(n_train=100, n_test=1000)
    print(f"\nDataset: {len(X_train)} training, {len(X_test)} test samples")
    print("(Small training set makes overfitting easy to demonstrate)")
    
    # Configuration
    input_dim = 2
    hidden_dim = 256  # Large network = more capacity to memorize
    output_dim = 2
    
    # --------------------------------------------------------------------------
    # Train model WITHOUT dropout
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Training model WITHOUT Dropout")
    print("-" * 60)
    
    model_no_dropout = MLP(input_dim, hidden_dim, output_dim, use_dropout=False)
    train_acc1, test_acc1 = train_model(
        model_no_dropout, X_train, y_train, X_test, y_test
    )
    
    # --------------------------------------------------------------------------
    # Train model WITH dropout
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Training model WITH Dropout (p=0.5)")
    print("-" * 60)
    
    model_dropout = MLP(input_dim, hidden_dim, output_dim, use_dropout=True, dropout_p=0.5)
    train_acc2, test_acc2 = train_model(
        model_dropout, X_train, y_train, X_test, y_test
    )
    
    # --------------------------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    gap1 = train_acc1[-1] - test_acc1[-1]
    gap2 = train_acc2[-1] - test_acc2[-1]
    
    print(f"\nWithout Dropout:")
    print(f"  Train Accuracy: {train_acc1[-1]:.1%}")
    print(f"  Test Accuracy:  {test_acc1[-1]:.1%}")
    print(f"  Gap (overfit):  {gap1:.1%}")
    
    print(f"\nWith Dropout:")
    print(f"  Train Accuracy: {train_acc2[-1]:.1%}")
    print(f"  Test Accuracy:  {test_acc2[-1]:.1%}")
    print(f"  Gap (overfit):  {gap2:.1%}")
    
    print("\n" + "-" * 60)
    print("CONCLUSION:")
    if test_acc2[-1] > test_acc1[-1]:
        print("✅ Dropout IMPROVED test accuracy!")
    print(f"✅ Dropout REDUCED overfitting gap: {gap1:.1%} → {gap2:.1%}")
    print("-" * 60)
    
    return train_acc1, test_acc1, train_acc2, test_acc2


if __name__ == "__main__":
    run_experiment()