"""
Dropout: A Simple Way to Prevent Neural Networks from Overfitting
==================================================================

Paper: Srivastava et al. (2014)
Link: https://jmlr.org/papers/v15/srivastava14a.html

This implementation teaches you:
1. What dropout is and why it works
2. How to implement dropout from scratch
3. The difference between training and inference modes
4. Visualizing dropout's effect on overfitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PART 1: Understanding Dropout
# =============================================================================

"""
THE PROBLEM: Overfitting
------------------------
Deep neural networks have millions of parameters. They can memorize 
the training data instead of learning generalizable patterns.

THE SOLUTION: Dropout
---------------------
During training, randomly "drop" (set to zero) neurons with probability p.

Why does this work?
1. Prevents co-adaptation: Neurons can't rely on specific other neurons
2. Ensemble effect: Like training many different networks
3. Forces redundancy: Network must learn robust features

Key insight: At test time, we use ALL neurons but scale by (1-p)
OR we scale during training (inverted dropout) - this is more common.
"""


# =============================================================================
# PART 2: Dropout from Scratch
# =============================================================================

class DropoutFromScratch(nn.Module):
    """
    Implements dropout exactly as described in the paper.
    
    Training: Randomly zero out neurons with probability p
    Inference: Use all neurons (no dropping)
    
    We use "inverted dropout" which scales during training instead of testing.
    This is what PyTorch and most frameworks use.
    """
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of dropping a neuron (default 0.5)
        """
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout to input tensor.
        
        Math:
            Training:   y = x * mask / (1 - p)   where mask ~ Bernoulli(1-p)
            Inference:  y = x                    (no change)
            
        The division by (1-p) is "inverted dropout" - it ensures the 
        expected value of activations is the same during train and test.
        """
        # During inference, return input unchanged
        if not self.training:
            return x
        
        # During training, create random mask and apply
        # mask[i] = 1 with probability (1-p), 0 with probability p
        mask = (torch.rand_like(x) > self.p).float()
        
        # Scale by 1/(1-p) to maintain expected value
        # E[y] = E[x * mask / (1-p)] = x * (1-p) / (1-p) = x
        return x * mask / (1 - self.p)
    
    def extra_repr(self) -> str:
        return f'p={self.p}'


# Visual explanation
def visualize_dropout():
    """
    Visualize how dropout works on a simple example.
    """
    print("=" * 60)
    print("DROPOUT VISUALIZATION")
    print("=" * 60)
    
    # Create a simple activation tensor
    x = torch.ones(1, 10) * 2.0  # All values are 2.0
    
    dropout = DropoutFromScratch(p=0.5)
    
    print("\nOriginal activations:")
    print(f"  x = {x.numpy().flatten()}")
    print(f"  Mean = {x.mean().item():.2f}")
    
    # Training mode - apply dropout multiple times
    dropout.train()
    print("\nTraining mode (p=0.5) - 3 random samples:")
    for i in range(3):
        y = dropout(x)
        print(f"  Sample {i+1}: {y.numpy().flatten()}")
        print(f"            Mean = {y.mean().item():.2f} (scaled to preserve expectation)")
    
    # Inference mode - no dropout
    dropout.eval()
    y = dropout(x)
    print("\nInference mode:")
    print(f"  y = {y.numpy().flatten()}")
    print(f"  Mean = {y.mean().item():.2f} (unchanged)")
    
    print("\n✅ Key insight: Mean is preserved (~2.0) in both modes!")
    print("   This is why inverted dropout scales by 1/(1-p)")


# =============================================================================
# PART 3: Neural Network with Dropout
# =============================================================================

class MLPWithoutDropout(nn.Module):
    """Standard MLP without dropout - prone to overfitting."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MLPWithDropout(nn.Module):
    """MLP with dropout - regularized against overfitting."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Use our custom dropout!
        self.dropout1 = DropoutFromScratch(p=dropout_p)
        self.dropout2 = DropoutFromScratch(p=dropout_p)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Dropout after activation
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Dropout after activation
        return self.fc3(x)


# =============================================================================
# PART 4: Experiment - Comparing With vs Without Dropout
# =============================================================================

def create_overfit_dataset(n_train=100, n_test=1000, noise=0.3):
    """
    Create a dataset where overfitting is easy to demonstrate.
    Small training set + large test set + noisy data.
    """
    torch.manual_seed(42)
    
    # Simple classification: two spirals
    def make_spiral(n, delta_t, noise):
        t = torch.linspace(0, 2, n) + delta_t
        x = t * torch.cos(t * 2 * np.pi) + torch.randn(n) * noise
        y = t * torch.sin(t * 2 * np.pi) + torch.randn(n) * noise
        return torch.stack([x, y], dim=1)
    
    # Class 0: one spiral
    X0_train = make_spiral(n_train // 2, 0, noise)
    X0_test = make_spiral(n_test // 2, 0, noise)
    
    # Class 1: another spiral (offset)
    X1_train = make_spiral(n_train // 2, 0.5, noise)
    X1_test = make_spiral(n_test // 2, 0.5, noise)
    
    X_train = torch.cat([X0_train, X1_train])
    y_train = torch.cat([torch.zeros(n_train // 2), torch.ones(n_train // 2)]).long()
    
    X_test = torch.cat([X0_test, X1_test])
    y_test = torch.cat([torch.zeros(n_test // 2), torch.ones(n_test // 2)]).long()
    
    return X_train, y_train, X_test, y_test


def train_epoch(model, X, y, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    _, predicted = outputs.max(1)
    accuracy = (predicted == y).float().mean().item()
    return loss.item(), accuracy


def evaluate(model, X, y, criterion):
    """Evaluate model."""
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y)
        _, predicted = outputs.max(1)
        accuracy = (predicted == y).float().mean().item()
    return loss.item(), accuracy


def run_experiment():
    """
    Compare models with and without dropout.
    This demonstrates dropout's regularization effect.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Dropout vs No Dropout")
    print("=" * 60)
    
    # Create dataset designed to cause overfitting
    X_train, y_train, X_test, y_test = create_overfit_dataset(
        n_train=100,  # Small training set
        n_test=1000,  # Large test set
        noise=0.3
    )
    
    print(f"\nDataset: {len(X_train)} train, {len(X_test)} test samples")
    print("(Small train set makes overfitting easy to demonstrate)")
    
    # Hyperparameters
    input_dim = 2
    hidden_dim = 256  # Large hidden layer to encourage overfitting
    output_dim = 2
    epochs = 500
    lr = 0.01
    
    # Model without dropout
    model_no_dropout = MLPWithoutDropout(input_dim, hidden_dim, output_dim)
    opt_no_dropout = torch.optim.Adam(model_no_dropout.parameters(), lr=lr)
    
    # Model with dropout
    model_dropout = MLPWithDropout(input_dim, hidden_dim, output_dim, dropout_p=0.5)
    opt_dropout = torch.optim.Adam(model_dropout.parameters(), lr=lr)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'no_dropout': {'train_acc': [], 'test_acc': []},
        'dropout': {'train_acc': [], 'test_acc': []}
    }
    
    print("\nTraining...")
    for epoch in range(epochs):
        # Train without dropout
        _, train_acc_no = train_epoch(model_no_dropout, X_train, y_train, opt_no_dropout, criterion)
        _, test_acc_no = evaluate(model_no_dropout, X_test, y_test, criterion)
        
        # Train with dropout
        _, train_acc_d = train_epoch(model_dropout, X_train, y_train, opt_dropout, criterion)
        _, test_acc_d = evaluate(model_dropout, X_test, y_test, criterion)
        
        history['no_dropout']['train_acc'].append(train_acc_no)
        history['no_dropout']['test_acc'].append(test_acc_no)
        history['dropout']['train_acc'].append(train_acc_d)
        history['dropout']['test_acc'].append(test_acc_d)
        
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:3d} | No Dropout: train={train_acc_no:.2%} test={test_acc_no:.2%} | "
                  f"Dropout: train={train_acc_d:.2%} test={test_acc_d:.2%}")
    
    # Final results
    print("\n" + "-" * 60)
    print("FINAL RESULTS")
    print("-" * 60)
    print(f"\nWithout Dropout:")
    print(f"  Train Accuracy: {history['no_dropout']['train_acc'][-1]:.2%}")
    print(f"  Test Accuracy:  {history['no_dropout']['test_acc'][-1]:.2%}")
    print(f"  Gap (overfit):  {history['no_dropout']['train_acc'][-1] - history['no_dropout']['test_acc'][-1]:.2%}")
    
    print(f"\nWith Dropout (p=0.5):")
    print(f"  Train Accuracy: {history['dropout']['train_acc'][-1]:.2%}")
    print(f"  Test Accuracy:  {history['dropout']['test_acc'][-1]:.2%}")
    print(f"  Gap (overfit):  {history['dropout']['train_acc'][-1] - history['dropout']['test_acc'][-1]:.2%}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy curves
    axes[0].plot(history['no_dropout']['train_acc'], 'r-', label='No Dropout (Train)', linewidth=2)
    axes[0].plot(history['no_dropout']['test_acc'], 'r--', label='No Dropout (Test)', linewidth=2)
    axes[0].plot(history['dropout']['train_acc'], 'b-', label='Dropout (Train)', linewidth=2)
    axes[0].plot(history['dropout']['test_acc'], 'b--', label='Dropout (Test)', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Training vs Test Accuracy', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Generalization gap
    gap_no_dropout = [t - v for t, v in zip(history['no_dropout']['train_acc'], history['no_dropout']['test_acc'])]
    gap_dropout = [t - v for t, v in zip(history['dropout']['train_acc'], history['dropout']['test_acc'])]
    
    axes[1].plot(gap_no_dropout, 'r-', label='No Dropout', linewidth=2)
    axes[1].plot(gap_dropout, 'b-', label='With Dropout', linewidth=2)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Train - Test Accuracy', fontsize=12)
    axes[1].set_title('Generalization Gap (Lower = Better)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/dropout_comparison.png', dpi=150)
    print("\n📊 Plot saved to results/dropout_comparison.png")
    
    return history


# =============================================================================
# PART 5: Key Takeaways
# =============================================================================

def print_takeaways():
    """Print the key lessons from this implementation."""
    print("\n" + "=" * 60)
    print("📚 KEY TAKEAWAYS")
    print("=" * 60)
    
    print("""
1. WHAT IS DROPOUT?
   - Randomly zero out neurons during training (probability p)
   - Forces network to learn redundant representations
   - Acts as a regularizer to prevent overfitting

2. INVERTED DROPOUT
   - Scale activations by 1/(1-p) during training
   - No scaling needed at inference
   - Maintains expected value of activations

3. WHEN TO USE
   - After fully-connected layers (most effective)
   - Also works after convolutional layers
   - Common values: p=0.5 for hidden layers, p=0.2 for input

4. TRAINING vs INFERENCE
   - model.train()  → dropout active
   - model.eval()   → dropout disabled
   - ALWAYS call model.eval() before testing!

5. WHY IT WORKS
   - Ensemble interpretation: averaging many "thinned" networks
   - Prevents co-adaptation of neurons
   - Adds noise that acts as regularization
""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DROPOUT PAPER IMPLEMENTATION")
    print("'A Simple Way to Prevent Neural Networks from Overfitting'")
    print("Srivastava et al., 2014")
    print("=" * 60)
    
    # Part 1: Visualize how dropout works
    visualize_dropout()
    
    # Part 2: Run comparison experiment
    run_experiment()
    
    # Part 3: Print key takeaways
    print_takeaways()
    
    print("\n✅ Implementation complete!")
