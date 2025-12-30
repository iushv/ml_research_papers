# Dropout: Preventing Neural Network Overfitting

**Paper**: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"  
**Authors**: Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov (2014)  
**Link**: https://jmlr.org/papers/v15/srivastava14a.html

---

## What You'll Learn

| File | Concepts |
|------|----------|
| `my_dropout.py` | Dropout math, masking, train/eval modes |
| `my_network.py` | Layer composition, forward pass, activation functions |
| `experiment.py` | Training loop (5 steps), overfitting demonstration |

---

## Quick Start

```bash
cd src
python my_dropout.py    # Test dropout implementation
python my_network.py    # Test neural network
python experiment.py    # Full experiment
```

---

## Results

```
Without Dropout: Train=100%, Test=89% → 10.7% overfit gap
With Dropout:    Train=93%,  Test=91% → 1.7% gap
```

**Dropout reduces overfitting by 84%!**

---

## Key Concepts

### The Dropout Formula
```python
# Training:
mask = random() > p
output = input * mask / (1 - p)

# Inference:
output = input  # unchanged
```

### The Training Loop
```python
optimizer.zero_grad()     # 1. Clear gradients
outputs = model(X)        # 2. Forward pass
loss = criterion(outputs) # 3. Compute loss
loss.backward()           # 4. Backward pass
optimizer.step()          # 5. Update weights
```

### Train vs Eval Mode
```python
model.train()  # Dropout ON
model.eval()   # Dropout OFF ← Don't forget this!
```

---

## Files

```
dropout/
├── src/
│   ├── my_dropout.py    # Custom Dropout class
│   ├── my_network.py    # MLP with Dropout
│   └── experiment.py    # Training comparison
└── results/             # Visualizations
```
