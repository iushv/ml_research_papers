# Dropout Paper Implementation

**Paper**: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"  
**Authors**: Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov (2014)  
**Link**: https://jmlr.org/papers/v15/srivastava14a.html

---

## 🎯 What This Project Teaches

This is a **beginner-friendly, from-scratch implementation** of the Dropout paper with comprehensive learning notes. You'll learn:

| Topic | File | Key Concepts |
|-------|------|--------------|
| Dropout Mechanism | `my_dropout.py` | Masking, scaling, train/eval modes |
| Neural Networks | `my_network.py` | Layers, activations, forward pass |
| Training Loop | `experiment.py` | Optimizer, loss, backpropagation |

---

## 🚀 Quick Start

```bash
cd papers/dropout/src

# Test dropout implementation
python my_dropout.py

# Test neural network
python my_network.py

# Run full experiment (compare with/without dropout)
python experiment.py
```

---

## 📊 Key Results

```
Without Dropout: Train=100%, Test=89%  ← OVERFIT!
With Dropout:    Train=93%,  Test=91%  ← Generalized!
```

Dropout sacrifices training accuracy for better test performance.

---

## 📚 Learning Notes Summary

### 1. Dropout Math
```python
# Training mode:
mask = random() > p
output = input * mask / (1 - p)

# Inference mode:
output = input  # unchanged
```

### 2. Neural Network Architecture
```
Input → Linear → ReLU → [Dropout] → Linear → ReLU → [Dropout] → Linear → Output
```

### 3. Training Loop (5 Essential Steps)
```python
optimizer.zero_grad()     # 1. Clear old gradients
outputs = model(X)        # 2. Forward pass
loss = criterion(outputs) # 3. Compute loss
loss.backward()           # 4. Backward pass
optimizer.step()          # 5. Update weights
```

### 4. Train vs Eval Mode
```python
model.train()  # Dropout ON
model.eval()   # Dropout OFF (use for testing!)
```

---

## 📁 Project Structure

```
dropout/
├── src/
│   ├── my_dropout.py    # Custom Dropout class with notes
│   ├── my_network.py    # MLP with optional Dropout
│   └── experiment.py    # Training loop & comparison
├── results/
│   ├── dropout_comparison_chart_*.png
│   ├── dropout_neuron_visual_*.png
│   └── dropout_code_snippet_*.png
└── README.md
```

---

## 🔑 Key Takeaways

1. **Dropout = Random dropping + Scaling** to maintain expected value
2. **Always use `model.eval()`** before testing (common bug!)
3. **Apply dropout AFTER activation**, not on output layer
4. **Overfitting** = High train accuracy, low test accuracy
5. **Regularization** trades train accuracy for generalization

---

## 📖 Further Reading

- [Original Paper (2014)](https://jmlr.org/papers/v15/srivastava14a.html)
- [PyTorch Dropout Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
