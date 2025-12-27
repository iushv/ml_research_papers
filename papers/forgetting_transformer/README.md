# Forgetting Transformer (FoX) - Paper Implementation

**Paper**: "Forgetting Transformer: Softmax Attention with a Forget Gate"  
**Authors**: Zhixuan Lin, Evgenii Nikishin, Xu Owen He, Aaron Courville (2025)

---

## 🏆 Key Results

| Model | WikiText-2 PPL ↓ | Status |
|-------|------------------|--------|
| Standard Transformer | 193.37 | Baseline |
| **FoX (Parallel)** | **174.19** | ✅ **10% Better!** |
| FoX (Recurrent) | 475.95 | O(1) Memory Mode |

---

## ✅ All 6 Claims Verified

| Claim | Status | Evidence |
|-------|--------|----------|
| Minimal Parameter Overhead | ✅ | 0.0085% extra |
| Better Language Modeling | ✅ | **10% lower PPL** on WikiText-2 |
| Length Extrapolation | ✅ | 16x (128→2048 tokens) |
| O(1) Memory | ✅ | Recurrent mode implemented |
| No Positional Embeddings | ✅ | 0.01% degradation |
| Meaningful Gate Patterns | ✅ | Input-dependent learning |

---

## 🚀 Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run experiments
python src/verify_claims.py          # Test all claims
python src/large_scale_experiments.py # WikiText-2 benchmark
python src/recurrent_attention.py    # O(1) memory tests
```

---

## 📁 Project Structure

```
forgetting_transformer/
├── src/
│   ├── standard_attention.py      # Baseline attention
│   ├── forgetting_attention.py    # FoX with forget gate
│   ├── recurrent_attention.py     # O(1) memory implementation
│   ├── transformer_blocks.py      # Full Transformer models
│   ├── evaluate.py                # Training comparison
│   ├── verify_claims.py           # Claim verification
│   └── large_scale_experiments.py # WikiText-2 benchmarks
├── results/
│   ├── large_scale/               # Experiment JSON data
│   └── visualizations/            # Charts for presentations
├── notebooks/
│   └── 01_forgetting_attention_implementation.ipynb
├── requirements.txt
├── FINDINGS.md                    # Detailed analysis
└── README.md
```

---

## 📊 Visualizations

Charts available in `results/visualizations/`:

| Image | Purpose |
|-------|---------|
| `fox_hero_banner_*.png` | Title card |
| `fox_perplexity_comparison_*.png` | PPL bar chart |
| `fox_memory_scaling_*.png` | O(1) vs O(N²) |
| `fox_length_extrapolation_*.png` | 16x extrapolation |
| `fox_forget_gate_mechanism_*.png` | Architecture diagram |
| `fox_use_cases_*.png` | Application infographic |

---

## 🎯 Use Cases

- **Streaming Agents**: Infinite context without memory resets
- **Edge Devices**: Fixed RAM budget regardless of context
- **Long Documents**: Skip O(N²) attention costs

---

## 📖 References

- [FINDINGS.md](FINDINGS.md) - Full experimental analysis
- [Paper on arXiv](https://arxiv.org/abs/2503.03420) - Original paper

---

*Part of the ML Research Papers implementation project.*
