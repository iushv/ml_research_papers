# ML Research Papers

A collection of **from-scratch implementations** of foundational and novel ML/AI research papers, with detailed learning notes and verified experiments.

---

## Implemented Papers

| Paper | Year | Key Innovation | Status |
|-------|------|----------------|--------|
| [Dropout](papers/dropout/) | 2014 | Regularization via random neuron dropping | Complete |
| [Forgetting Transformer (FoX)](papers/forgetting_transformer/) | 2025 | O(1) memory attention with forget gates | 6/6 claims verified |

---

## Project Goals

1. **Implement** research papers from scratch (not just use libraries)
2. **Understand** the math and intuition behind each innovation
3. **Verify** paper claims with reproducible experiments
4. **Document** learnings for the community

---

## Quick Start

```bash
git clone https://github.com/iushv/ml_research_papers.git
cd ml_research_papers

# Pick a paper
cd papers/dropout  # or papers/forgetting_transformer

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib

# Run experiments
python src/experiment.py
```

---

## Repository Structure

```
ml_research_papers/
├── papers/
│   ├── dropout/                    # Dropout (2014)
│   │   ├── src/
│   │   │   ├── my_dropout.py       # Custom Dropout implementation
│   │   │   ├── my_network.py       # Neural network with Dropout
│   │   │   └── experiment.py       # Training loop & comparison
│   │   ├── results/                # Visualizations
│   │   └── README.md               # Paper-specific guide
│   │
│   └── forgetting_transformer/     # FoX (2025)
│       ├── src/
│       │   ├── forgetting_attention.py
│       │   ├── recurrent_attention.py
│       │   └── large_scale_experiments.py
│       ├── results/
│       └── README.md
│
├── README.md                       # This file
└── pyproject.toml                  # Dependencies
```

---

## Highlight Results

### Dropout
```
Without Dropout: Train=100%, Test=89% (10.7% overfit gap)
With Dropout:    Train=93%,  Test=91% (1.7% gap) 
```

### Forgetting Transformer
```
Standard Transformer: PPL=193.37
FoX (Parallel):       PPL=174.19 (10% better)
FoX (Recurrent):      O(1) memory, 16x length extrapolation
```

---

## Links

- [Dropout Paper (2014)](https://jmlr.org/papers/v15/srivastava14a.html)
- [Forgetting Transformer Paper (2025)](https://arxiv.org/abs/2503.03420)

---

## Learning Resources

Each paper folder contains detailed learning notes covering:
- Mathematical foundations
- Step-by-step implementation guide
- Training loop mechanics
- Common pitfalls and debugging tips

---

*Built for learning, research, and the open-source ML community.* ⭐
