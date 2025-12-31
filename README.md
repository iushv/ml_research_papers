# ML Research Papers

A collection of **from-scratch implementations** of foundational and innovative ML/AI research papers, with detailed learning notes and verified experiments.

---

## Implemented Papers

| Paper | Year | Key Innovation | Status |
|-------|------|----------------|--------|
| [Dropout](papers/dropout/) | 2014 | Regularization via random neuron dropping | Complete |
| [Seq2Seq with Attention](papers/seq2seq_attention/) | 2014/2015 | Encoder-decoder with Luong attention | Complete |
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
cd papers/seq2seq_attention  # or papers/dropout

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install torch

# Run experiments
python train.py
```

---

## Repository Structure

```
ml_research_papers/
├── papers/
│   ├── dropout/                    # Dropout (2014)
│   ├── seq2seq_attention/          # Seq2Seq + Luong Attention (2014/2015)
│   │   ├── models/
│   │   │   ├── encoder.py
│   │   │   ├── attention.py
│   │   │   ├── decoder.py
│   │   │   └── seq2seq.py
│   │   ├── train.py
│   │   └── README.md
│   └── forgetting_transformer/     # FoX (2025)
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

### Seq2Seq with Attention
```
20M trainable parameters
Loss: 8.5 -> 7.9 in 3 epochs
Greedy decoding inference working
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
- [Seq2Seq Paper (2014)](https://arxiv.org/abs/1409.3215)
- [Luong Attention Paper (2015)](https://arxiv.org/abs/1508.04025)
- [Forgetting Transformer Paper (2025)](https://arxiv.org/abs/2503.03420)

---

## Learning Resources

Each paper folder contains detailed learning notes covering:
- Mathematical foundations
- Step-by-step implementation guide
- Training loop mechanics
- Common pitfalls and debugging tips

---

*Built for learning, research, and the open-source ML community.*
