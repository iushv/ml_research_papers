# ML Research Papers

From-scratch implementations of ML/AI research papers with verified experiments.

---

## What This Is

I learn best by building things, not reading about them. This repo is where I implement research papers from scratch to actually understand what's happening under the hood.

No libraries doing the heavy lifting. Just PyTorch, numpy, and the original papers.

---

## Implemented Papers

| Paper | Year | What I Learned |
|-------|------|----------------|
| [Dropout](papers/dropout/) | 2014 | Why randomly breaking your network makes it stronger |
| [Seq2Seq with Attention](papers/seq2seq_attention/) | 2015 | How attention lets decoders focus on relevant parts |
| [Forgetting Transformer (FoX)](papers/forgetting_transformer/) | 2025 | O(1) memory attention that actually works |

---

## Why From Scratch

Most tutorials show you how to call `model.fit()`. That's not really understanding.

When you implement dropout yourself, you realize it's embarrassingly simple—just multiply by a random binary mask during training, scale during inference. But until you write those 10 lines of code, it feels like magic.

Same with attention. The papers are dense, but the core ideas are often surprisingly elegant once you strip away the notation.

---

## Results

### Dropout
```
Without Dropout: Train=100%, Test=89% (overfitting)
With Dropout:    Train=93%,  Test=91% (generalization)
```

### Forgetting Transformer
```
Standard Transformer: PPL=193.37
FoX (Parallel):       PPL=174.19 (10% better)
FoX (Recurrent):      O(1) memory, 16x length extrapolation
```

---

## Running the Experiments

```bash
git clone https://github.com/iushv/ml_research_papers.git
cd ml_research_papers

# Pick a paper
cd papers/dropout  # or papers/forgetting_transformer

# Setup
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib

# Run
python src/experiment.py
```

---

## Structure

```
ml_research_papers/
├── papers/
│   ├── dropout/
│   │   ├── src/
│   │   │   ├── my_dropout.py      # The actual implementation
│   │   │   ├── my_network.py      # Simple network to test it
│   │   │   └── experiment.py      # Training + comparison
│   │   └── README.md              # Paper-specific notes
│   │
│   ├── seq2seq_attention/
│   │   ├── models/
│   │   │   ├── encoder.py         # Bidirectional LSTM encoder
│   │   │   ├── attention.py       # Luong attention mechanism
│   │   │   ├── decoder.py         # Attention-based decoder
│   │   │   └── seq2seq.py         # Full model
│   │   ├── train.py
│   │   └── README.md
│   │
│   └── forgetting_transformer/
│       ├── src/
│       │   ├── forgetting_attention.py
│       │   ├── recurrent_attention.py
│       │   └── large_scale_experiments.py
│       └── README.md
│
└── pyproject.toml
```

Each paper folder has its own README with:
- The core insight in plain English
- Step-by-step implementation notes
- Pitfalls I ran into
- Visual results

---

## Papers I Want to Implement Next

- [ ] Mamba (2024) — State space models
- [ ] LoRA (2021) — Low-rank adaptation
- [ ] BERT (2018) — Masked language modeling
- [ ] Transformer (2017) — The original "Attention Is All You Need"

---

Learning in public. Feel free to use these implementations or point out where I got things wrong.
