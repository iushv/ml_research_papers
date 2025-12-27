# ArXiv Paper Implementations

A collection of AI/ML research paper implementations with **critical evaluations** and **reproducible experiments**.

---

## 📚 Implemented Papers

| Paper | Status | Key Result |
|-------|--------|------------|
| [Forgetting Transformer (FoX)](papers/forgetting_transformer/) | ✅ Complete | **10% better PPL**, all 6 claims verified |

---

## 🏆 Highlights

### Forgetting Transformer (FoX)
- **All 6 paper claims verified** ✅
- 10% lower perplexity on WikiText-2 (174.19 vs 193.37)
- O(1) memory with recurrent formulation
- 16x length extrapolation (128 → 2048 tokens)

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/iushv/ml_research_papers.git
cd ml_research_papers

# Navigate to a paper
cd papers/forgetting_transformer

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run experiments
python src/verify_claims.py           # Verify paper claims
python src/large_scale_experiments.py # WikiText-2 benchmark
```

---

## 📁 Project Structure

```
arxiv_paper_implementation/
├── papers/
│   └── forgetting_transformer/     # FoX implementation
│       ├── src/                    # Code
│       ├── results/                # Experiment data
│       ├── notebooks/              # Jupyter notebooks
│       ├── FINDINGS.md             # Detailed analysis
│       └── README.md               # Paper guide
├── FINDINGS.md                     # Summary of all findings
├── README.md                       # This file
├── pyproject.toml                  # Dependencies (uv)
└── .venv/                          # Virtual environment
```

---

## 📖 References

- [Forgetting Transformer Paper (arXiv:2503.03420)](https://arxiv.org/abs/2503.03420)

---

*Built for learning, research, and critical evaluation of ML papers.*
