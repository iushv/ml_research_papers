# Seq2Seq with Luong Attention

PyTorch implementation of Sequence-to-Sequence with Luong (Multiplicative) Attention.

## Papers Implemented

- Sutskever et al. 2014: [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- Luong et al. 2015: [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

## Architecture

- Bidirectional LSTM Encoder
- Luong Multiplicative (General) Attention
- LSTM Decoder with Attention
- Teacher forcing during training
- Greedy decoding for inference

## Project Structure

```
seq2seq_from_paper/
├── models/
│   ├── __init__.py      # Package exports
│   ├── encoder.py       # Bidirectional LSTM Encoder
│   ├── attention.py     # Luong Attention
│   ├── decoder.py       # Decoder with Attention
│   └── seq2seq.py       # Complete Seq2Seq model
└── train.py             # Training pipeline
```

## Usage

```python
from models import Encoder, Decoder, LuongAttention, Seq2Seq

# Create model
encoder = Encoder(vocab_size, embed_dim, hidden_dim, num_layers)
attention = LuongAttention(hidden_dim * 2, hidden_dim)
decoder = Decoder(vocab_size, embed_dim, hidden_dim, num_layers, attention)
model = Seq2Seq(encoder, decoder, device)

# Training
output = model(src, trg, teacher_forcing_ratio=0.5)

# Inference
translations = model.translate(src, max_len=50)
```

## Run Training

```bash
python train.py
```

## Model Stats

- ~20M trainable parameters (with default hyperparameters)
- 2-layer bidirectional encoder, 2-layer unidirectional decoder
