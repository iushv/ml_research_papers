"""Bidirectional LSTM Encoder for Seq2Seq."""
import torch
import torch.nn as nn
from typing import Tuple


class Encoder(nn.Module):
    """
    Bidirectional LSTM Encoder.
    Returns all hidden states (for attention) and final states (for decoder init).
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src: [batch, src_len]
        Returns:
            outputs: [batch, src_len, hidden_dim * 2]
            (hidden, cell): [num_layers * 2, batch, hidden_dim]
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)
