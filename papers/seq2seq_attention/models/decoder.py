"""LSTM Decoder with Luong Attention."""
import torch
import torch.nn as nn
from typing import Tuple
from .attention import LuongAttention


class Decoder(nn.Module):
    """
    LSTM Decoder with Luong Attention.
    Combines LSTM output with attention context for predictions.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        attention: LuongAttention,
        dropout: float = 0.2
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Combines LSTM output + context (hidden_dim + hidden_dim*2 = hidden_dim*3)
        self.wc = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single decoding step.
        
        Args:
            input_token: [batch]
            hidden: [num_layers, batch, hidden_dim]
            cell: [num_layers, batch, hidden_dim]
            encoder_outputs: [batch, src_len, encoder_hidden_dim]
        Returns:
            prediction: [batch, vocab_size]
            hidden: updated hidden state
            cell: updated cell state
        """
        # Embed and pass through LSTM
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        lstm_out = lstm_out.squeeze(1)
        
        # Compute attention context
        context, _ = self.attention(lstm_out, encoder_outputs)
        
        # Combine and project
        combined = torch.cat([lstm_out, context], dim=1)
        tilde_h = torch.tanh(self.wc(combined))
        prediction = self.fc_out(tilde_h)
        
        return prediction, hidden, cell
