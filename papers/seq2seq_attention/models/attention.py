"""Luong (Multiplicative) Attention Mechanism."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LuongAttention(nn.Module):
    """
    Luong Multiplicative (General) Attention.
    score(h_dec, h_enc) = h_dec^T * W * h_enc
    """
    
    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=False)
    
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: [batch, decoder_hidden_dim]
            encoder_outputs: [batch, src_len, encoder_hidden_dim]
        Returns:
            context: [batch, encoder_hidden_dim]
            attention_weights: [batch, src_len]
        """
        # Project encoder outputs to decoder space
        encoder_projected = self.W(encoder_outputs)
        
        # Compute attention scores
        scores = torch.bmm(encoder_projected, decoder_hidden.unsqueeze(2)).squeeze(2)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)
        
        # Weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights
