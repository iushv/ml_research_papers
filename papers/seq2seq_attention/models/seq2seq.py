"""Complete Seq2Seq Model with Luong Attention."""
import torch
import torch.nn as nn
import random
from typing import Tuple
from .encoder import Encoder
from .decoder import Decoder


class Seq2Seq(nn.Module):
    """
    Seq2Seq with Luong Attention.
    Combines bidirectional encoder and attentional decoder.
    """
    
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Bridge: transforms bidirectional encoder state to unidirectional decoder state
        self.fc_hidden = nn.Linear(encoder.hidden_dim * 2, decoder.hidden_dim)
        self.fc_cell = nn.Linear(encoder.hidden_dim * 2, decoder.hidden_dim)
    
    def _transform_encoder_state(
        self,
        hidden: torch.Tensor,
        cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform bidirectional encoder state to decoder state."""
        num_layers = hidden.shape[0] // 2
        batch_size = hidden.shape[1]
        hidden_dim = hidden.shape[2]
        
        # Reshape and concatenate forward/backward directions
        hidden = hidden.view(num_layers, 2, batch_size, hidden_dim)
        cell = cell.view(num_layers, 2, batch_size, hidden_dim)
        
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
        
        # Project to decoder size
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))
        
        return hidden, cell
    
    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing.
        
        Args:
            src: [batch, src_len]
            trg: [batch, trg_len]
            teacher_forcing_ratio: probability of using ground truth
        Returns:
            outputs: [batch, trg_len - 1, vocab_size]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, trg_len - 1, vocab_size).to(self.device)
        
        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(src)
        hidden, cell = self._transform_encoder_state(hidden, cell)
        
        # Decode step by step
        input_token = trg[:, 0]
        
        for t in range(trg_len - 1):
            prediction, hidden, cell = self.decoder(
                input_token, hidden, cell, encoder_outputs
            )
            outputs[:, t, :] = prediction
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = trg[:, t + 1] if teacher_force else top1
        
        return outputs
    
    @torch.no_grad()
    def translate(
        self,
        src: torch.Tensor,
        max_len: int = 50,
        sos_idx: int = 1,
        eos_idx: int = 2
    ) -> torch.Tensor:
        """
        Generate translation using greedy decoding.
        
        Args:
            src: [batch, src_len]
            max_len: maximum output length
            sos_idx: start-of-sequence token index
            eos_idx: end-of-sequence token index
        Returns:
            generated: [batch, generated_len]
        """
        self.eval()
        batch_size = src.shape[0]
        
        encoder_outputs, (hidden, cell) = self.encoder(src)
        hidden, cell = self._transform_encoder_state(hidden, cell)
        
        input_token = torch.full((batch_size,), sos_idx, device=self.device)
        generated = [input_token]
        
        for _ in range(max_len):
            prediction, hidden, cell = self.decoder(
                input_token, hidden, cell, encoder_outputs
            )
            
            top1 = prediction.argmax(1)
            generated.append(top1)
            
            if (top1 == eos_idx).all():
                break
            
            input_token = top1
        
        return torch.stack(generated, dim=1)
