"""Training Pipeline for Seq2Seq with Attention."""
import torch
import torch.nn as nn
import torch.optim as optim
from models import Encoder, Decoder, LuongAttention, Seq2Seq


def train_one_epoch(
    model: Seq2Seq,
    data_loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    clip: float,
    device: torch.device
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    
    for src, trg in data_loader:
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)


def evaluate(
    model: Seq2Seq,
    data_loader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Evaluate the model. Returns average loss."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, trg in data_loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)
            
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)


if __name__ == "__main__":
    # Hyperparameters
    VOCAB_SIZE = 5000
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    SRC_LEN = 20
    TRG_LEN = 15
    NUM_BATCHES = 10
    CLIP = 1.0
    PAD_IDX = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    encoder = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS)
    attention = LuongAttention(HIDDEN_DIM * 2, HIDDEN_DIM)
    decoder = Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, attention)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")
    
    # Dummy data
    dummy_data = [
        (
            torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SRC_LEN)),
            torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, TRG_LEN))
        )
        for _ in range(NUM_BATCHES)
    ]
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    print("\n--- Training ---")
    for epoch in range(3):
        train_loss = train_one_epoch(model, dummy_data, optimizer, criterion, CLIP, device)
        eval_loss = evaluate(model, dummy_data, criterion, device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f}")
    
    # Test inference
    print("\n--- Inference ---")
    src = torch.randint(1, VOCAB_SIZE, (2, 10))
    translations = model.translate(src.to(device), max_len=15)
    print(f"Translation shape: {translations.shape}")
    
    print("\nAll tests passed.")