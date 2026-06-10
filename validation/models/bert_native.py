import torch
import torch.nn as nn
from validate import Validator

class TinyBERT(nn.Module):
    def __init__(self, vocab_size=128, hidden_size=32, num_heads=2):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            "word_embeddings": nn.Embedding(vocab_size, hidden_size),
            "position_embeddings": nn.Embedding(64, hidden_size),
            "LayerNorm": nn.LayerNorm(hidden_size),
        })
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_heads, dim_feedforward=128,
                dropout=0.0, batch_first=True, activation='gelu'
            )
        ])

    def forward(self, x):
        words = self.embeddings["word_embeddings"](x)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        positions = self.embeddings["position_embeddings"](pos)
        x = words + positions
        x = self.embeddings["LayerNorm"](x)
        for layer in self.encoder:
            x = layer(x)
        return x

def main():
    model = TinyBERT()
    model.eval()
    torch.manual_seed(42)
    idx = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        output = model(idx)
    validator = Validator("bert")
    validator.save_native_output({
        "input": idx,
        "output": output,
        "params": {name: p for name, p in model.state_dict().items()}
    })
    print("Tiny BERT native output saved.")

if __name__ == "__main__":
    main()
