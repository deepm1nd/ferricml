import torch
import torch.nn as nn
import torch.nn.functional as F
from validate import Validator
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class TinyLlama(nn.Module):
    def __init__(self, vocab_size=128, dim=32, n_heads=2, n_kv_heads=1):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.attention_norm = RMSNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, (dim // n_heads) * n_kv_heads, bias=False)
        self.wv = nn.Linear(dim, (dim // n_heads) * n_kv_heads, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = RMSNorm(dim)
        self.w1 = nn.Linear(dim, 64, bias=False)
        self.w2 = nn.Linear(dim, 64, bias=False)
        self.w3 = nn.Linear(64, dim, bias=False)

    def forward(self, x):
        h = self.tok_embeddings(x)
        norm_h = self.ffn_norm(h)
        ffn = self.w3(F.silu(self.w1(norm_h)) * self.w2(norm_h))
        return h + ffn

def main():
    model = TinyLlama()
    model.eval()
    torch.manual_seed(42)
    idx = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        output = model(idx)
    validator = Validator("tinyllama")
    validator.save_native_output({
        "input": idx,
        "output": output,
        "params": {name: p for name, p in model.state_dict().items()}
    })
    print("Tiny Llama native output saved.")

if __name__ == "__main__":
    main()
