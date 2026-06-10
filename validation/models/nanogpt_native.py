import torch
import torch.nn as nn
import torch.nn.functional as F
from validate import Validator
import math

class NanoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['n_embd']),
            wpe = nn.Embedding(config['block_size'], config['n_embd']),
            h = nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),
            ln_f = nn.LayerNorm(config['n_embd']),
        ))
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'])
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size']))
                                     .view(1, 1, config['block_size'], config['block_size']))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config['n_embd'], 4 * config['n_embd'])
        self.c_proj  = nn.Linear(4 * config['n_embd'], config['n_embd'])

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

def main():
    config = dict(n_layer=2, n_head=2, n_embd=32, block_size=64, vocab_size=128)
    model = NanoGPT(config)
    model.eval()
    torch.manual_seed(42)
    idx = torch.randint(0, config['vocab_size'], (1, 8))
    with torch.no_grad():
        logits = model(idx)
    validator = Validator("nanogpt")
    validator.save_native_output({
        "input": idx,
        "logits": logits,
        "params": {name: p for name, p in model.state_dict().items()}
    })
    print("NanoGPT native output saved.")

if __name__ == "__main__":
    main()
