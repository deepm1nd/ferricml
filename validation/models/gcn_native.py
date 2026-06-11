import torch
import torch.nn as nn
from validate import Validator

class TinyGCN(nn.Module):
    def __init__(self, in_features=16, out_features=16):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: (N, in_features)
        # adj: (N, N) sparse-ish
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        return output + self.bias

def main():
    model = TinyGCN()
    model.eval()
    torch.manual_seed(42)
    x = torch.randn(8, 16)
    adj = torch.eye(8) # Simple adj matrix
    with torch.no_grad():
        output = model(x, adj)
    validator = Validator("gcn")
    validator.save_native_output({
        "input": x,
        "adj": adj,
        "output": output,
        "params": {name: p for name, p in model.state_dict().items()}
    })
    print("Tiny GCN native output saved.")

if __name__ == "__main__":
    main()
