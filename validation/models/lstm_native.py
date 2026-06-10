import torch
import torch.nn as nn
from validate import Validator

class TinyLSTM(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return out

def main():
    model = TinyLSTM()
    model.eval()
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 8, 16) # B, T, C
    with torch.no_grad():
        output = model(dummy_input)
    validator = Validator("lstm")
    validator.save_native_output({
        "input": dummy_input,
        "output": output,
        "params": {name: p for name, p in model.state_dict().items()}
    })
    print("Tiny LSTM native output saved.")

if __name__ == "__main__":
    main()
