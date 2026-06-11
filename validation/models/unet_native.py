import torch
import torch.nn as nn
from validate import Validator

class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down = nn.Conv2d(in_channels, 16, 3, stride=2, padding=1)
        self.up = nn.ConvTranspose2d(16, out_channels, 4, stride=2, padding=1)

    def forward(self, x):
        x = torch.relu(self.down(x))
        x = torch.sigmoid(self.up(x))
        return x

def main():
    model = TinyUNet()
    model.eval()
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    validator = Validator("unet")
    validator.save_native_output({
        "input": dummy_input,
        "output": output,
        "params": {name: p for name, p in model.state_dict().items()}
    })
    print("Tiny U-Net native output saved.")

if __name__ == "__main__":
    main()
