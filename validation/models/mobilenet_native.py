import torch
import torch.nn as nn
from validate import Validator

class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super().__init__()
        mid_channels = in_channels * expansion
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.use_res_connect = stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

def main():
    model = MobileNetBlock(16, 16)
    model.eval()
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 16, 14, 14)
    with torch.no_grad():
        output = model(dummy_input)
    validator = Validator("mobilenet")
    validator.save_native_output({
        "input": dummy_input,
        "output": output,
        "params": {name: p for name, p in model.state_dict().items()}
    })
    print("MobileNet block native output saved.")

if __name__ == "__main__":
    main()
