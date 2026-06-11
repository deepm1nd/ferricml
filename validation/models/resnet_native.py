import torch
import torch.nn as nn
from validate import Validator

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

def main():
    model = ResNetBlock(16, 16)
    model.eval()
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 16, 14, 14)
    with torch.no_grad():
        output = model(dummy_input)
    validator = Validator("resnet")
    validator.save_native_output({
        "input": dummy_input,
        "output": output,
        "params": {name: p for name, p in model.state_dict().items()}
    })
    print("ResNet block native output saved.")

if __name__ == "__main__":
    main()
