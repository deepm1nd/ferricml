import torch
import torch.nn as nn
import torch.nn.functional as F
from validate import Validator

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    model = LeNet5()
    model.eval()
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        output = model(dummy_input)
    validator = Validator("lenet5")
    validator.save_native_output({
        "input": dummy_input,
        "output": output,
        "params": {name: p for name, p in model.state_dict().items()}
    })
    print("LeNet-5 native output saved.")

if __name__ == "__main__":
    main()
