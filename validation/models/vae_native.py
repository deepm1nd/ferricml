import torch
import torch.nn as nn
from validate import Validator

class TinyVAE(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=16, latent_dim=8):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def encode(self, x):
        h = torch.relu(self.encoder(x))
        return self.fc_mu(h), self.fc_logvar(h)

def main():
    model = TinyVAE()
    model.eval()
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 64)
    with torch.no_grad():
        mu, logvar = model.encode(dummy_input)
        # Verify z = mu + eps * exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.ones_like(std) * 0.1
        z = mu + eps * std
    validator = Validator("vae")
    validator.save_native_output({
        "input": dummy_input,
        "output": z,
        "mu": mu,
        "logvar": logvar,
        "params": {name: p for name, p in model.state_dict().items()}
    })
    print("Tiny VAE native output saved.")

if __name__ == "__main__":
    main()
