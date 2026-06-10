import torch
import torch.nn as nn
from validate import Validator

class TinyVAE(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=16, latent_dim=8):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = torch.relu(self.encoder(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def main():
    model = TinyVAE()
    model.eval()
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 64)
    with torch.no_grad():
        recon, mu, logvar = model(dummy_input)
    validator = Validator("vae")
    validator.save_native_output({
        "input": dummy_input,
        "recon": recon,
        "mu": mu,
        "logvar": logvar,
        "params": {name: p for name, p in model.state_dict().items()}
    })
    print("Tiny VAE native output saved.")

if __name__ == "__main__":
    main()
