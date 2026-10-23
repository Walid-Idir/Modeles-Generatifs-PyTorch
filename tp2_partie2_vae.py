"""
TP2 - Modèles Génératifs avec PyTorch
Deuxième Fichier : Variational Autoencoder (VAE)
4. Création, entraînement, évaluation et exploration complète d'un VAE

Basé sur le cours et travaux pratiques du Pr. Mohamed YOUSSFI.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------
# 0. Configuration de l'environnement & Device
# -------------------------------------------------------------
print("=" * 60)
print("TP2 Partie 2 : Variational Autoencoder (VAE)")
print("=" * 60)
print("PyTorch Version :", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé   :", device)

os.makedirs("./figures", exist_ok=True)
os.makedirs("./models", exist_ok=True)

# -------------------------------------------------------------
# 1. Chargement des données MNIST
# -------------------------------------------------------------
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Jeu d'entraînement : {len(train_dataset)} images")
print(f"Jeu de test         : {len(test_dataset)} images")

# -------------------------------------------------------------
# 2. Architecture du VAE (Reparameterization Trick)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Définition de l'Architecture VAE")
print("=" * 60)

class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        # Encodeur : extrait les représentations cachées
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        # Deux couches linéaires en parallèle pour prédire moyenne et log-variance
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Décodeur : reconstruit l'image depuis l'espace latent z
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization Trick:
        z = mu + std * epsilon, avec epsilon ~ N(0, I)
        std = exp(0.5 * logvar)
        Permet de propager le gradient à travers l'échantillonnage stochastique.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat.view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

vae = VAE(latent_dim=2).to(device)
print(vae)

# -------------------------------------------------------------
# 3. Fonction de Perte du VAE (ELBO : Reconstruction + KL Divergence)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Fonction de Perte du VAE")
print("=" * 60)

def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    """
    L_VAE = L_rec + beta * D_KL
    - L_rec : Binary Cross-Entropy (BCE) sur les pixels
    - D_KL : Divergence de Kullback-Leibler entre q(z|x) et la loi a priori N(0, I)
    """
    rec_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = rec_loss + beta * kl_loss
    return total_loss, rec_loss, kl_loss

vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# -------------------------------------------------------------
# 4. Entraînement du VAE
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("4. Entraînement du VAE")
print("=" * 60)

def train_vae(model, loader, epochs=5, beta=1.0):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_rec = 0.0
        total_kl = 0.0

        for x, _ in loader:
            x = x.to(device)

            x_hat, mu, logvar = model(x)
            loss, rec, kl = vae_loss(x_hat, x, mu, logvar, beta=beta)

            vae_optimizer.zero_grad()
            loss.backward()
            vae_optimizer.step()

            total_loss += loss.item()
            total_rec += rec.item()
            total_kl += kl.item()

        n = len(loader.dataset)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Loss = {total_loss/n:.2f} | "
              f"Rec = {total_rec/n:.2f} | "
              f"KL = {total_kl/n:.2f}")

train_vae(vae, train_loader, epochs=5, beta=1.0)
torch.save(vae.state_dict(), "./models/vae.pth")
print("Modèle VAE sauvegardé dans './models/vae.pth'")

# -------------------------------------------------------------
# 5. Visualisation des Reconstructions avec le VAE
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("5. Visualisation des Reconstructions VAE")
print("=" * 60)

vae.eval()
x_test, _ = next(iter(test_loader))
x_test = x_test[:10].to(device)

with torch.no_grad():
    x_hat, _, _ = vae(x_test)

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(x_test[i].cpu().squeeze(), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(x_hat[i].cpu().squeeze(), cmap="gray")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=12)
axes[1, 0].set_ylabel("VAE", fontsize=12)
plt.suptitle("Reconstruction d'images MNIST par le VAE", fontsize=14)
plt.tight_layout()
plt.savefig("./figures/vae_reconstructions.png", dpi=150)
plt.close()
print("Reconstructions sauvegardées dans './figures/vae_reconstructions.png'")

# -------------------------------------------------------------
# 6. Visualisation de l'Espace Latent du VAE
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("6. Visualisation de l'Espace Latent 2D (mu1, mu2)")
print("=" * 60)

vae.eval()
mus = []
ys = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        mu, logvar = vae.encode(x)
        mus.append(mu.cpu())
        ys.append(y)

MU = torch.cat(mus).numpy()
Y = torch.cat(ys).numpy()

plt.figure(figsize=(9, 7))
scatter = plt.scatter(MU[:, 0], MU[:, 1], c=Y, s=5, cmap="tab10", alpha=0.7)
plt.xlabel("mu1 (Dimension latente 1)", fontsize=12)
plt.ylabel("mu2 (Dimension latente 2)", fontsize=12)
plt.title("Espace latent régularisé du VAE (centré en 0)", fontsize=14)
cbar = plt.colorbar(scatter, ticks=range(10))
cbar.set_label("Chiffres (Labels MNIST)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("./figures/vae_latent_space.png", dpi=150)
plt.close()
print("Espace latent VAE sauvegardé dans './figures/vae_latent_space.png'")

# -------------------------------------------------------------
# 7. Génération de Nouvelles Images (Échantillonnage z ~ N(0, I))
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("7. Génération pure : Échantillonnage z ~ N(0, I)")
print("=" * 60)

vae.eval()
with torch.no_grad():
    z_random = torch.randn(20, 2).to(device)
    generated_samples = vae.decode(z_random).cpu()

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated_samples[i].squeeze(), cmap="gray")
    ax.axis("off")

plt.suptitle("Nouveaux chiffres générés par le VAE (z ~ N(0, I))", fontsize=14)
plt.tight_layout()
plt.savefig("./figures/vae_generated_samples.png", dpi=150)
plt.close()
print("Nouvelles images générées sauvegardées dans './figures/vae_generated_samples.png'")

# -------------------------------------------------------------
# 8. Exploration Continue de l'Espace Latent (Manifold 2D)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("8. Exploration continue de la Variété Latente 2D")
print("=" * 60)

vae.eval()
grid_x = np.linspace(-3, 3, 15)
grid_y = np.linspace(-3, 3, 15)

figure = np.zeros((28 * 15, 28 * 15))

with torch.no_grad():
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z = torch.tensor([[xi, yi]], dtype=torch.float32, device=device)
            x_hat = vae.decode(z)
            digit = x_hat.cpu().numpy()[0, 0]
            figure[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap="gray")
plt.axis("off")
plt.title("Exploration continue de l'espace latent du VAE (Grille 15x15)", fontsize=14)
plt.tight_layout()
plt.savefig("./figures/vae_latent_manifold.png", dpi=150)
plt.close()
print("Mosaïque de l'espace latent sauvegardée dans './figures/vae_latent_manifold.png'")

print("\n" + "=" * 60)
print("Deuxième Fichier (VAE) terminé avec succès !")
print("=" * 60)
