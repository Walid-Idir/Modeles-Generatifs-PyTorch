"""
TP2 - Modèles Génératifs avec PyTorch
Première Partie :
1. Chargement et exploration du dataset MNIST
2. Création, entraînement et évaluation du modèle discriminatif (Classifier)
3. Création, entraînement, évaluation et exploration d'un Auto-encodeur simple (AE)

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
from torcheval.metrics import MulticlassAccuracy

# -------------------------------------------------------------
# 0. Configuration de l'environnement & Device
# -------------------------------------------------------------
print("=" * 60)
print("0. Configuration de l'environnement PyTorch")
print("=" * 60)
print("PyTorch Version :", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé   :", device)

os.makedirs("./figures", exist_ok=True)
os.makedirs("./models", exist_ok=True)

# -------------------------------------------------------------
# 1. Chargement du dataset MNIST
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("1. Chargement et exploration du dataset MNIST")
print("=" * 60)

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

print(f"Taille du jeu d'entraînement : {len(train_dataset)} images (28x28)")
print(f"Taille du jeu de test         : {len(test_dataset)} images (28x28)")

# Visualisation d'un batch d'exemples
images, labels = next(iter(train_loader))
fig, axes = plt.subplots(2, 5, figsize=(8, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap="gray")
    ax.set_title(f"Label = {labels[i].item()}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("./figures/mnist_samples.png", dpi=150)
plt.close()
print("Exemples MNIST sauvegardés dans './figures/mnist_samples.png'")

# -------------------------------------------------------------
# 2. Modèle Discriminatif (Classification : P(y|x))
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Modèle Discriminatif : Classifier (P(y|x))")
print("=" * 60)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

classifier = Classifier().to(device)
print(classifier)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

def train_classifier(model, loader, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss = {avg_loss:.4f}")

print("\n--- Entraînement du Classifieur ---")
train_classifier(classifier, train_loader, epochs=3)

# Évaluation manuelle standard
def evaluate_classifier_manual(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

accuracy_manual = evaluate_classifier_manual(classifier, test_loader)
print(f"Accuracy (Méthode manuelle) : {100 * accuracy_manual:.2f}%")

# Évaluation avec TorchEval (comme le prof)
@torch.no_grad()
def evaluate_classifier_torcheval(model, loader):
    model.eval()
    dev = next(model.parameters()).device
    metric = MulticlassAccuracy(device=dev)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        metric.update(logits, y)
    return metric.compute().item()

accuracy_torcheval = evaluate_classifier_torcheval(classifier, test_loader)
print(f"Accuracy (TorchEval)       : {100 * accuracy_torcheval:.2f}%")

torch.save(classifier.state_dict(), "./models/classifier.pth")
print("Modèle classifieur sauvegardé dans './models/classifier.pth'")

# -------------------------------------------------------------
# 3. Auto-encodeur Simple (Autoencoder AE)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Auto-encodeur Simple (AE)")
print("=" * 60)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat.view(-1, 1, 28, 28)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

ae = Autoencoder(latent_dim=2).to(device)
print(ae)

ae_criterion = nn.MSELoss()
ae_optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

def train_autoencoder(model, loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, _ in loader:
            x = x.to(device)
            x_hat = model(x)
            loss = ae_criterion(x_hat, x)

            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Reconstruction Loss = {avg_loss:.4f}")

print("\n--- Entraînement de l'Auto-encodeur ---")
train_autoencoder(ae, train_loader, epochs=5)

torch.save(ae.state_dict(), "./models/autoencoder.pth")
print("Modèle auto-encodeur sauvegardé dans './models/autoencoder.pth'")

# 3.1 Visualisation des reconstructions
ae.eval()
x_test, _ = next(iter(test_loader))
x_test = x_test[:10].to(device)

with torch.no_grad():
    x_hat = ae(x_test)

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(x_test[i].cpu().squeeze(), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(x_hat[i].cpu().squeeze(), cmap="gray")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=12)
axes[1, 0].set_ylabel("Reconstruit", fontsize=12)
plt.suptitle("Comparaison Original vs Reconstitution (Auto-encodeur)", fontsize=14)
plt.tight_layout()
plt.savefig("./figures/ae_reconstructions.png", dpi=150)
plt.close()
print("Reconstructions sauvegardées dans './figures/ae_reconstructions.png'")

# 3.2 Visualisation de l'espace latent 2D
print("\nProjection de l'espace latent 2D...")
ae.eval()
zs = []
ys = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        z = ae.encode(x)
        zs.append(z.cpu())
        ys.append(y)

Z = torch.cat(zs).numpy()
Y = torch.cat(ys).numpy()

plt.figure(figsize=(9, 7))
scatter = plt.scatter(Z[:, 0], Z[:, 1], c=Y, s=5, cmap="tab10", alpha=0.7)
plt.xlabel("z1", fontsize=12)
plt.ylabel("z2", fontsize=12)
plt.title("Espace latent 2D appris par l'Auto-encodeur classique", fontsize=14)
cbar = plt.colorbar(scatter, ticks=range(10))
cbar.set_label("Chiffres (Labels MNIST)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("./figures/ae_latent_space.png", dpi=150)
plt.close()
print("Espace latent sauvegardé dans './figures/ae_latent_space.png'")

# 3.3 Génération à partir d'un chiffre connu (barycentre latent)
@torch.no_grad()
def generer_chiffre(model, dataloader, chiffre):
    if chiffre not in range(10):
        raise ValueError("Le chiffre doit être compris entre 0 et 9.")

    dev = next(model.parameters()).device
    model.eval()

    somme_z = None
    nombre = 0

    for imgs, lbls in dataloader:
        mask = (lbls == chiffre)
        imgs_c = imgs[mask]
        if imgs_c.shape[0] == 0:
            continue

        z = model.encode(imgs_c.to(dev))
        if somme_z is None:
            somme_z = z.sum(dim=0, keepdim=True)
        else:
            somme_z += z.sum(dim=0, keepdim=True)
        nombre += z.shape[0]

    if nombre == 0:
        raise ValueError(f"Aucune image trouvée pour le chiffre {chiffre}.")

    z_moyen = somme_z / nombre
    return model.decode(z_moyen).cpu()

fig, axes = plt.subplots(1, 10, figsize=(15, 2.5))
for c in range(10):
    img_gen = generer_chiffre(ae, train_loader, chiffre=c)
    axes[c].imshow(img_gen[0, 0].numpy(), cmap="gray")
    axes[c].set_title(f"Chiffre {c}")
    axes[c].axis("off")
plt.suptitle("Génération à partir du centre latent moyen de chaque chiffre", fontsize=14)
plt.tight_layout()
plt.savefig("./figures/ae_generated_digits.png", dpi=150)
plt.close()
print("Chiffres générés par barycentre sauvegardés dans './figures/ae_generated_digits.png'")

print("\n" + "=" * 60)
print("Première Partie terminée avec succès !")
print("=" * 60)
