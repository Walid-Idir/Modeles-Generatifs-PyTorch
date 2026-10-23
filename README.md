# Modèles Génératifs avec PyTorch sur MNIST (TP2)

Ce projet implémente, entraîne, évalue et explore différents paradigmes d'apprentissage sur le dataset **MNIST** en utilisant **PyTorch** et **UV** :
1. **Modèle Discriminatif** : Classification supervisée (apprentissage de $P(y|x)$).
2. **Auto-encodeur Simple (AE)** : Réduction de dimension non supervisée vers un espace latent 2D ($x \to z \to \hat{x}$).
3. **Variational Autoencoder (VAE)** : Modèle génératif probabiliste avec *Reparameterization Trick*, régularisation par divergence de Kullback-Leibler et interpolation continue le long de la variété latente 2D.

Projet basé sur le cours du **Pr. Mohamed YOUSSFI**.

---

## 📂 Structure du Répertoire

```text
├── TP2_Partie1_MNIST_Classifieur_Autoencodeur.ipynb   # Notebook Partie 1 : MNIST, Classifieur et Auto-encodeur 2D
├── TP2_Partie2_VAE.ipynb                              # Notebook Partie 2 : Variational Autoencoder (VAE) complet
├── tp2_partie1.py                                     # Script Python exécutable (Partie 1)
├── tp2_partie2_vae.py                                 # Script Python exécutable (Partie 2 VAE)
├── pyproject.toml                                     # Configuration du projet UV
├── uv.lock                                            # Fichier de verrouillage des dépendances
├── figures/                                           # Visualisations et graphiques générés
│   ├── mnist_samples.png                              # Échantillons du dataset MNIST
│   ├── ae_reconstructions.png                         # Reconstructions de l'Auto-encodeur
│   ├── ae_latent_space.png                            # Espace latent 2D de l'Auto-encodeur
│   ├── ae_generated_digits.png                        # Chiffres générés par barycentre latent
│   ├── vae_reconstructions.png                        # Reconstructions du VAE
│   ├── vae_latent_space.png                           # Espace latent régularisé 2D du VAE
│   ├── vae_generated_samples.png                      # Échantillonnage pur z ~ N(0, I)
│   └── vae_latent_manifold.png                        # Variété 2D continue (grille 15x15)
└── models/                                            # Poids des modèles sauvegardés (.pth)
    ├── classifier.pth
    ├── autoencoder.pth
    └── vae.pth
```

---

## 🛠️ Installation et Environnement avec UV

Le projet utilise **UV** et **Python 3.11** pour une gestion rapide et reproductible de l'environnement :

```bash
# 1. Cloner le repository
git clone https://github.com/Walid-Idir/Modeles-Generatifs-PyTorch.git
cd Modeles-Generatifs-PyTorch

# 2. Créer l'environnement virtuel avec UV
uv venv

# 3. Activer l'environnement :
# Windows (PowerShell) :
.venv\Scripts\Activate.ps1
# Linux / macOS :
source .venv/bin/activate

# 4. Installer les dépendances :
uv add ipykernel torch torchvision matplotlib torcheval
```

---

## 📊 Détails des Notebooks et Résultats

### 1️⃣ Notebook 1 : `TP2_Partie1_MNIST_Classifieur_Autoencodeur.ipynb`
- **Modèle Discriminatif (Classifier)** :
  - Architecture MLP : `784 -> 128 (ReLU) -> 10`
  - Perte : `nn.CrossEntropyLoss()`, Optimiseur : `Adam(lr=1e-3)`
  - Accuracy sur le test set : **96.31%** (validée par méthode standard et `torcheval.metrics.MulticlassAccuracy`).
  - *Observation* : Le modèle apprend $P(y|x)$ pour discriminer les classes mais ne peut pas générer de nouvelles images.
- **Auto-encodeur Simple (AE)** :
  - Espace latent 2D (`latent_dim = 2`) : Encodeur `784 -> 128 -> 32 -> 2`, Décodeur `2 -> 32 -> 128 -> 784` (`Sigmoid`).
  - Perte MSE sur les pixels.
  - *Visualisation* : Projection 2D du test set et calcul des barycentres pour chaque chiffre.
  - *Limite* : L'espace latent n'est pas régularisé et présente des zones vides (trous), empêchant un échantillonnage aléatoire réaliste.

### 2️⃣ Notebook 2 : `TP2_Partie2_VAE.ipynb` (Fichier Dédié VAE)
- **Modélisation Probabiliste** : L'encodeur prédit les paramètres d'une distribution normale $\mu(x)$ et $\log \sigma^2(x)$.
- **Reparameterization Trick** :
  $$z = \mu + \sigma \odot \epsilon \quad \text{avec} \quad \epsilon \sim \mathcal{N}(0, I)$$
- **Fonction de Perte (ELBO)** :
  $$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{rec}} + \beta D_{\text{KL}}$$
  où $\mathcal{L}_{\text{rec}}$ est la Binary Cross-Entropy (BCE) et $D_{\text{KL}}$ est la divergence de Kullback-Leibler analytique forçant $q(z|x) \approx \mathcal{N}(0, I)$.
- **Génération & Exploration** :
  - **Génération pure** sans image d'entrée par tirage direct $z \sim \mathcal{N}(0, I)$.
  - **Exploration continue de la variété latente 2D** : cartographie complète sur une grille $15 \times 15$ démontrant des transitions fluides entre chiffres.

---

## 💻 Exécution en ligne de commande

Vous pouvez également lancer directement les scripts Python équivalents :

```bash
# Partie 1 : Modèle discriminatif + Auto-encodeur simple
python tp2_partie1.py

# Partie 2 : Variational Autoencoder (VAE)
python tp2_partie2_vae.py
```
