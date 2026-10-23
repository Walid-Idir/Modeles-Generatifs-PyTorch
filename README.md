# Modèles Génératifs avec PyTorch 🧠🔥

Ce dépôt contient les notebooks complets pour la mise en œuvre, l'entraînement, l'évaluation et l'exploration des modèles discriminatifs et génératifs (**Auto-encodeur** et **Variational Autoencoder - VAE**) sur le dataset **MNIST** en utilisant **PyTorch**.

---

## 📋 Contenu du Projet

1. **`TP2_Partie1_MNIST_Classifieur_Autoencodeur.ipynb`** :
   - **Préparation de l'environnement** : Gestion du `device` (CPU / GPU CUDA).
   - **Jeu de données MNIST** : Chargement, normalisation (`transforms.ToTensor()`) et exploration par lots.
   - **Modèle Discriminatif (Classifier)** :
     - Architecture MLP (`784 -> 128 -> 10`).
     - Fonction de perte `nn.CrossEntropyLoss` et optimiseur `Adam`.
     - Entraînement et double évaluation : méthode manuelle et métrique officielle `torcheval.metrics.MulticlassAccuracy` (Accuracy ~96.3%).
     - Analyse du rôle discriminatif : apprentissage de $P(y|x)$ sans capacité de génération d'images.
   - **Auto-encodeur Simple (AE)** :
     - Réduction de dimension vers un espace latent 2D (`latent_dim = 2`).
     - Encodeur et Décodeur non supervisés ($x \to z \to \hat{x}$).
     - Perte de reconstruction `nn.MSELoss`.
     - Visualisation des reconstructions et projection 2D de l'espace latent.
     - Génération de chiffres à partir des barycentres de chaque classe.
     - Analyse des limites de l'auto-encodeur classique (espace discontinu avec zones vides).

2. **`TP2_Partie2_VAE.ipynb`** :
   - **Principe du Variational Autoencoder (VAE)** : Modélisation probabiliste de l'espace latent $q(z|x) = \mathcal{N}(\mu, \sigma^2)$.
   - **Reparameterization Trick** :
     $$z = \mu + \sigma \odot \epsilon \quad \text{avec} \quad \epsilon \sim \mathcal{N}(0, I)$$
   - **Architecture du VAE** : Encodeur avec têtes parallèles $\mu$ et $\log \sigma^2$ et décodeur avec `Sigmoid`.
   - **Fonction de Perte (ELBO)** : Combinaison de la perte de reconstruction (BCE) et de la divergence de Kullback-Leibler ($D_{\text{KL}}$ analytique).
   - **Visualisation & Reconstructions** : Comparaison images réelles vs reconstructions VAE.
   - **Espace Latent Régularisé 2D** : Projection $(\mu_1, \mu_2)$ centrée en $(0, 0)$ et continue.
   - **Génération Pure** : Échantillonnage aléatoire $z \sim \mathcal{N}(0, I)$ et décodage direct en nouveaux chiffres manuscrits.
   - **Exploration Continue de la Variété Latente** : Mosaïque 2D sur une grille $15 \times 15$ démontrant des transitions douces et continues entre les chiffres.

3. **Environnement & Dépendances** :
   - `requirements.txt` : Liste des bibliothèques nécessaires à l'exécution des notebooks.
   - `.gitignore` : Configuration excluant les données brutes (`data/`), l'environnement virtuel (`.venv/`), les caches et les fichiers de poids binaires (`*.pt`, `*.pth`).

---

## 🚀 Installation & Prérequis

### 1. Cloner le dépôt
```bash
git clone https://github.com/Walid-Idir/Modeles-Generatifs-PyTorch.git
cd Modeles-Generatifs-PyTorch
```

### 2. Installer les dépendances
Installez les packages nécessaires listés dans `requirements.txt` :
```bash
pip install -r requirements.txt
```

*Ou en utilisant UV :*
```bash
uv venv
uv add ipykernel torch torchvision matplotlib torcheval
```

---

## 💻 Exécution du Notebook

Lancez Jupyter Notebook ou Jupyter Lab :
```bash
jupyter notebook
```

Ouvrez ensuite les fichiers `TP2_Partie1_MNIST_Classifieur_Autoencodeur.ipynb` et `TP2_Partie2_VAE.ipynb` et exécutez les cellules. L'entraînement des modèles et la sauvegarde locale des poids (`.pt` / `.pth`) s'effectueront directement lors de l'exécution.

---

## 👤 Auteur
- **Walid Idir** - [GitHub](https://github.com/Walid-Idir)
