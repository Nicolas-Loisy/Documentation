# Module 9 : Apprentissage Non Supervisé

## 📋 Table des Matières
1. [Introduction](#introduction)
2. [Clustering](#clustering)
3. [Réduction de Dimensionnalité](#réduction-de-dimensionnalité)
4. [Autoencodeurs](#autoencodeurs)
5. [Détection d'Anomalies](#détection-danomalies)
6. [Systèmes de Recommandation](#systèmes-de-recommandation)
7. [Introduction à l'Apprentissage par Renforcement](#introduction-à-lapprentissage-par-renforcement)
8. [Projets Pratiques](#projets-pratiques)
9. [Résumé](#résumé)

---

## Introduction

L'**apprentissage non supervisé** consiste à découvrir des structures cachées dans des données **non labelisées**.

### Différence avec l'Apprentissage Supervisé

| Aspect | Supervisé | Non Supervisé |
|--------|-----------|---------------|
| **Données** | $\{(\mathbf{x}_i, y_i)\}$ (avec labels) | $\{\mathbf{x}_i\}$ (sans labels) |
| **Objectif** | Prédire $y$ | Découvrir structure |
| **Exemples** | Classification, régression | Clustering, réduction dim. |

### Applications

| Domaine | Tâche | Exemple |
|---------|-------|---------|
| **Segmentation client** | Grouper clients similaires | Marketing ciblé |
| **Compression** | Réduire dimensionnalité | Images, données |
| **Détection anomalies** | Identifier outliers | Fraude, défauts |
| **Recommandation** | Produits similaires | Netflix, Amazon |
| **Visualisation** | Projeter données 2D/3D | t-SNE, UMAP |

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
```

---

## Clustering

Le **clustering** consiste à regrouper des données similaires en **clusters** (groupes).

### Objectif

Partitionner $\mathcal{D} = \{\mathbf{x}_1, \ldots, \mathbf{x}_N\}$ en $K$ clusters tels que :
- **Intra-cluster** : Points dans même cluster sont similaires
- **Inter-cluster** : Points de clusters différents sont dissimilaires

### 1. K-Means

**Algorithme le plus populaire** pour clustering.

#### Principe

1. Initialiser $K$ centroids aléatoirement
2. **Assignment** : Assigner chaque point au centroid le plus proche
3. **Update** : Recalculer les centroids comme moyenne des points assignés
4. Répéter 2-3 jusqu'à convergence

#### Formule Mathématique

**Objectif** : Minimiser l'inertie (somme des distances au carré)

$$
J = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
$$

où $\boldsymbol{\mu}_k$ est le centroid du cluster $k$.

#### Implémentation

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Générer données
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)

# Visualisation
plt.figure(figsize=(14, 5))

# Vraies classes
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.6, edgecolors='k')
plt.title('Vraies Classes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Clustering K-Means
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.6, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='red', marker='X', edgecolors='black', linewidths=2, label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Inertia: {kmeans.inertia_:.2f}")
```

#### Choisir $K$ : Méthode du Coude (Elbow Method)

```python
# Tester différentes valeurs de K
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Visualisation
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Nombre de Clusters (K)')
plt.ylabel('Inertie')
plt.title('Méthode du Coude pour Choisir K')
plt.grid(True, alpha=0.3)
plt.show()

print("Chercher le 'coude' (elbow) dans la courbe")
```

#### Limitations de K-Means

- **Suppose clusters sphériques** et de taille similaire
- **Sensible à l'initialisation**
- **Doit choisir $K$ a priori**
- **Sensible aux outliers**

```python
# Exemple : K-Means échoue sur données non-sphériques
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

kmeans_moons = KMeans(n_clusters=2, random_state=42)
y_pred_moons = kmeans_moons.fit_predict(X_moons)

plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_pred_moons, cmap='viridis',
            s=50, alpha=0.6, edgecolors='k')
plt.scatter(kmeans_moons.cluster_centers_[:, 0], kmeans_moons.cluster_centers_[:, 1],
            s=300, c='red', marker='X', edgecolors='black', linewidths=2)
plt.title('K-Means échoue sur données en forme de lunes')
plt.show()
```

### 2. DBSCAN (Density-Based Spatial Clustering)

**Principe** : Grouper points denses, identifier outliers.

#### Paramètres

- **$\varepsilon$ (eps)** : Rayon du voisinage
- **MinPts** : Nombre minimum de points pour former un cluster

#### Types de Points

- **Core point** : $\geq$ MinPts voisins dans rayon $\varepsilon$
- **Border point** : < MinPts voisins, mais dans rayon d'un core point
- **Noise point** : Ni core ni border (outlier)

#### Avantages

- **Pas besoin de spécifier $K$**
- **Détecte clusters de forme arbitraire**
- **Identifie les outliers**

```python
from sklearn.cluster import DBSCAN

# DBSCAN sur données en lunes
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_pred_dbscan = dbscan.fit_predict(X_moons)

plt.figure(figsize=(14, 5))

# K-Means (échec)
plt.subplot(1, 2, 1)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_pred_moons, cmap='viridis',
            s=50, alpha=0.6, edgecolors='k')
plt.title('K-Means (Échec)')

# DBSCAN (succès)
plt.subplot(1, 2, 2)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_pred_dbscan, cmap='viridis',
            s=50, alpha=0.6, edgecolors='k')
plt.title('DBSCAN (Succès)')

plt.tight_layout()
plt.show()

# Nombre de clusters trouvés
n_clusters = len(set(y_pred_dbscan)) - (1 if -1 in y_pred_dbscan else 0)
n_noise = list(y_pred_dbscan).count(-1)

print(f"Nombre de clusters: {n_clusters}")
print(f"Nombre d'outliers: {n_noise}")
```

### 3. Clustering Hiérarchique

**Principe** : Construire une hiérarchie de clusters.

#### Deux Approches

1. **Agglomératif** (bottom-up) : Fusion progressive
2. **Divisif** (top-down) : Division progressive

#### Dendrogramme

Visualise la hiérarchie.

```python
from scipy.cluster.hierarchy import dendrogram, linkage

# Générer données
X_hier, _ = make_blobs(n_samples=50, centers=3, random_state=42)

# Linkage (agglomératif)
linkage_matrix = linkage(X_hier, method='ward')

# Dendrogramme
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
dendrogram(linkage_matrix)
plt.xlabel('Index Échantillon')
plt.ylabel('Distance')
plt.title('Dendrogramme')

# Clustering avec seuil
from sklearn.cluster import AgglomerativeClustering

agg_clust = AgglomerativeClustering(n_clusters=3)
y_pred_agg = agg_clust.fit_predict(X_hier)

plt.subplot(1, 2, 2)
plt.scatter(X_hier[:, 0], X_hier[:, 1], c=y_pred_agg, cmap='viridis',
            s=100, alpha=0.6, edgecolors='k')
plt.title('Clustering Hiérarchique (3 clusters)')

plt.tight_layout()
plt.show()
```

### Comparaison des Méthodes

| Méthode | Avantages | Inconvénients | Usage |
|---------|-----------|---------------|-------|
| **K-Means** | Rapide, scalable | Clusters sphériques, $K$ fixe | Grandes données, clusters sphériques |
| **DBSCAN** | Formes arbitraires, outliers | Sensible à eps/MinPts | Densité variable, outliers |
| **Hiérarchique** | Pas de $K$, dendrogramme | Lent ($O(n^2)$) | Petites données, hiérarchie |

---

## Réduction de Dimensionnalité

**Objectif** : Projeter données haute dimension vers dimension réduite tout en **préservant l'information**.

### Pourquoi Réduire la Dimensionnalité ?

**Motivations** :
- **Visualisation** : Projeter en 2D/3D
- **Curse of dimensionality** : Performances dégradées en haute dim.
- **Compression** : Réduire stockage
- **Accélération** : Moins de features → entraînement plus rapide
- **Élimination de bruit**

### 1. PCA (Principal Component Analysis)

**Principe** : Trouver les directions de **variance maximale**.

#### Formulation Mathématique

Soit $\mathbf{X} \in \mathbb{R}^{N \times d}$ les données centrées.

1. **Matrice de covariance** :
   $$\mathbf{C} = \frac{1}{N} \mathbf{X}^T \mathbf{X}$$

2. **Décomposition en valeurs propres** :
   $$\mathbf{C} \mathbf{v}_i = \lambda_i \mathbf{v}_i$$

3. **Projection** sur $k$ premières composantes principales :
   $$\mathbf{Z} = \mathbf{X} \mathbf{V}_k$$

   où $\mathbf{V}_k = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$

#### Implémentation

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Charger Iris (4 features)
iris = load_iris()
X = iris.data
y = iris.target

# PCA : 4D → 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Shape originale: {X.shape}")
print(f"Shape après PCA: {X_pca.shape}")
print(f"Variance expliquée: {pca.explained_variance_ratio_}")
print(f"Variance totale expliquée: {pca.explained_variance_ratio_.sum():.2%}")

# Visualisation
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Données Originales (2 features sur 4)')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)
plt.xlabel('PC1 ({:.1%} variance)'.format(pca.explained_variance_ratio_[0]))
plt.ylabel('PC2 ({:.1%} variance)'.format(pca.explained_variance_ratio_[1]))
plt.title('Après PCA (2 composantes)')

plt.tight_layout()
plt.show()
```

#### Choisir le Nombre de Composantes

```python
# PCA complète
pca_full = PCA()
pca_full.fit(X)

# Variance cumulée
cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_)
plt.xlabel('Composante Principale')
plt.ylabel('Variance Expliquée')
plt.title('Variance par Composante')
plt.xticks(range(1, len(pca_full.explained_variance_ratio_) + 1))

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'bo-', linewidth=2)
plt.axhline(0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Nombre de Composantes')
plt.ylabel('Variance Cumulée Expliquée')
plt.title('Variance Cumulée')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Nombre de composantes pour 95% variance
n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
print(f"Composantes pour 95% variance: {n_components_95}")
```

### 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Usage** : **Visualisation** de données haute dimension.

**Principe** : Préserver les similarités locales.

**Avantages** :
- Excellente visualisation
- Capture structure non-linéaire

**Inconvénients** :
- Lent ($O(n^2)$)
- Non déterministe
- Pas pour réduction générale (seulement visualisation)

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Charger MNIST digits (64 features)
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print(f"Shape: {X_digits.shape}")  # (1797, 64)

# t-SNE : 64D → 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_digits)

# Visualisation
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10',
                      s=5, alpha=0.7)
plt.colorbar(scatter, label='Chiffre')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE sur MNIST Digits (64D → 2D)')
plt.show()
```

### 3. UMAP (Uniform Manifold Approximation and Projection)

**Alternative moderne à t-SNE** :
- **Plus rapide**
- **Préserve structure globale et locale**
- **Déterministe** (avec seed)

```python
# Installer: pip install umap-learn
import umap

# UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_digits)

# Comparaison t-SNE vs UMAP
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# t-SNE
axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10',
                s=5, alpha=0.7)
axes[0].set_title('t-SNE')
axes[0].set_xlabel('Dimension 1')
axes[0].set_ylabel('Dimension 2')

# UMAP
axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=y_digits, cmap='tab10',
                s=5, alpha=0.7)
axes[1].set_title('UMAP')
axes[1].set_xlabel('Dimension 1')
axes[1].set_ylabel('Dimension 2')

plt.tight_layout()
plt.show()
```

---

## Autoencodeurs

Les **autoencodeurs** sont des réseaux de neurones pour apprendre des **représentations compressées**.

### Architecture

```
Input (x) → Encoder → Latent Code (z) → Decoder → Reconstruction (x̂)
```

**Objectif** : Minimiser la reconstruction error
$$
\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2
$$

### Types d'Autoencodeurs

#### 1. Autoencodeur Classique

```python
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# Dimensions
input_dim = 784  # MNIST 28×28
encoding_dim = 32  # Code latent

# Encoder
encoder = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dense(64, activation='relu'),
    Dense(encoding_dim, activation='relu')
])

# Decoder
decoder = Sequential([
    Dense(64, activation='relu', input_shape=(encoding_dim,)),
    Dense(128, activation='relu'),
    Dense(input_dim, activation='sigmoid')
])

# Autoencodeur complet
autoencoder = Sequential([encoder, decoder])

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

print(autoencoder.summary())
```

#### 2. Autoencodeur Variationnel (VAE)

**Principe** : Apprendre une **distribution probabiliste** du code latent.

$$
q(\mathbf{z}|\mathbf{x}) \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)
$$

**Avantage** : Génération de nouvelles données.

```python
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K

# Sampling layer
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# VAE
latent_dim = 2

# Encoder
inputs = Input(shape=(input_dim,))
h = Dense(256, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z = Lambda(sampling)([z_mean, z_log_var])

encoder_vae = Model(inputs, [z_mean, z_log_var, z])

# Decoder
latent_inputs = Input(shape=(latent_dim,))
h_decoded = Dense(256, activation='relu')(latent_inputs)
outputs = Dense(input_dim, activation='sigmoid')(h_decoded)

decoder_vae = Model(latent_inputs, outputs)

# VAE complet
outputs_vae = decoder_vae(encoder_vae(inputs)[2])
vae = Model(inputs, outputs_vae)
```

#### 3. Denoising Autoencoder

**Objectif** : Apprendre à **débruiter** les données.

```python
# Ajouter du bruit
noise_factor = 0.5
X_noisy = X + noise_factor * np.random.normal(size=X.shape)
X_noisy = np.clip(X_noisy, 0., 1.)

# Entraîner à reconstruire version propre
autoencoder.fit(X_noisy, X, epochs=10, batch_size=256)
```

### Application : Compression MNIST

```python
from tensorflow.keras.datasets import mnist

# Charger MNIST
(X_train, _), (X_test, _) = mnist.load_data()

# Prétraitement
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Entraîner autoencodeur
autoencoder.fit(X_train, X_train,
                epochs=20,
                batch_size=256,
                validation_data=(X_test, X_test))

# Encoder/décoder
encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)

print(f"Original: {X_test.shape}")  # (10000, 784)
print(f"Encodé: {encoded_imgs.shape}")  # (10000, 32)
print(f"Taux compression: {784/32:.1f}x")

# Visualisation
n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # Reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title('Reconstruit')
    plt.axis('off')

plt.tight_layout()
plt.show()
```

---

## Détection d'Anomalies

**Objectif** : Identifier les points **inhabituels** ou **aberrants**.

### Applications

- **Fraude** : Transactions bancaires
- **Maintenance** : Défauts machines
- **Cybersécurité** : Intrusions réseau
- **Santé** : Maladies rares

### Méthodes

#### 1. Isolation Forest

**Principe** : Points anormaux sont **plus faciles à isoler**.

```python
from sklearn.ensemble import IsolationForest

# Générer données avec anomalies
np.random.seed(42)
X_normal = np.random.randn(300, 2)
X_anomalies = np.random.uniform(-4, 4, (20, 2))
X_combined = np.vstack([X_normal, X_anomalies])

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
y_pred = iso_forest.fit_predict(X_combined)

# Visualisation
plt.figure(figsize=(10, 6))
plt.scatter(X_combined[y_pred == 1, 0], X_combined[y_pred == 1, 1],
            c='blue', label='Normal', s=50, alpha=0.6, edgecolors='k')
plt.scatter(X_combined[y_pred == -1, 0], X_combined[y_pred == -1, 1],
            c='red', label='Anomalie', s=100, marker='X', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Isolation Forest - Détection d\'Anomalies')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Anomalies détectées: {(y_pred == -1).sum()}")
```

#### 2. One-Class SVM

**Principe** : Apprendre la frontière de la **distribution normale**.

```python
from sklearn.svm import OneClassSVM

# One-Class SVM
oc_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
y_pred_svm = oc_svm.fit_predict(X_combined)

# Visualisation
plt.scatter(X_combined[y_pred_svm == 1, 0], X_combined[y_pred_svm == 1, 1],
            c='blue', label='Normal', s=50, alpha=0.6, edgecolors='k')
plt.scatter(X_combined[y_pred_svm == -1, 0], X_combined[y_pred_svm == -1, 1],
            c='red', label='Anomalie', s=100, marker='X', edgecolors='k')
plt.title('One-Class SVM')
plt.legend()
plt.show()
```

#### 3. Autoencodeur pour Anomalies

**Principe** : Anomalies ont **reconstruction error élevée**.

```python
# Entraîner autoencodeur sur données normales uniquement
autoencoder.fit(X_normal, X_normal, epochs=50, batch_size=32, verbose=0)

# Calculer reconstruction error
X_reconstructed = autoencoder.predict(X_combined)
reconstruction_error = np.mean(np.square(X_combined - X_reconstructed), axis=1)

# Seuil pour anomalies
threshold = np.percentile(reconstruction_error, 90)

# Prédictions
y_pred_ae = (reconstruction_error > threshold).astype(int)
y_pred_ae = np.where(y_pred_ae == 1, -1, 1)  # -1 pour anomalie

print(f"Anomalies (Autoencodeur): {(y_pred_ae == -1).sum()}")
```

---

## Systèmes de Recommandation

**Objectif** : Recommander items (films, produits) aux utilisateurs.

### Types

#### 1. Collaborative Filtering

**Principe** : Utiliser les **préférences d'utilisateurs similaires**.

**Approches** :
- **User-based** : Utilisateurs similaires aiment items similaires
- **Item-based** : Items similaires sont aimés par utilisateurs similaires

#### 2. Content-Based Filtering

**Principe** : Recommander items **similaires** à ceux déjà aimés.

#### 3. Matrix Factorization

Décomposer matrice utilisateur×item :

$$
\mathbf{R} \approx \mathbf{U} \mathbf{V}^T
$$

```python
from sklearn.decomposition import NMF

# Matrice utilisateur×item (ratings)
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

# NMF (Non-negative Matrix Factorization)
n_components = 2
nmf = NMF(n_components=n_components, random_state=42)
U = nmf.fit_transform(R)
V = nmf.components_

# Reconstruction
R_pred = U @ V

print("Matrice Originale:")
print(R)
print("\nMatrice Prédite:")
print(R_pred.astype(int))

# Recommandations
user_id = 0
unrated_items = np.where(R[user_id] == 0)[0]
predictions = R_pred[user_id, unrated_items]

print(f"\nRecommandations pour User {user_id}:")
for item, score in zip(unrated_items, predictions):
    print(f"  Item {item}: Score prédit = {score:.2f}")
```

---

## Introduction à l'Apprentissage par Renforcement

L'**apprentissage par renforcement** (Reinforcement Learning, RL) consiste à apprendre par **interaction** avec un environnement.

### Concepts Clés

- **Agent** : Apprenant/décideur
- **Environnement** : Monde avec lequel l'agent interagit
- **État** $s$ : Situation actuelle
- **Action** $a$ : Décision de l'agent
- **Récompense** $r$ : Feedback de l'environnement
- **Politique** $\pi(a|s)$ : Stratégie de l'agent

### Cycle RL

```
Agent
  ↓ Action (a_t)
Environnement
  ↓ État (s_{t+1}), Récompense (r_t)
Agent
  ...
```

### Q-Learning

**Objectif** : Apprendre la fonction Q (valeur action-état)

$$
Q(s, a) = r + \gamma \max_{a'} Q(s', a')
$$

```python
import gym
import numpy as np

# Environnement simple (FrozenLake)
env = gym.make('FrozenLake-v1', is_slippery=False)

# Initialiser Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparamètres
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 1000

# Q-Learning
for episode in range(episodes):
    state = env.reset()[0]
    done = False

    while not done:
        # Epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state])  # Exploit

        # Step
        next_state, reward, done, _, _ = env.step(action)

        # Q-update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

print("Q-table apprise:")
print(Q)

# Tester politique
state = env.reset()[0]
done = False
total_reward = 0

while not done:
    action = np.argmax(Q[state])
    state, reward, done, _, _ = env.step(action)
    total_reward += reward

print(f"Récompense totale: {total_reward}")
```

---

## Projets Pratiques

### Projet 1 : Segmentation Client

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Générer données clients synthétiques
np.random.seed(42)
n_customers = 500

customers = pd.DataFrame({
    'Age': np.random.randint(18, 70, n_customers),
    'Income': np.random.normal(50000, 20000, n_customers),
    'Spending': np.random.normal(30000, 15000, n_customers)
})

# Normaliser
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customers)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
customers['Segment'] = kmeans.fit_predict(X_scaled)

# Analyse des segments
print(customers.groupby('Segment').mean())

# Visualisation
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(customers['Age'], customers['Income'], customers['Spending'],
                     c=customers['Segment'], cmap='viridis', s=50)

ax.set_xlabel('Age')
ax.set_ylabel('Revenu')
ax.set_zlabel('Dépenses')
ax.set_title('Segmentation Client (K-Means)')
plt.colorbar(scatter, label='Segment')
plt.show()
```

### Projet 2 : Compression d'Images avec PCA

```python
from sklearn.datasets import fetch_olivetti_faces

# Charger visages
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X_faces = faces.data  # (400, 4096) - Images 64×64

# PCA avec différents nombres de composantes
n_components_list = [50, 100, 200, 400]

fig, axes = plt.subplots(2, len(n_components_list) + 1, figsize=(15, 6))

# Original
axes[0, 0].imshow(X_faces[0].reshape(64, 64), cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

axes[1, 0].imshow(X_faces[1].reshape(64, 64), cmap='gray')
axes[1, 0].axis('off')

for idx, n_comp in enumerate(n_components_list, 1):
    # PCA
    pca = PCA(n_components=n_comp)
    X_reduced = pca.fit_transform(X_faces)
    X_reconstructed = pca.inverse_transform(X_reduced)

    # Afficher
    axes[0, idx].imshow(X_reconstructed[0].reshape(64, 64), cmap='gray')
    axes[0, idx].set_title(f'{n_comp} comp\n({pca.explained_variance_ratio_.sum():.1%})')
    axes[0, idx].axis('off')

    axes[1, idx].imshow(X_reconstructed[1].reshape(64, 64), cmap='gray')
    axes[1, idx].axis('off')

plt.tight_layout()
plt.show()
```

---

## Résumé

### Points Clés

#### Clustering

| Méthode | Type | Avantages | Limites |
|---------|------|-----------|---------|
| **K-Means** | Partitionnement | Rapide, scalable | Clusters sphériques, $K$ fixe |
| **DBSCAN** | Densité | Formes arbitraires, outliers | Sensible aux paramètres |
| **Hiérarchique** | Hiérarchie | Dendrogramme, pas de $K$ | Lent |

#### Réduction Dimensionnalité

| Méthode | Type | Usage | Préserve |
|---------|------|-------|----------|
| **PCA** | Linéaire | Compression, preprocessing | Variance globale |
| **t-SNE** | Non-linéaire | Visualisation | Structure locale |
| **UMAP** | Non-linéaire | Visualisation, général | Structure locale+globale |
| **Autoencodeur** | Neural | Compression, génération | Features apprises |

#### Détection Anomalies

| Méthode | Principe |
|---------|----------|
| **Isolation Forest** | Isolation plus facile |
| **One-Class SVM** | Frontière distribution |
| **Autoencodeur** | Reconstruction error |

### Workflow Type

```python
# 1. Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=K)
labels = kmeans.fit_predict(X)

# 2. Réduction dimensionnalité
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# 3. Détection anomalies
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.1)
anomalies = iso.fit_predict(X)
```

### Checklist

- [ ] Normaliser données avant clustering/PCA
- [ ] Utiliser Elbow Method pour choisir $K$
- [ ] Comparer plusieurs méthodes de clustering
- [ ] PCA pour compression, t-SNE/UMAP pour visualisation
- [ ] Autoencodeur pour données complexes (images)
- [ ] Valider détection anomalies avec domaine expert
- [ ] Combiner méthodes (ex: PCA puis K-Means)

### Applications Réelles

| Domaine | Technique | Exemple |
|---------|-----------|---------|
| **Marketing** | Clustering | Segmentation client |
| **Finance** | Détection anomalies | Fraude bancaire |
| **Vision** | Autoencodeur | Compression, débruitage |
| **Biologie** | PCA, clustering | Analyse génomique |
| **Recommandation** | Matrix factorization | Netflix, Spotify |

---

**🎉 Félicitations ! Vous avez terminé la formation complète en Machine Learning ! 🎉**

---

**Navigation :**
- [⬅️ Module 8 : CNN](08_CNN.md)
- [🏠 Retour au Sommaire](README.md)
