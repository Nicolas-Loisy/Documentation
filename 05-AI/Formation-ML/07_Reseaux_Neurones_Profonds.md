# Module 7 : Réseaux de Neurones Profonds

## 📋 Table des Matières

1. [Introduction](#introduction)
2. [Le Perceptron](#le-perceptron)
3. [Perceptron Multicouche (MLP)](#perceptron-multicouche-mlp)
4. [Fonctions d'Activation](#fonctions-dactivation)
5. [Backpropagation](#backpropagation)
6. [Optimiseurs](#optimiseurs)
7. [Régularisation](#régularisation)
8. [Techniques Avancées](#techniques-avancées)
9. [Implémentation avec TensorFlow/Keras](#implémentation-avec-tensorflowkeras)
10. [Projet Pratique](#projet-pratique)
11. [Résumé](#résumé)

---

## Introduction

Les **réseaux de neurones artificiels** (Artificial Neural Networks, ANN) sont des modèles inspirés du cerveau humain, capables d'apprendre des représentations complexes.

### Pourquoi les Réseaux de Neurones ?

**Limitations des modèles classiques** :

- Nécessitent feature engineering manuel
- Difficulté avec données non-structurées (images, texte, audio)
- Relations non-linéaires complexes

**Avantages des réseaux de neurones** :

- **Apprentissage de représentations** : Extraction automatique de features
- **Universalité** : Peuvent approximer n'importe quelle fonction continue
- **Performance** : État de l'art sur vision, NLP, jeux, etc.

### Applications

| Domaine    | Tâche                     | Exemple                                    |
| ---------- | ------------------------- | ------------------------------------------ |
| **Vision** | Classification, détection | Reconnaissance faciale, voitures autonomes |
| **NLP**    | Traduction, génération    | Google Translate, ChatGPT                  |
| **Audio**  | Reconnaissance vocale     | Siri, Alexa                                |
| **Jeux**   | IA                        | AlphaGo, OpenAI Five                       |
| **Santé**  | Diagnostic                | Détection de cancer                        |

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow version: {tf.__version__}")
```

---

## Le Perceptron

Le **perceptron** est le neurone artificiel le plus simple (1957, Frank Rosenblatt).

### Modèle Mathématique

$$
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(\mathbf{w}^T \mathbf{x} + b)
$$

où :

- $\mathbf{x} = (x_1, \ldots, x_n)$ : entrées
- $\mathbf{w} = (w_1, \ldots, w_n)$ : poids
- $b$ : biais
- $f$ : fonction d'activation (ex: signe, sigmoïde)

### Fonction d'Activation

**Perceptron classique** : fonction signe

$$
f(z) = \begin{cases}
1 & \text{si } z \geq 0 \\
0 & \text{si } z < 0
\end{cases}
$$

### Algorithme d'Apprentissage

Pour classification binaire :

```
Pour chaque échantillon (x, y):
    1. Calculer prédiction: ŷ = f(w^T x + b)
    2. Calculer erreur: e = y - ŷ
    3. Mise à jour: w ← w + α·e·x
                    b ← b + α·e
```

### Implémentation

```python
class Perceptron:
    def __init__(self, n_features, learning_rate=0.01, n_iter=100):
        self.w = np.zeros(n_features)
        self.b = 0
        self.lr = learning_rate
        self.n_iter = n_iter

    def activation(self, z):
        """Fonction signe"""
        return np.where(z >= 0, 1, 0)

    def predict(self, X):
        """Prédictions"""
        z = np.dot(X, self.w) + self.b
        return self.activation(z)

    def fit(self, X, y):
        """Entraînement"""
        for _ in range(self.n_iter):
            for xi, yi in zip(X, y):
                # Prédiction
                y_pred = self.predict(xi.reshape(1, -1))[0]
                # Mise à jour
                error = yi - y_pred
                self.w += self.lr * error * xi
                self.b += self.lr * error

# Exemple : Porte logique AND
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

perceptron = Perceptron(n_features=2, learning_rate=0.1, n_iter=10)
perceptron.fit(X, y)

print("Poids:", perceptron.w)
print("Biais:", perceptron.b)
print("Prédictions:", perceptron.predict(X))
print("Vérité terrain:", y)
```

### Limitation du Perceptron

**Problème XOR** : Le perceptron ne peut résoudre que des problèmes **linéairement séparables**.

```python
# XOR : non linéairement séparable
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])  # XOR

# Le perceptron échouera
perceptron_xor = Perceptron(n_features=2, learning_rate=0.1, n_iter=100)
perceptron_xor.fit(X_xor, y_xor)

print("\nXOR - Prédictions:", perceptron_xor.predict(X_xor))
print("XOR - Vérité terrain:", y_xor)
print("→ Le perceptron ne peut résoudre XOR!")
```

**Solution** : Utiliser des réseaux **multicouches** (MLP).

---

## Perceptron Multicouche (MLP)

Le **MLP** (Multi-Layer Perceptron) empile plusieurs couches de neurones.

### Architecture

```
Entrée (X)  →  Couche Cachée 1  →  Couche Cachée 2  →  ...  →  Sortie (ŷ)
    (d)           (h₁ neurones)      (h₂ neurones)           (K classes)
```

**Composantes** :

- **Couche d'entrée** : Features brutes
- **Couches cachées** : Extraction de représentations
- **Couche de sortie** : Prédictions

### Forward Propagation

Pour une couche $l$ :

$$
\begin{align}
\mathbf{z}^{[l]} &= \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]} \\
\mathbf{a}^{[l]} &= g^{[l]}(\mathbf{z}^{[l]})
\end{align}
$$

où :

- $\mathbf{W}^{[l]}$ : matrice de poids
- $\mathbf{b}^{[l]}$ : vecteur de biais
- $g^{[l]}$ : fonction d'activation
- $\mathbf{a}^{[l]}$ : activations (sorties)

### Exemple : MLP pour XOR

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Données XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_xor = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Créer MLP
model = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),  # Couche cachée
    Dense(1, activation='sigmoid')                   # Sortie
])

# Compiler
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Entraîner
history = model.fit(X_xor, y_xor, epochs=1000, verbose=0)

# Prédictions
predictions = model.predict(X_xor)
print("\nXOR avec MLP:")
print("Prédictions:", (predictions > 0.5).astype(int).flatten())
print("Vérité terrain:", y_xor.flatten().astype(int))

# Visualisation de l'apprentissage
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Évolution de la Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Évolution de l\'Accuracy')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Fonctions d'Activation

Les fonctions d'activation introduisent la **non-linéarité**, essentielle pour apprendre des fonctions complexes.

### 1. Sigmoïde

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Propriétés** :

- Output : $(0, 1)$
- Dérivée : $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- **Usage** : Couche de sortie (classification binaire)
- **Problème** : Saturation (gradients vanishing)

### 2. Tanh (Tangente Hyperbolique)

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

**Propriétés** :

- Output : $(-1, 1)$
- Dérivée : $\tanh'(z) = 1 - \tanh^2(z)$
- **Avantage** : Centré sur 0
- **Problème** : Saturation

### 3. ReLU (Rectified Linear Unit)

$$
\text{ReLU}(z) = \max(0, z)
$$

**Propriétés** :

- Output : $[0, +\infty)$
- Dérivée :
  $$
  \text{ReLU}'(z) = \begin{cases}
  1 & \text{si } z > 0 \\
  0 & \text{si } z \leq 0
  \end{cases}
  $$
- **Avantages** :
  - Pas de saturation pour $z > 0$
  - Calcul rapide
  - Convergence plus rapide
- **Problème** : Dying ReLU (neurones "morts" si $z < 0$)

### 4. Leaky ReLU

$$
\text{LeakyReLU}(z) = \max(\alpha z, z)
$$

où $\alpha$ est petit (ex: 0.01)

**Avantage** : Résout le problème de dying ReLU

### 5. Softmax (Sortie Multi-classe)

$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Propriétés** :

- Output : Probabilités $\in [0, 1]$, $\sum_i = 1$
- **Usage** : Classification multiclasse

### Visualisation

```python
# Définir les fonctions
x = np.linspace(-5, 5, 1000)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Sigmoïde
axes[0, 0].plot(x, sigmoid(x), 'b-', linewidth=2)
axes[0, 0].set_title('Sigmoïde: σ(z) = 1/(1+e⁻ᶻ)')
axes[0, 0].set_xlabel('z')
axes[0, 0].set_ylabel('σ(z)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(0, color='k', linewidth=0.5)
axes[0, 0].axvline(0, color='k', linewidth=0.5)

# Tanh
axes[0, 1].plot(x, tanh(x), 'g-', linewidth=2)
axes[0, 1].set_title('Tanh: tanh(z)')
axes[0, 1].set_xlabel('z')
axes[0, 1].set_ylabel('tanh(z)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(0, color='k', linewidth=0.5)
axes[0, 1].axvline(0, color='k', linewidth=0.5)

# ReLU
axes[1, 0].plot(x, relu(x), 'r-', linewidth=2)
axes[1, 0].set_title('ReLU: max(0, z)')
axes[1, 0].set_xlabel('z')
axes[1, 0].set_ylabel('ReLU(z)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(0, color='k', linewidth=0.5)
axes[1, 0].axvline(0, color='k', linewidth=0.5)

# Leaky ReLU
axes[1, 1].plot(x, leaky_relu(x), 'm-', linewidth=2)
axes[1, 1].set_title('Leaky ReLU: max(0.01z, z)')
axes[1, 1].set_xlabel('z')
axes[1, 1].set_ylabel('Leaky ReLU(z)')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(0, color='k', linewidth=0.5)
axes[1, 1].axvline(0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()
```

**Choix de l'activation** :

| Usage              | Activation Recommandée      |
| ------------------ | --------------------------- |
| Couches cachées    | ReLU ou Leaky ReLU          |
| Sortie binaire     | Sigmoïde                    |
| Sortie multiclasse | Softmax                     |
| Régression         | Linéaire (pas d'activation) |

---

## Backpropagation

La **backpropagation** (rétropropagation) est l'algorithme pour calculer efficacement les gradients.

### Principe

1. **Forward pass** : Calculer les activations et la loss
2. **Backward pass** : Calculer les gradients en remontant
3. **Mise à jour** : Ajuster les poids avec descente de gradient

### Formules (Gradient de la Loss)

**Sortie** :

$$
\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}}
$$

**Couche $l$** :

$$
\delta^{[l]} = (\mathbf{W}^{[l+1]})^T \delta^{[l+1]} \odot g'^{[l]}(\mathbf{z}^{[l]})
$$

**Gradients des paramètres** :

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} &= \delta^{[l]} (\mathbf{a}^{[l-1]})^T \\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} &= \delta^{[l]}
\end{align}
$$

### Exemple Visuel

```
Forward (Calcul):
Input → [Layer 1] → [Layer 2] → Output → Loss

Backward (Gradients):
∂Loss/∂W₁ ← [Layer 1] ← [Layer 2] ← ∂Loss/∂Output
```

### Implémentation (TensorFlow gère automatiquement)

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Modèle
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compiler (backprop automatique)
model.compile(optimizer='adam', loss='binary_crossentropy')

# TensorFlow/Keras gère la backpropagation automatiquement lors de fit()
```

---

## Optimiseurs

Les **optimiseurs** contrôlent comment les poids sont mis à jour.

### 1. SGD (Stochastic Gradient Descent)

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla \mathcal{L}(\mathbf{w}_t)
$$

**Simple mais lent.**

### 2. SGD avec Momentum

$$
\begin{align}
\mathbf{v}_{t+1} &= \beta \mathbf{v}_t + (1-\beta) \nabla \mathcal{L}(\mathbf{w}_t) \\
\mathbf{w}_{t+1} &= \mathbf{w}_t - \alpha \mathbf{v}_{t+1}
\end{align}
$$

**Accélère la convergence.**

### 3. RMSprop

$$
\begin{align}
\mathbf{s}_{t+1} &= \beta \mathbf{s}_t + (1-\beta) (\nabla \mathcal{L}(\mathbf{w}_t))^2 \\
\mathbf{w}_{t+1} &= \mathbf{w}_t - \frac{\alpha}{\sqrt{\mathbf{s}_{t+1} + \epsilon}} \nabla \mathcal{L}(\mathbf{w}_t)
\end{align}
$$

**Learning rate adaptatif.**

### 4. Adam (Adaptive Moment Estimation)

Combine **Momentum** + **RMSprop** :

$$
\begin{align}
\mathbf{m}_{t+1} &= \beta_1 \mathbf{m}_t + (1-\beta_1) \nabla \mathcal{L}(\mathbf{w}_t) \\
\mathbf{v}_{t+1} &= \beta_2 \mathbf{v}_t + (1-\beta_2) (\nabla \mathcal{L}(\mathbf{w}_t))^2 \\
\hat{\mathbf{m}}_{t+1} &= \frac{\mathbf{m}_{t+1}}{1 - \beta_1^{t+1}} \\
\hat{\mathbf{v}}_{t+1} &= \frac{\mathbf{v}_{t+1}}{1 - \beta_2^{t+1}} \\
\mathbf{w}_{t+1} &= \mathbf{w}_t - \frac{\alpha}{\sqrt{\hat{\mathbf{v}}_{t+1}} + \epsilon} \hat{\mathbf{m}}_{t+1}
\end{align}
$$

**Optimiseur par défaut, très performant.**

### Comparaison

```python
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

optimizers = {
    'SGD': SGD(learning_rate=0.01),
    'SGD + Momentum': SGD(learning_rate=0.01, momentum=0.9),
    'RMSprop': RMSprop(learning_rate=0.001),
    'Adam': Adam(learning_rate=0.001)
}

# Entraîner avec différents optimiseurs
for name, opt in optimizers.items():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(10,)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    # ... fit() ...
```

**Recommandation** : Commencer avec **Adam**.

---

## Régularisation

La **régularisation** prévient l'overfitting.

### 1. L1/L2 Regularization

Ajouter pénalité sur les poids :

**L2 (Ridge)** :

$$
\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda \sum_{i} w_i^2
$$

**L1 (Lasso)** :

$$
\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda \sum_{i} |w_i|
$$

```python
from tensorflow.keras.regularizers import l1, l2, l1_l2

model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='sigmoid')
])
```

### 2. Dropout

**Principe** : Désactiver aléatoirement des neurones pendant l'entraînement.

$$
\text{Dropout rate } p : \text{ probabilité de désactivation}
$$

```python
from tensorflow.keras.layers import Dropout

model = Sequential([
    Dense(64, activation='relu'),
    Dropout(0.5),  # Désactive 50% des neurones
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

**Avantages** :

- Réduit l'overfitting
- Force le réseau à apprendre des représentations robustes

### 3. Early Stopping

Arrêter l'entraînement quand la performance sur validation cesse de s'améliorer.

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Attendre 10 epochs sans amélioration
    restore_best_weights=True
)

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=100,
          callbacks=[early_stop])
```

### 4. Data Augmentation

Augmenter artificiellement les données d'entraînement.

**Images** : Rotation, flip, zoom, crop

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
```

---

## Techniques Avancées

### 1. Batch Normalization

**Principe** : Normaliser les activations de chaque couche.

$$
\hat{z}^{[l]} = \frac{z^{[l]} - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

**Avantages** :

- Accélère l'entraînement
- Permet des learning rates plus élevés
- Réduit la sensibilité à l'initialisation

```python
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Dense(64),
    BatchNormalization(),
    Activation('relu'),
    Dense(32),
    BatchNormalization(),
    Activation('relu'),
    Dense(1, activation='sigmoid')
])
```

### 2. Initialisation des Poids

**Méthodes** :

- **Xavier/Glorot** : $W \sim \mathcal{N}(0, \frac{2}{n_{\text{in}} + n_{\text{out}}})$
- **He** : $W \sim \mathcal{N}(0, \frac{2}{n_{\text{in}}})$ (mieux pour ReLU)

```python
from tensorflow.keras.initializers import GlorotUniform, HeNormal

Dense(64, activation='relu', kernel_initializer=HeNormal())
```

### 3. Learning Rate Scheduling

Réduire le learning rate progressivement.

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Diviser par 2
    patience=5,
    min_lr=1e-7
)

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=100,
          callbacks=[lr_scheduler])
```

---

## Implémentation avec TensorFlow/Keras

### Workflow Complet

```python
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Données
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 2. Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 3. Créer le modèle
model = Sequential([
    Dense(128, activation='relu', input_shape=(20,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

# 4. Compiler
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
]

# 6. Entraîner
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 7. Évaluer
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# 8. Visualisation
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Évolution de la Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Évolution de l\'Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. Sauvegarder le modèle
model.save('mon_modele.h5')

# 10. Charger le modèle
# from tensorflow.keras.models import load_model
# model_loaded = load_model('mon_modele.h5')
```

---

## Projet Pratique

### Classification MNIST (Chiffres Manuscrits)

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Charger MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"X_train shape: {X_train.shape}")  # (60000, 28, 28)
print(f"y_train shape: {y_train.shape}")  # (60000,)

# Visualiser quelques exemples
plt.figure(figsize=(12, 3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f'{y_train[i]}')
    plt.axis('off')
plt.show()

# Prétraitement
X_train = X_train.reshape(-1, 28*28) / 255.0  # Flatten + normaliser
X_test = X_test.reshape(-1, 28*28) / 255.0

y_train = to_categorical(y_train, 10)  # One-hot encoding
y_test = to_categorical(y_test, 10)

# Créer modèle
model = Sequential([
    Dense(512, activation='relu', input_shape=(784,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.2),

    Dense(10, activation='softmax')  # 10 classes
])

# Compiler
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Summary
print(model.summary())

# Entraîner
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=128,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    ]
)

# Évaluer
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Prédictions
predictions = model.predict(X_test[:10])
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test[:10], axis=1)

# Afficher
plt.figure(figsize=(15, 3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    color = 'green' if predicted_classes[i] == true_classes[i] else 'red'
    plt.title(f'P:{predicted_classes[i]}\nT:{true_classes[i]}', color=color)
    plt.axis('off')
plt.show()
```

---

## Résumé

### Points Clés

#### Architecture

| Élément      | Description                    |
| ------------ | ------------------------------ |
| **Couches**  | Input → Hidden layers → Output |
| **Neurone**  | $a = g(w^T x + b)$             |
| **Forward**  | Calcul des activations         |
| **Backward** | Backpropagation des gradients  |

#### Fonctions d'Activation

| Fonction    | Usage              | Formule                |
| ----------- | ------------------ | ---------------------- |
| **ReLU**    | Couches cachées    | $\max(0, z)$           |
| **Sigmoid** | Sortie binaire     | $1/(1+e^{-z})$         |
| **Softmax** | Sortie multiclasse | $e^{z_i}/\sum e^{z_j}$ |

#### Régularisation

| Technique          | Principe                          |
| ------------------ | --------------------------------- |
| **Dropout**        | Désactiver neurones aléatoirement |
| **L2**             | Pénaliser gros poids              |
| **Early Stopping** | Arrêter si val ne s'améliore plus |
| **Batch Norm**     | Normaliser activations            |

#### Optimiseurs

- **SGD** : Simple mais lent
- **Momentum** : Accélère
- **RMSprop** : Learning rate adaptatif
- **Adam** : Meilleur choix par défaut

### Checklist Entraînement

- [ ] Normaliser les données (StandardScaler)
- [ ] Split train/val/test
- [ ] Choisir architecture (nombre de couches, neurones)
- [ ] Choisir activation (ReLU, sigmoid, softmax)
- [ ] Choisir optimiseur (Adam recommandé)
- [ ] Ajouter régularisation (Dropout, L2)
- [ ] Définir callbacks (EarlyStopping, LR scheduler)
- [ ] Entraîner avec validation
- [ ] Évaluer sur test
- [ ] Visualiser loss et accuracy
- [ ] Analyser prédictions

### Code Type

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=100, callbacks=[EarlyStopping(patience=10)])
```

### Prochaine Étape

**Module 8 : Réseaux de Neurones Convolutifs (CNN)** - Architecture pour images

---

**Navigation :**

- [⬅️ Module 6 : Apprentissage Supervisé](06_Apprentissage_Supervise.md)
- [🏠 Retour au Sommaire](README_ML.md)
- [➡️ Module 8 : Réseaux de Neurones Convolutifs (CNN)](08_CNN.md)
