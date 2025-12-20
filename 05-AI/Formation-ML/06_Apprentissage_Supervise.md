# Module 6 : Apprentissage Supervisé

## 📋 Table des Matières
1. [Introduction](#introduction)
2. [Régression Linéaire](#régression-linéaire)
3. [Régression Logistique](#régression-logistique)
4. [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
5. [Arbres de Décision](#arbres-de-décision)
6. [Support Vector Machines (SVM)](#support-vector-machines-svm)
7. [Méthodes d'Ensemble](#méthodes-densemble)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Évaluation des Modèles](#évaluation-des-modèles)
10. [Projet Complet : Prédiction de Prix de Maisons](#projet-complet--prédiction-de-prix-de-maisons)
11. [Résumé](#résumé)

---

## Introduction

L'**apprentissage supervisé** consiste à entraîner un modèle sur des données labelisées pour prédire une valeur cible.

### Définition

Étant donné un dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ où :
- $\mathbf{x}_i \in \mathbb{R}^d$ : vecteur de features (caractéristiques)
- $y_i$ : label/cible

**Objectif** : Apprendre une fonction $f : \mathbb{R}^d \to \mathcal{Y}$ telle que :
$$
f(\mathbf{x}) \approx y
$$

### Types de Problèmes

| Type | Cible | Exemples |
|------|-------|----------|
| **Régression** | Continue ($\mathcal{Y} = \mathbb{R}$) | Prix immobilier, température |
| **Classification** | Discrète ($\mathcal{Y} = \{c_1, \ldots, c_K\}$) | Spam/non-spam, reconnaissance d'images |

### Workflow ML Supervisé

```
1. Collecte des données
   ↓
2. Exploration et prétraitement (EDA)
   ↓
3. Feature engineering
   ↓
4. Split train/validation/test
   ↓
5. Entraînement du modèle
   ↓
6. Évaluation et validation
   ↓
7. Hyperparameter tuning
   ↓
8. Prédiction sur nouvelles données
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
```

---

## Régression Linéaire

La **régression linéaire** modélise la relation entre features et cible par une combinaison linéaire.

### Modèle

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \cdots + w_d x_d + b
$$

où :
- $\mathbf{w} \in \mathbb{R}^d$ : poids (coefficients)
- $b \in \mathbb{R}$ : biais (intercept)

**Forme vectorielle** :
$$
\hat{y} = \mathbf{w}^T \mathbf{x}
$$
(en incluant $b$ dans $\mathbf{w}$ et ajoutant une feature constante = 1)

### Fonction de Coût (MSE)

**Mean Squared Error** :
$$
\mathcal{L}(\mathbf{w}) = \frac{1}{2N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 = \frac{1}{2N} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2
$$

**Objectif** : $\mathbf{w}^* = \arg\min_{\mathbf{w}} \mathcal{L}(\mathbf{w})$

### Solution Analytique (Normal Equation)

$$
\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

**Avantages** :
- Solution exacte en une étape
- Pas de learning rate

**Inconvénients** :
- Coûteux si $d$ grand (inversion de matrice $O(d^3)$)
- Nécessite $\mathbf{X}^T \mathbf{X}$ inversible

### Solution Itérative (Gradient Descent)

$$
\mathbf{w}_{k+1} = \mathbf{w}_k - \alpha \nabla \mathcal{L}(\mathbf{w}_k)
$$

**Gradient** :
$$
\nabla \mathcal{L}(\mathbf{w}) = \frac{1}{N} \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{y})
$$

### Implémentation avec Scikit-Learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Générer des données synthétiques
X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Visualisation
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.6, label='Train')
plt.scatter(X_test, y_test, alpha=0.6, label='Test')
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, 'r-', linewidth=2, label='Régression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Régression Linéaire')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Parfait')
plt.xlabel('Vraie valeur')
plt.ylabel('Prédiction')
plt.title(f'Prédictions vs Valeurs Réelles (R²={r2:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Régression Polynomiale

Extension pour capturer des relations non-linéaires :

$$
\hat{y} = w_0 + w_1 x + w_2 x^2 + \cdots + w_p x^p
$$

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Créer pipeline
degree = 3
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=degree)),
    ('linear', LinearRegression())
])

# Entraîner
poly_model.fit(X_train, y_train)

# Prédire
y_pred_poly = poly_model.predict(X_test)

# Évaluation
r2_poly = r2_score(y_test, y_pred_poly)
print(f"R² (polynôme degré {degree}): {r2_poly:.4f}")

# Visualisation
plt.scatter(X_train, y_train, alpha=0.6, label='Train')
plt.scatter(X_test, y_test, alpha=0.6, label='Test')
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line_poly = poly_model.predict(X_line)
plt.plot(X_line, y_line_poly, 'g-', linewidth=2, label=f'Polynôme (degré {degree})')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Régression Polynomiale')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Régularisation

Pour éviter l'**overfitting**, on ajoute un terme de pénalité.

#### Ridge Regression (L2)

$$
\mathcal{L}(\mathbf{w}) = \frac{1}{2N} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|^2
$$

- Pénalise les gros poids
- Solution analytique : $\mathbf{w}^* = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)  # alpha = λ
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

print(f"R² (Ridge): {r2_score(y_test, y_pred_ridge):.4f}")
```

#### Lasso Regression (L1)

$$
\mathcal{L}(\mathbf{w}) = \frac{1}{2N} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|_1
$$

- Favorise la **sparsité** (certains poids = 0)
- Utile pour la **sélection de features**

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

print(f"R² (Lasso): {r2_score(y_test, y_pred_lasso):.4f}")
print(f"Nombre de coefficients non-nuls: {np.sum(lasso.coef_ != 0)}")
```

---

## Régression Logistique

La **régression logistique** est utilisée pour la **classification binaire**.

### Modèle

$$
P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}
$$

où $\sigma(z)$ est la **fonction sigmoïde** :

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Propriétés** :
- $\sigma(z) \in (0, 1)$
- $\sigma(0) = 0.5$
- $\sigma(-z) = 1 - \sigma(z)$

### Fonction de Coût (Cross-Entropy)

**Binary Cross-Entropy** :

$$
\mathcal{L}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]
$$

où $\hat{y}_i = \sigma(\mathbf{w}^T \mathbf{x}_i)$

### Implémentation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Générer données
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Prédictions
y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)  # Probabilités

# Évaluation
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualisation
plt.figure(figsize=(12, 5))

# Frontière de décision
plt.subplot(1, 2, 1)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu',
            edgecolors='k', alpha=0.7, label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu',
            edgecolors='k', marker='^', s=100, alpha=0.7, label='Test')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Frontière de Décision')
plt.legend()

# Probabilités
plt.subplot(1, 2, 2)
plt.scatter(range(len(y_proba)), y_proba[:, 1], c=y_test,
            cmap='RdYlBu', edgecolors='k', alpha=0.7)
plt.axhline(0.5, color='k', linestyle='--', label='Seuil = 0.5')
plt.xlabel('Échantillon')
plt.ylabel('P(y=1)')
plt.title('Probabilités de Prédiction')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Classification Multiclasse

Pour $K > 2$ classes : **Softmax Regression**

$$
P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x}}}
$$

```python
# Générer données multiclasses
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                           n_informative=2, n_classes=3, n_clusters_per_class=1,
                           random_state=42)

# Entraînement
log_reg_multi = LogisticRegression(multi_class='multinomial', max_iter=200)
log_reg_multi.fit(X_train, y_train)

# Évaluation
y_pred_multi = log_reg_multi.predict(X_test)
print(f"Accuracy (multiclasse): {accuracy_score(y_test, y_pred_multi):.4f}")
```

---

## K-Nearest Neighbors (KNN)

**K-Nearest Neighbors** est un algorithme non-paramétrique basé sur la proximité.

### Principe

Pour prédire $y$ pour un nouveau point $\mathbf{x}$ :

1. Trouver les $k$ points les plus proches de $\mathbf{x}$ dans l'ensemble d'entraînement
2. **Classification** : Vote majoritaire parmi les $k$ voisins
3. **Régression** : Moyenne des $k$ voisins

**Distance** : Euclidienne (par défaut)
$$
d(\mathbf{x}, \mathbf{x}') = \|\mathbf{x} - \mathbf{x}'\| = \sqrt{\sum_{i=1}^{d} (x_i - x_i')^2}
$$

### Hyperparamètres

- **$k$** : Nombre de voisins
  - $k$ petit : Frontière complexe, risque d'overfitting
  - $k$ grand : Frontière lisse, risque d'underfitting
- **Distance** : Euclidienne, Manhattan, Minkowski, etc.
- **Poids** : Uniforme ou pondéré par la distance

### Implémentation

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Classification
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_informative=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)

y_pred = knn_clf.predict(X_test)
print(f"Accuracy (KNN): {accuracy_score(y_test, y_pred):.4f}")

# Effet de k
k_values = [1, 3, 5, 10, 20, 50]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    accuracies.append(acc)

# Visualisation
plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Nombre de voisins (k)')
plt.ylabel('Accuracy')
plt.title('Impact de k sur la Performance')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Meilleur k: {k_values[np.argmax(accuracies)]} (Accuracy: {max(accuracies):.4f})")
```

### Régression KNN

```python
# Régression
X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)

y_pred = knn_reg.predict(X_test)
print(f"R² (KNN Régression): {r2_score(y_test, y_pred):.4f}")

# Visualisation
plt.scatter(X_train, y_train, alpha=0.6, label='Train')
plt.scatter(X_test, y_test, alpha=0.6, label='Test')
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = knn_reg.predict(X_line)
plt.plot(X_line, y_line, 'r-', linewidth=2, label='Prédiction KNN')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Régression KNN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Avantages** :
- Simple, pas d'entraînement
- Non-paramétrique, flexible

**Inconvénients** :
- Lent en prédiction (O(N))
- Sensible à la dimensionnalité (curse of dimensionality)
- Nécessite normalisation des features

---

## Arbres de Décision

Les **arbres de décision** divisent récursivement l'espace des features.

### Structure

```
                [X₁ < 5?]
                /        \
             Oui          Non
             /              \
        [X₂ < 3?]      [X₁ < 8?]
         /    \          /    \
      Oui    Non      Oui    Non
      /        \      /        \
   Classe A  Classe B  Classe C  Classe D
```

### Algorithme de Construction

**CART (Classification And Regression Trees)** :

1. Choisir la meilleure séparation (feature + seuil)
2. Diviser les données
3. Répéter récursivement jusqu'à un critère d'arrêt

**Critère de séparation** :

- **Classification** : Gini Impurity ou Entropie
- **Régression** : MSE

**Gini Impurity** :
$$
\text{Gini}(S) = 1 - \sum_{k=1}^{K} p_k^2
$$

**Entropie** :
$$
H(S) = -\sum_{k=1}^{K} p_k \log_2(p_k)
$$

où $p_k$ est la proportion de la classe $k$ dans l'ensemble $S$.

### Implémentation

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

# Classification
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_informative=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)
print(f"Accuracy (Arbre): {accuracy_score(y_test, y_pred):.4f}")

# Visualisation de l'arbre
plt.figure(figsize=(15, 10))
plot_tree(tree_clf, filled=True, feature_names=['X1', 'X2'],
          class_names=['Classe 0', 'Classe 1'], rounded=True)
plt.title('Arbre de Décision')
plt.show()

# Feature Importance
importances = tree_clf.feature_importances_
print(f"Feature Importances: {importances}")

# Visualisation de la frontière
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = tree_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu',
            edgecolors='k', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Frontière de Décision (Arbre)')
plt.show()
```

### Hyperparamètres Importants

| Paramètre | Description |
|-----------|-------------|
| `max_depth` | Profondeur maximale de l'arbre |
| `min_samples_split` | Nombre minimum d'échantillons pour diviser |
| `min_samples_leaf` | Nombre minimum d'échantillons dans une feuille |
| `max_features` | Nombre maximum de features à considérer |
| `criterion` | Critère de division (gini, entropy, mse) |

**Avantages** :
- Interprétable
- Pas besoin de normalisation
- Gère features catégorielles
- Capture relations non-linéaires

**Inconvénients** :
- Tendance à l'overfitting
- Instable (petite variation → arbre différent)
- Biais si classes déséquilibrées

---

## Support Vector Machines (SVM)

**SVM** trouve l'hyperplan qui maximise la marge entre les classes.

### Principe (Classification Binaire)

**Hyperplan séparateur** :
$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

**Marge** : Distance minimale entre l'hyperplan et les points les plus proches.

**Objectif** : Maximiser la marge
$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{sujet à} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1
$$

### Soft Margin (avec erreurs)

Pour données non-linéairement séparables :

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i
$$

où :
- $\xi_i$ : variables de relâchement (slack variables)
- $C$ : paramètre de régularisation (trade-off marge/erreurs)

### Kernel Trick

Pour problèmes non-linéaires, projeter dans un espace de dimension supérieure.

**Kernels courants** :

| Kernel | Formule |
|--------|---------|
| **Linéaire** | $K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$ |
| **Polynomial** | $K(\mathbf{x}, \mathbf{x}') = (\gamma \mathbf{x}^T \mathbf{x}' + r)^d$ |
| **RBF (Gaussien)** | $K(\mathbf{x}, \mathbf{x}') = e^{-\gamma \|\mathbf{x} - \mathbf{x}'\|^2}$ |

### Implémentation

```python
from sklearn.svm import SVC, SVR

# Classification
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM linéaire
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)
print(f"Accuracy (SVM linéaire): {svm_linear.score(X_test, y_test):.4f}")

# SVM RBF
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)
print(f"Accuracy (SVM RBF): {svm_rbf.score(X_test, y_test):.4f}")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, model, title in zip(axes, [svm_linear, svm_rbf], ['SVM Linéaire', 'SVM RBF']):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu',
               edgecolors='k', alpha=0.7)
    # Support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=200, linewidth=2, facecolors='none', edgecolors='k', label='Support Vectors')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()
```

**Avantages** :
- Performant en haute dimension
- Efficace en mémoire (seuls les support vectors)
- Kernels pour non-linéarité

**Inconvénients** :
- Lent sur gros datasets
- Sensible au choix du kernel et des hyperparamètres
- Difficile à interpréter

---

## Méthodes d'Ensemble

Les **méthodes d'ensemble** combinent plusieurs modèles pour améliorer les performances.

### 1. Bagging (Bootstrap Aggregating)

**Principe** :
1. Créer $B$ sous-ensembles par bootstrap (tirage avec remise)
2. Entraîner un modèle sur chaque sous-ensemble
3. Agréger les prédictions (vote ou moyenne)

**Objectif** : Réduire la variance

#### Random Forest

**Random Forest** = Bagging d'arbres de décision + randomisation des features

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)
print(f"Accuracy (Random Forest): {accuracy_score(y_test, y_pred):.4f}")

# Feature Importance
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(X.shape[1]), importances[indices])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances (Random Forest)')
plt.show()
```

### 2. Boosting

**Principe** : Entraîner séquentiellement des modèles faibles, chacun corrigeant les erreurs du précédent.

**Objectif** : Réduire le biais

#### Gradient Boosting

$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta h_m(\mathbf{x})
$$

où :
- $h_m$ : modèle faible (souvent un arbre peu profond)
- $\eta$ : learning rate

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Classification
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                     max_depth=3, random_state=42)
gb_clf.fit(X_train, y_train)

y_pred = gb_clf.predict(X_test)
print(f"Accuracy (Gradient Boosting): {accuracy_score(y_test, y_pred):.4f}")
```

#### XGBoost

**XGBoost** (Extreme Gradient Boosting) : Version optimisée et régularisée de Gradient Boosting.

```python
from xgboost import XGBClassifier, XGBRegressor

# Classification
xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                        random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_test)
print(f"Accuracy (XGBoost): {accuracy_score(y_test, y_pred):.4f}")
```

### 3. Stacking

**Principe** :
1. Diviser les données en K folds
2. Entraîner plusieurs modèles de base (niveau 1) avec validation croisée
3. Utiliser les prédictions comme features pour un méta-modèle (niveau 2)

```python
from sklearn.ensemble import StackingClassifier

# Modèles de base
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
]

# Méta-modèle
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

stacking_clf.fit(X_train, y_train)
y_pred = stacking_clf.predict(X_test)

print(f"Accuracy (Stacking): {accuracy_score(y_test, y_pred):.4f}")
```

### Comparaison des Méthodes

```python
from sklearn.model_selection import cross_val_score

models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"{name:20s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Visualisation
names = list(results.keys())
means = [results[name]['mean'] for name in names]
stds = [results[name]['std'] for name in names]

plt.figure(figsize=(12, 6))
plt.barh(names, means, xerr=stds, alpha=0.7, capsize=5)
plt.xlabel('Accuracy (CV)')
plt.title('Comparaison des Modèles')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Hyperparameter Tuning

L'optimisation des hyperparamètres est cruciale pour maximiser les performances.

### 1. Grid Search

**Principe** : Tester toutes les combinaisons d'hyperparamètres dans une grille.

```python
from sklearn.model_selection import GridSearchCV

# Définir la grille
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Grid Search
grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Meilleurs paramètres: {grid_search.best_params_}")
print(f"Meilleure accuracy (CV): {grid_search.best_score_:.4f}")

# Évaluation sur test
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Accuracy sur test: {accuracy_score(y_test, y_pred):.4f}")
```

**Avantages** :
- Exhaustif
- Simple

**Inconvénients** :
- Très coûteux si grille large
- Croissance exponentielle avec le nombre de paramètres

### 2. Random Search

**Principe** : Tirer aléatoirement des combinaisons d'hyperparamètres.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Définir distributions
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.2)
}

# Random Search
random_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions,
    n_iter=50,  # Nombre de combinaisons à tester
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)

print(f"Meilleurs paramètres: {random_search.best_params_}")
print(f"Meilleure accuracy (CV): {random_search.best_score_:.4f}")
```

**Avantages** :
- Plus rapide que Grid Search
- Explore mieux l'espace

### 3. Bayesian Optimization

**Principe** : Utiliser un modèle probabiliste (Processus Gaussien) pour guider la recherche.

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Définir l'espace de recherche
search_spaces = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(3, 15),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform')
}

# Bayesian Optimization
bayes_search = BayesSearchCV(
    GradientBoostingClassifier(random_state=42),
    search_spaces,
    n_iter=30,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

bayes_search.fit(X_train, y_train)

print(f"Meilleurs paramètres: {bayes_search.best_params_}")
print(f"Meilleure accuracy (CV): {bayes_search.best_score_:.4f}")
```

**Avantages** :
- Plus efficace que Random Search
- Explore intelligemment

**Inconvénients** :
- Plus complexe
- Nécessite une librairie externe (scikit-optimize)

---

## Évaluation des Modèles

### Métriques de Régression

| Métrique | Formule | Description |
|----------|---------|-------------|
| **MSE** | $\frac{1}{N}\sum(y_i - \hat{y}_i)^2$ | Erreur quadratique moyenne |
| **RMSE** | $\sqrt{MSE}$ | Même unité que $y$ |
| **MAE** | $\frac{1}{N}\sum|y_i - \hat{y}_i|$ | Erreur absolue moyenne |
| **MAPE** | $\frac{1}{N}\sum\frac{|y_i - \hat{y}_i|}{y_i}$ | Erreur en pourcentage |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Coefficient de détermination |

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Supposons des prédictions y_pred pour une régression
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"R²: {r2:.4f}")
```

### Métriques de Classification

#### Classification Binaire

**Confusion Matrix** :

```
                  Prédit Négatif    Prédit Positif
Vrai Négatif (TN)       TN                FP
Vrai Positif (TP)       FN                TP
```

**Métriques** :

| Métrique | Formule | Description |
|----------|---------|-------------|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Proportion correcte |
| **Precision** | $\frac{TP}{TP + FP}$ | Parmi prédits positifs, % vrais |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Parmi vrais positifs, % détectés |
| **F1-Score** | $\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$ | Moyenne harmonique |
| **Specificity** | $\frac{TN}{TN + FP}$ | Parmi vrais négatifs, % détectés |

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# Supposons des prédictions y_pred pour classification binaire
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# AUC-ROC
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilités classe positive
auc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC: {auc:.4f}")

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Hasard')
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Courbe ROC')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Validation Croisée

**K-Fold Cross-Validation** :

```python
from sklearn.model_selection import cross_val_score, KFold

# 5-Fold CV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"Scores CV: {scores}")
print(f"Moyenne: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

**Stratified K-Fold** : Préserve la proportion des classes dans chaque fold.

```python
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')
```

---

## Projet Complet : Prédiction de Prix de Maisons

### Dataset

Prédire le prix de vente de maisons à partir de diverses caractéristiques.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Générer un dataset synthétique (ou charger vos données)
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'LotArea': np.random.randint(5000, 20000, n),
    'OverallQual': np.random.randint(1, 11, n),
    'YearBuilt': np.random.randint(1950, 2023, n),
    'GrLivArea': np.random.randint(800, 4000, n),
    'FullBath': np.random.randint(1, 4, n),
    'BedroomAbvGr': np.random.randint(1, 6, n),
    'GarageCars': np.random.randint(0, 4, n)
})

# Prix calculé (avec bruit)
data['SalePrice'] = (
    data['GrLivArea'] * 100 +
    data['OverallQual'] * 10000 +
    data['GarageCars'] * 5000 +
    (2023 - data['YearBuilt']) * (-200) +
    np.random.normal(0, 20000, n)
).clip(lower=50000)

print(data.head())
print(f"\nShape: {data.shape}")
print(f"\nInfo:")
print(data.info())
```

### Étape 1 : EDA (Exploratory Data Analysis)

```python
# Statistiques descriptives
print("\n" + "="*60)
print("STATISTIQUES DESCRIPTIVES")
print("="*60)
print(data.describe())

# Distribution du prix
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(data['SalePrice'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Prix de vente ($)')
plt.ylabel('Fréquence')
plt.title('Distribution des prix')

plt.subplot(1, 2, 2)
sns.boxplot(data=data, y='SalePrice')
plt.ylabel('Prix de vente ($)')
plt.title('Boxplot des prix')

plt.tight_layout()
plt.show()

# Skewness et Kurtosis
print(f"\nSkewness (SalePrice): {data['SalePrice'].skew():.4f}")
print(f"Kurtosis (SalePrice): {data['SalePrice'].kurtosis():.4f}")

# Matrice de corrélation
corr_matrix = data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1)
plt.title('Matrice de Corrélation')
plt.tight_layout()
plt.show()

# Corrélations avec SalePrice
sale_price_corr = corr_matrix['SalePrice'].sort_values(ascending=False)
print("\nCorrélations avec SalePrice:")
print(sale_price_corr)

# Visualisation
plt.figure(figsize=(10, 6))
sns.barplot(x=sale_price_corr.values, y=sale_price_corr.index, palette='viridis')
plt.xlabel('Corrélation')
plt.title('Corrélations avec le Prix de Vente')
plt.tight_layout()
plt.show()
```

### Étape 2 : Prétraitement

```python
# Séparer features et target
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split train/validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test: {X_test.shape}")

# Vérifier les valeurs manquantes
print(f"\nValeurs manquantes (train): {X_train.isna().sum().sum()}")

# Normalisation (optionnel pour certains modèles)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

### Étape 3 : Entraînement de Modèles

```python
# Définir les modèles
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

# Entraîner et évaluer
results = {}

for name, model in models.items():
    # Entraîner
    model.fit(X_train, y_train)

    # Prédire sur validation
    y_pred_val = model.predict(X_val)

    # Métriques
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mape_val = mean_absolute_percentage_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)

    results[name] = {
        'model': model,
        'rmse_val': rmse_val,
        'mape_val': mape_val,
        'r2_val': r2_val
    }

    print(f"{name:20s} | RMSE: {rmse_val:10.2f} | MAPE: {mape_val:.4f} | R²: {r2_val:.4f}")
```

### Étape 4 : Hyperparameter Tuning

```python
# Tuning du meilleur modèle (ex: Gradient Boosting)
print("\n" + "="*60)
print("HYPERPARAMETER TUNING (Gradient Boosting)")
print("="*60)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nMeilleurs paramètres: {grid_search.best_params_}")
print(f"Meilleur score (CV): {-grid_search.best_score_:.2f}")

# Modèle optimisé
best_model = grid_search.best_estimator_
y_pred_val_tuned = best_model.predict(X_val)

rmse_tuned = np.sqrt(mean_squared_error(y_val, y_pred_val_tuned))
mape_tuned = mean_absolute_percentage_error(y_val, y_pred_val_tuned)
r2_tuned = r2_score(y_val, y_pred_val_tuned)

print(f"\nPerformances après tuning:")
print(f"RMSE: {rmse_tuned:.2f}")
print(f"MAPE: {mape_tuned:.4f}")
print(f"R²: {r2_tuned:.4f}")
```

### Étape 5 : Ensemble (Stacking)

```python
print("\n" + "="*60)
print("ENSEMBLE MODELING (STACKING)")
print("="*60)

# Niveau 1 : Entraîner plusieurs modèles
base_models = {
    'RF': RandomForestRegressor(n_estimators=200, random_state=42),
    'GB': GradientBoostingRegressor(n_estimators=200, random_state=42),
    'XGB': XGBRegressor(n_estimators=200, random_state=42),
    'Ridge': Ridge(alpha=1.0)
}

# Prédictions niveau 1
meta_features_train = pd.DataFrame()
meta_features_val = pd.DataFrame()
meta_features_test = pd.DataFrame()

for name, model in base_models.items():
    # Entraîner
    model.fit(X_train, y_train)

    # Prédictions
    meta_features_train[name] = model.predict(X_train)
    meta_features_val[name] = model.predict(X_val)
    meta_features_test[name] = model.predict(X_test)

# Niveau 2 : Méta-modèle
meta_model = LinearRegression()
meta_model.fit(meta_features_val, y_val)

# Prédictions finales
y_pred_test_stacking = meta_model.predict(meta_features_test)

# Évaluation
rmse_stacking = np.sqrt(mean_squared_error(y_test, y_pred_test_stacking))
mape_stacking = mean_absolute_percentage_error(y_test, y_pred_test_stacking)
r2_stacking = r2_score(y_test, y_pred_test_stacking)

print(f"Performances Stacking (test):")
print(f"RMSE: {rmse_stacking:.2f}")
print(f"MAPE: {mape_stacking:.4f}")
print(f"R²: {r2_stacking:.4f}")
```

### Étape 6 : Évaluation Finale sur Test

```python
print("\n" + "="*60)
print("ÉVALUATION FINALE SUR TEST SET")
print("="*60)

# Évaluer tous les modèles sur test
for name, info in results.items():
    model = info['model']
    y_pred_test = model.predict(X_test)

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"{name:20s} | RMSE: {rmse_test:10.2f} | MAPE: {mape_test:.4f} | R²: {r2_test:.4f}")

# Visualisation finale
best_model_name = min(results, key=lambda k: results[k]['rmse_val'])
best_model_obj = results[best_model_name]['model']
y_pred_final = best_model_obj.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_final, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Parfait')
plt.xlabel('Prix Réel ($)')
plt.ylabel('Prix Prédit ($)')
plt.title(f'Prédictions vs Réalité ({best_model_name})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Résumé

### Points Clés à Retenir

#### 1. Types de Problèmes

| Type | Objectif | Métriques |
|------|----------|-----------|
| **Régression** | Prédire valeur continue | MSE, RMSE, MAE, MAPE, R² |
| **Classification** | Prédire classe | Accuracy, Precision, Recall, F1, AUC |

#### 2. Modèles Linéaires

| Modèle | Usage | Régularisation |
|--------|-------|----------------|
| **Régression Linéaire** | Régression | - |
| **Ridge** | Régression | L2 ($\lambda \|\mathbf{w}\|^2$) |
| **Lasso** | Régression + sélection | L1 ($\lambda \|\mathbf{w}\|_1$) |
| **Régression Logistique** | Classification binaire | - |

#### 3. Modèles Non-Linéaires

| Modèle | Principe | Avantages | Inconvénients |
|--------|----------|-----------|---------------|
| **KNN** | Proximité | Simple, non-paramétrique | Lent, sensible à dim. |
| **Arbres de Décision** | Divisions récursives | Interprétable | Overfitting |
| **SVM** | Maximisation marge | Haute dim. | Lent, choix kernel |

#### 4. Méthodes d'Ensemble

| Méthode | Stratégie | Algorithmes |
|---------|-----------|-------------|
| **Bagging** | Réduire variance | Random Forest |
| **Boosting** | Réduire biais | Gradient Boosting, XGBoost |
| **Stacking** | Combiner modèles | Méta-modèle |

#### 5. Workflow ML

```
1. EDA (Analyse exploratoire)
2. Prétraitement (nettoyage, features)
3. Split train/val/test
4. Baseline model
5. Modèles avancés
6. Hyperparameter tuning
7. Ensemble methods
8. Évaluation finale
```

### Métriques Essentielles

**Régression** :
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
MAE = mean_absolute_error(y_true, y_pred)
MAPE = mean_absolute_percentage_error(y_true, y_pred)
R2 = r2_score(y_true, y_pred)
```

**Classification** :
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

Accuracy = accuracy_score(y_true, y_pred)
Precision = precision_score(y_true, y_pred)
Recall = recall_score(y_true, y_pred)
F1 = f1_score(y_true, y_pred)
AUC = roc_auc_score(y_true, y_proba)
```

### Bibliothèques Python

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import *
```

### Checklist Projet ML

- [ ] Charger et explorer les données (EDA)
- [ ] Traiter les valeurs manquantes
- [ ] Feature engineering et sélection
- [ ] Normaliser/standardiser si nécessaire
- [ ] Split train/validation/test
- [ ] Entraîner baseline model
- [ ] Comparer plusieurs modèles
- [ ] Validation croisée
- [ ] Hyperparameter tuning
- [ ] Essayer ensemble methods
- [ ] Évaluer sur test set
- [ ] Analyser les erreurs
- [ ] Interpréter le modèle (feature importance)

### Prochaine Étape

**Module 7 : Réseaux de Neurones Profonds** - Perceptron, MLP, backpropagation, régularisation

---

**Navigation :**
- [⬅️ Module 5 : Optimisation Numérique](05_Optimisation_Numerique.md)
- [🏠 Retour au Sommaire](README.md)
- [➡️ Module 7 : Réseaux de Neurones Profonds](07_Reseaux_Neurones_Profonds.md)
