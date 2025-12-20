# Module 5 : Optimisation Numérique

## 📋 Table des Matières
1. [Introduction](#introduction)
2. [Fondements Mathématiques](#fondements-mathématiques)
3. [Gradient et Dérivées](#gradient-et-dérivées)
4. [Descente de Gradient](#descente-de-gradient)
5. [Variantes de la Descente de Gradient](#variantes-de-la-descente-de-gradient)
6. [Algorithmes d'Optimisation Avancés](#algorithmes-doptimisation-avancés)
7. [Convergence et Taux d'Apprentissage](#convergence-et-taux-dapprentissage)
8. [Optimisation Sous Contraintes](#optimisation-sous-contraintes)
9. [Applications au Machine Learning](#applications-au-machine-learning)
10. [Exercices Pratiques](#exercices-pratiques)
11. [Résumé](#résumé)

---

## Introduction

L'**optimisation numérique** est au cœur du Machine Learning. Elle consiste à trouver les paramètres d'un modèle qui minimisent (ou maximisent) une fonction objectif.

### Problème d'Optimisation Général

$$
\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})
$$

où :
- $f : \mathbb{R}^n \to \mathbb{R}$ est la **fonction objectif** (ou fonction de coût)
- $\mathbf{x}$ est le vecteur de **paramètres** à optimiser
- Le minimum recherché est $\mathbf{x}^* = \arg\min_{\mathbf{x}} f(\mathbf{x})$

### Pourquoi l'Optimisation en ML ?

En Machine Learning, l'optimisation permet de :

1. **Entraîner des modèles** : Trouver les poids optimaux
2. **Minimiser l'erreur** : Réduire la fonction de perte
3. **Ajuster les hyperparamètres** : Optimiser les performances

**Exemples** :
- **Régression linéaire** : Minimiser l'erreur quadratique moyenne (MSE)
- **Régression logistique** : Minimiser la cross-entropy
- **Réseaux de neurones** : Minimiser la fonction de perte par backpropagation

### Types d'Optimisation

| Type | Description | Exemples |
|------|-------------|----------|
| **Sans contraintes** | $\min f(\mathbf{x})$ | Descente de gradient |
| **Avec contraintes** | $\min f(\mathbf{x})$ sujet à $g(\mathbf{x}) \leq 0$ | SVM, programmation quadratique |
| **Convexe** | $f$ est convexe | Régression linéaire |
| **Non-convexe** | $f$ n'est pas convexe | Réseaux de neurones profonds |

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configuration
plt.rcParams['figure.figsize'] = (12, 6)
```

---

## Fondements Mathématiques

### Conditions d'Optimalité

#### 1. Condition Nécessaire du Premier Ordre

Si $\mathbf{x}^*$ est un minimum local de $f$, alors :

$$
\nabla f(\mathbf{x}^*) = \mathbf{0}
$$

**Point critique** : Point où le gradient s'annule.

#### 2. Condition Suffisante du Second Ordre

Si $\nabla f(\mathbf{x}^*) = \mathbf{0}$ et la matrice hessienne $H(\mathbf{x}^*)$ est **définie positive**, alors $\mathbf{x}^*$ est un minimum local.

**Matrice hessienne** :
$$
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

### Convexité

Une fonction $f$ est **convexe** si pour tous $\mathbf{x}, \mathbf{y}$ et $\lambda \in [0, 1]$ :

$$
f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y})
$$

**Propriété importante** : Pour une fonction convexe, **tout minimum local est un minimum global**.

```python
import numpy as np
import matplotlib.pyplot as plt

# Fonction convexe
x = np.linspace(-5, 5, 1000)
f_convex = x**2

# Fonction non-convexe
f_non_convex = x**4 - 5*x**2 + 4

# Visualisation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(x, f_convex, 'b-', linewidth=2)
ax1.set_title('Fonction Convexe: f(x) = x²')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='k', linewidth=0.5)
ax1.axvline(0, color='k', linewidth=0.5)

ax2.plot(x, f_non_convex, 'r-', linewidth=2)
ax2.set_title('Fonction Non-Convexe: f(x) = x⁴ - 5x² + 4')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='k', linewidth=0.5)
ax2.axvline(0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

# Minima
print("Fonction convexe: un seul minimum global en x=0")
print("Fonction non-convexe: plusieurs minima locaux")
```

---

## Gradient et Dérivées

### Dérivée Partielle

Pour une fonction $f : \mathbb{R}^n \to \mathbb{R}$, la dérivée partielle par rapport à $x_i$ :

$$
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}
$$

### Gradient

Le **gradient** est le vecteur des dérivées partielles :

$$
\nabla f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

**Propriétés** :
- Le gradient **pointe dans la direction de plus forte croissance**
- $-\nabla f(\mathbf{x})$ pointe vers la **direction de plus forte décroissance**
- Le gradient est **perpendiculaire aux lignes de niveau**

### Calcul de Gradient : Exemples

#### Exemple 1 : Fonction Quadratique

$$
f(x) = x^2 - 4x + 4 = (x-2)^2
$$

**Gradient** :
$$
\nabla f(x) = \frac{df}{dx} = 2x - 4 = 2(x-2)
$$

**Minimum** : $\nabla f(x) = 0 \Rightarrow x^* = 2$

```python
# Définir la fonction et son gradient
def f(x):
    return (x - 2)**2

def grad_f(x):
    return 2 * (x - 2)

# Vérification
x_vals = np.linspace(-2, 6, 1000)
y_vals = f(x_vals)
grad_vals = grad_f(x_vals)

# Visualisation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Fonction objectif
ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = (x-2)²')
ax1.axvline(2, color='r', linestyle='--', label='Minimum en x=2')
ax1.scatter([2], [f(2)], color='r', s=100, zorder=5)
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_title('Fonction Objectif')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gradient
ax2.plot(x_vals, grad_vals, 'g-', linewidth=2, label="∇f(x) = 2(x-2)")
ax2.axhline(0, color='k', linewidth=0.5)
ax2.axvline(2, color='r', linestyle='--', label='Gradient = 0 en x=2')
ax2.set_xlabel('x')
ax2.set_ylabel('∇f(x)')
ax2.set_title('Gradient de la Fonction')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Minimum analytique: x* = 2")
print(f"f(x*) = {f(2)}")
print(f"∇f(x*) = {grad_f(2)}")
```

#### Exemple 2 : Fonction Multivariée

$$
f(x, y) = x^2 + y^2 - 2x - 4y + 5
$$

**Gradient** :
$$
\nabla f(x, y) = \begin{bmatrix}
\frac{\partial f}{\partial x} \\
\frac{\partial f}{\partial y}
\end{bmatrix} = \begin{bmatrix}
2x - 2 \\
2y - 4
\end{bmatrix}
$$

**Minimum** : $\nabla f = \mathbf{0}$ :
- $2x - 2 = 0 \Rightarrow x^* = 1$
- $2y - 4 = 0 \Rightarrow y^* = 2$

```python
def f_2d(x, y):
    return x**2 + y**2 - 2*x - 4*y + 5

def grad_f_2d(x, y):
    df_dx = 2*x - 2
    df_dy = 2*y - 4
    return np.array([df_dx, df_dy])

# Visualisation 3D
x = np.linspace(-2, 4, 100)
y = np.linspace(-1, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f_2d(X, Y)

fig = plt.figure(figsize=(14, 6))

# Surface 3D
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.scatter([1], [2], [f_2d(1, 2)], color='r', s=100, label='Minimum')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')
ax1.set_title('Fonction Objectif 2D')
ax1.legend()

# Contour plot avec gradient
ax2 = fig.add_subplot(122)
contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.scatter([1], [2], color='r', s=100, marker='*', label='Minimum (1, 2)')

# Champ de gradients
x_grid = np.linspace(-2, 4, 15)
y_grid = np.linspace(-1, 5, 15)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
U = 2*X_grid - 2
V = 2*Y_grid - 4
ax2.quiver(X_grid, Y_grid, -U, -V, alpha=0.5, color='red')

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Lignes de Niveau et Gradient (flèches rouges)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Minimum: (x*, y*) = (1, 2)")
print(f"f(x*, y*) = {f_2d(1, 2)}")
print(f"∇f(x*, y*) = {grad_f_2d(1, 2)}")
```

---

## Descente de Gradient

La **descente de gradient** (Gradient Descent) est l'algorithme d'optimisation le plus fondamental en Machine Learning.

### Principe

À partir d'un point initial $\mathbf{x}_0$, on itère :

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)
$$

où :
- $\alpha > 0$ est le **taux d'apprentissage** (learning rate)
- $\nabla f(\mathbf{x}_k)$ est le gradient au point $\mathbf{x}_k$

**Intuition** : On se déplace dans la direction opposée au gradient (descente).

### Algorithme

```
Entrées: x₀ (point initial), α (learning rate), M (nb d'itérations) ou ε (précision)
Sortie: x* (minimum approximatif)

1. Initialiser: x ← x₀, k ← 0
2. Répéter jusqu'à convergence:
   a. Calculer le gradient: g ← ∇f(x)
   b. Mettre à jour: x ← x - α·g
   c. k ← k + 1
3. Retourner x
```

**Critères d'arrêt** :
- **Nombre d'itérations** : $k \geq M$
- **Précision** : $\|\nabla f(\mathbf{x}_k)\| < \varepsilon$
- **Changement minimal** : $\|\mathbf{x}_{k+1} - \mathbf{x}_k\| < \varepsilon$

### Implémentation Python

```python
import numpy as np

def gradient_descent(f, grad_f, x0, alpha, max_iter=100, tol=1e-6, verbose=False):
    """
    Descente de gradient

    Paramètres:
    -----------
    f : fonction objectif
    grad_f : gradient de f
    x0 : point initial
    alpha : taux d'apprentissage (learning rate)
    max_iter : nombre maximal d'itérations
    tol : tolérance pour le critère d'arrêt
    verbose : afficher les informations

    Retourne:
    ---------
    x_opt : minimum trouvé
    f_opt : valeur de f au minimum
    history : historique des itérations
    """
    x = x0
    history = {
        'x': [x0],
        'f': [f(x0)],
        'grad_norm': [np.linalg.norm(grad_f(x0))]
    }

    for k in range(max_iter):
        # Calculer le gradient
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)

        # Critère d'arrêt
        if grad_norm < tol:
            if verbose:
                print(f"Convergence atteinte à l'itération {k}")
            break

        # Mise à jour
        x = x - alpha * grad

        # Historique
        history['x'].append(x)
        history['f'].append(f(x))
        history['grad_norm'].append(grad_norm)

        if verbose and k % 10 == 0:
            print(f"Iter {k}: x = {x:.6f}, f(x) = {f(x):.6f}, ||∇f|| = {grad_norm:.6f}")

    return x, f(x), history
```

### Exemple 1 : Fonction 1D

```python
# Fonction: f(x) = (x-2)²
def f(x):
    return (x - 2)**2

def grad_f(x):
    return 2 * (x - 2)

# Optimisation
x0 = 6.0  # Point initial
alpha = 0.1  # Learning rate
max_iter = 50

x_opt, f_opt, history = gradient_descent(f, grad_f, x0, alpha, max_iter, verbose=True)

print(f"\n{'='*60}")
print(f"Résultat:")
print(f"x* = {x_opt:.6f} (analytique: 2.0)")
print(f"f(x*) = {f_opt:.6f}")
print(f"Nombre d'itérations: {len(history['x']) - 1}")

# Visualisation
x_vals = np.linspace(-1, 7, 1000)
y_vals = f(x_vals)

plt.figure(figsize=(14, 5))

# Fonction et trajectoire
plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = (x-2)²')
plt.scatter(history['x'], history['f'], c=range(len(history['x'])),
            cmap='Reds', s=100, edgecolors='black', label='Itérations', zorder=5)
plt.plot(history['x'], history['f'], 'r--', alpha=0.5, linewidth=1)
plt.scatter([2], [0], color='green', s=200, marker='*',
            label='Minimum analytique', zorder=10)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Trajectoire de la Descente de Gradient')
plt.legend()
plt.grid(True, alpha=0.3)

# Convergence
plt.subplot(1, 2, 2)
plt.plot(history['f'], 'r-', linewidth=2, marker='o')
plt.xlabel('Itération')
plt.ylabel('f(x)')
plt.title('Évolution de la Fonction Objectif')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.show()
```

### Exemple 2 : Fonction 2D

```python
def f_2d_vec(x):
    """f(x, y) = x² + y² - 2x - 4y + 5"""
    return x[0]**2 + x[1]**2 - 2*x[0] - 4*x[1] + 5

def grad_f_2d_vec(x):
    """Gradient de f"""
    return np.array([2*x[0] - 2, 2*x[1] - 4])

# Optimisation
x0 = np.array([5.0, -1.0])
alpha = 0.3
max_iter = 30

x_opt, f_opt, history = gradient_descent(f_2d_vec, grad_f_2d_vec, x0, alpha,
                                          max_iter, verbose=True)

print(f"\nRésultat:")
print(f"x* = {x_opt} (analytique: [1, 2])")
print(f"f(x*) = {f_opt:.6f}")

# Visualisation
x = np.linspace(-1, 6, 100)
y = np.linspace(-3, 5, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2 - 2*X - 4*Y + 5

plt.figure(figsize=(12, 5))

# Contour avec trajectoire
plt.subplot(1, 2, 1)
contour = plt.contour(X, Y, Z, levels=30, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)

# Trajectoire
history_x = np.array(history['x'])
plt.plot(history_x[:, 0], history_x[:, 1], 'r-o', linewidth=2,
         markersize=8, label='Trajectoire')
plt.scatter([1], [2], color='green', s=200, marker='*',
            label='Minimum', zorder=10)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectoire de la Descente de Gradient (2D)')
plt.legend()
plt.grid(True, alpha=0.3)

# Convergence
plt.subplot(1, 2, 2)
plt.plot(history['f'], 'r-', linewidth=2, marker='o')
plt.xlabel('Itération')
plt.ylabel('f(x, y)')
plt.title('Convergence')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.show()
```

---

## Variantes de la Descente de Gradient

En Machine Learning, on traite souvent des datasets très larges. Calculer le gradient sur toutes les données peut être coûteux. D'où les variantes.

### 1. Batch Gradient Descent (BGD)

**Principe** : Utilise **toutes les données** pour calculer le gradient.

$$
\mathbf{w}_{k+1} = \mathbf{w}_k - \alpha \frac{1}{N} \sum_{i=1}^{N} \nabla L(\mathbf{w}_k; \mathbf{x}_i, y_i)
$$

**Avantages** :
- Convergence stable
- Garantie de convergence vers minimum global (si fonction convexe)

**Inconvénients** :
- Lent pour gros datasets
- Peut être piégé dans des minima locaux (non-convexe)

### 2. Stochastic Gradient Descent (SGD)

**Principe** : Utilise **une seule donnée** aléatoire à chaque itération.

$$
\mathbf{w}_{k+1} = \mathbf{w}_k - \alpha \nabla L(\mathbf{w}_k; \mathbf{x}_i, y_i)
$$

où $i$ est tiré aléatoirement.

**Avantages** :
- Très rapide
- Peut échapper aux minima locaux (bruit)
- Permet apprentissage en ligne

**Inconvénients** :
- Convergence bruitée
- Nécessite un bon réglage du learning rate

### 3. Mini-Batch Gradient Descent

**Principe** : Compromis entre BGD et SGD. Utilise un **petit batch** de données.

$$
\mathbf{w}_{k+1} = \mathbf{w}_k - \alpha \frac{1}{B} \sum_{i \in \text{batch}} \nabla L(\mathbf{w}_k; \mathbf{x}_i, y_i)
$$

**Taille de batch typique** : 32, 64, 128, 256

**Avantages** :
- Bon compromis vitesse/stabilité
- Utilise efficacement le hardware (GPU)
- Convergence plus stable que SGD

```python
import numpy as np
import matplotlib.pyplot as plt

# Générer des données synthétiques
np.random.seed(42)
N = 1000
X = np.random.randn(N, 1)
y = 2 * X + 3 + np.random.randn(N, 1) * 0.5

# Fonction de coût (MSE)
def mse(w, X, y):
    """Mean Squared Error"""
    predictions = X @ w
    return np.mean((predictions - y)**2)

def grad_mse(w, X, y):
    """Gradient de MSE"""
    N = len(y)
    predictions = X @ w
    return (2/N) * X.T @ (predictions - y)

# Batch Gradient Descent
def batch_gd(X, y, alpha, max_iter):
    w = np.zeros((X.shape[1], 1))
    history = []

    for _ in range(max_iter):
        grad = grad_mse(w, X, y)
        w = w - alpha * grad
        history.append(mse(w, X, y))

    return w, history

# Stochastic Gradient Descent
def sgd(X, y, alpha, max_iter):
    w = np.zeros((X.shape[1], 1))
    history = []
    N = len(y)

    for _ in range(max_iter):
        i = np.random.randint(0, N)
        Xi = X[i:i+1]
        yi = y[i:i+1]
        grad = 2 * Xi.T @ (Xi @ w - yi)
        w = w - alpha * grad
        history.append(mse(w, X, y))

    return w, history

# Mini-Batch Gradient Descent
def minibatch_gd(X, y, alpha, max_iter, batch_size=32):
    w = np.zeros((X.shape[1], 1))
    history = []
    N = len(y)

    for _ in range(max_iter):
        indices = np.random.choice(N, batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]
        grad = grad_mse(w, X_batch, y_batch)
        w = w - alpha * grad
        history.append(mse(w, X, y))

    return w, history

# Entraînement
alpha = 0.01
max_iter = 100

w_batch, hist_batch = batch_gd(X, y, alpha, max_iter)
w_sgd, hist_sgd = sgd(X, y, alpha, max_iter * N)  # Plus d'itérations pour SGD
w_minibatch, hist_minibatch = minibatch_gd(X, y, alpha, max_iter * 10, batch_size=32)

# Visualisation
plt.figure(figsize=(12, 5))
plt.plot(hist_batch, label='Batch GD', linewidth=2)
plt.plot(hist_sgd, label='SGD', alpha=0.7)
plt.plot(hist_minibatch, label='Mini-Batch GD', linewidth=2)
plt.xlabel('Itération')
plt.ylabel('MSE')
plt.title('Comparaison des Variantes de Descente de Gradient')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()

print("Poids optimaux:")
print(f"Batch GD: {w_batch.flatten()}")
print(f"SGD: {w_sgd.flatten()}")
print(f"Mini-Batch GD: {w_minibatch.flatten()}")
print(f"Analytique: [2.0] (environ)")
```

---

## Algorithmes d'Optimisation Avancés

### 1. Momentum

**Principe** : Accumuler une "vitesse" pour accélérer dans les directions cohérentes.

$$
\begin{align}
\mathbf{v}_{k+1} &= \beta \mathbf{v}_k + (1-\beta) \nabla f(\mathbf{w}_k) \\
\mathbf{w}_{k+1} &= \mathbf{w}_k - \alpha \mathbf{v}_{k+1}
\end{align}
$$

où $\beta \in [0, 1]$ (typiquement 0.9) est le coefficient de momentum.

**Avantages** :
- Accélère la convergence
- Réduit les oscillations
- Aide à sortir des minima locaux peu profonds

```python
def gradient_descent_momentum(f, grad_f, x0, alpha, beta=0.9, max_iter=100, tol=1e-6):
    """Descente de gradient avec momentum"""
    x = x0
    v = 0  # Vitesse initiale
    history = {'x': [x0], 'f': [f(x0)]}

    for k in range(max_iter):
        grad = grad_f(x)

        if np.linalg.norm(grad) < tol:
            break

        # Mise à jour avec momentum
        v = beta * v + (1 - beta) * grad
        x = x - alpha * v

        history['x'].append(x)
        history['f'].append(f(x))

    return x, f(x), history

# Comparaison GD classique vs Momentum
x0 = 6.0
alpha = 0.1

# Sans momentum
x_gd, _, hist_gd = gradient_descent(f, grad_f, x0, alpha, max_iter=50)

# Avec momentum
x_mom, _, hist_mom = gradient_descent_momentum(f, grad_f, x0, alpha, beta=0.9, max_iter=50)

# Visualisation
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
x_vals = np.linspace(0, 7, 1000)
plt.plot(x_vals, f(x_vals), 'b-', linewidth=2, alpha=0.5)
plt.plot(hist_gd['x'], hist_gd['f'], 'ro-', label='GD classique', markersize=6)
plt.plot(hist_mom['x'], hist_mom['f'], 'go-', label='GD + Momentum', markersize=6)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Trajectoires de Convergence')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(hist_gd['f'], 'r-', label='GD classique', linewidth=2)
plt.plot(hist_mom['f'], 'g-', label='GD + Momentum', linewidth=2)
plt.xlabel('Itération')
plt.ylabel('f(x)')
plt.title('Vitesse de Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.show()

print(f"Itérations GD classique: {len(hist_gd['x'])}")
print(f"Itérations GD + Momentum: {len(hist_mom['x'])}")
```

### 2. RMSprop

**Principe** : Adapter le learning rate pour chaque paramètre en fonction de l'historique des gradients.

$$
\begin{align}
s_{k+1} &= \beta s_k + (1-\beta) (\nabla f(\mathbf{w}_k))^2 \\
\mathbf{w}_{k+1} &= \mathbf{w}_k - \frac{\alpha}{\sqrt{s_{k+1} + \epsilon}} \nabla f(\mathbf{w}_k)
\end{align}
$$

**Avantages** :
- Learning rate adaptatif
- Fonctionne bien sur problèmes non-stationnaires

### 3. Adam (Adaptive Moment Estimation)

**Principe** : Combine Momentum et RMSprop.

$$
\begin{align}
m_{k+1} &= \beta_1 m_k + (1-\beta_1) \nabla f(\mathbf{w}_k) \\
v_{k+1} &= \beta_2 v_k + (1-\beta_2) (\nabla f(\mathbf{w}_k))^2 \\
\hat{m}_{k+1} &= \frac{m_{k+1}}{1 - \beta_1^{k+1}} \\
\hat{v}_{k+1} &= \frac{v_{k+1}}{1 - \beta_2^{k+1}} \\
\mathbf{w}_{k+1} &= \mathbf{w}_k - \alpha \frac{\hat{m}_{k+1}}{\sqrt{\hat{v}_{k+1}} + \epsilon}
\end{align}
$$

**Hyperparamètres typiques** :
- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- $\epsilon = 10^{-8}$
- $\alpha = 0.001$

**Avantages** :
- Très performant en pratique
- Peu sensible aux hyperparamètres
- Algorithme par défaut pour Deep Learning

```python
def adam(f, grad_f, x0, alpha=0.001, beta1=0.9, beta2=0.999,
         eps=1e-8, max_iter=100, tol=1e-6):
    """Optimiseur Adam"""
    x = x0
    m = 0  # Premier moment
    v = 0  # Second moment
    history = {'x': [x0], 'f': [f(x0)]}

    for k in range(1, max_iter + 1):
        grad = grad_f(x)

        if np.linalg.norm(grad) < tol:
            break

        # Mise à jour des moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)

        # Correction du biais
        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)

        # Mise à jour des paramètres
        x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)

        history['x'].append(x)
        history['f'].append(f(x))

    return x, f(x), history

# Comparaison
x0 = 6.0

x_gd, _, hist_gd = gradient_descent(f, grad_f, x0, alpha=0.1, max_iter=50)
x_mom, _, hist_mom = gradient_descent_momentum(f, grad_f, x0, alpha=0.1, max_iter=50)
x_adam, _, hist_adam = adam(f, grad_f, x0, alpha=0.5, max_iter=50)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
x_vals = np.linspace(0, 7, 1000)
plt.plot(x_vals, f(x_vals), 'k-', linewidth=2, alpha=0.3)
plt.plot(hist_gd['x'], hist_gd['f'], 'r.-', label='GD', markersize=8)
plt.plot(hist_mom['x'], hist_mom['f'], 'g.-', label='Momentum', markersize=8)
plt.plot(hist_adam['x'], hist_adam['f'], 'b.-', label='Adam', markersize=8)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Trajectoires')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(hist_gd['f'], 'r-', label='GD', linewidth=2)
plt.plot(hist_mom['f'], 'g-', label='Momentum', linewidth=2)
plt.plot(hist_adam['f'], 'b-', label='Adam', linewidth=2)
plt.xlabel('Itération')
plt.ylabel('f(x)')
plt.title('Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.show()
```

---

## Convergence et Taux d'Apprentissage

### Choix du Learning Rate

Le **learning rate** $\alpha$ est crucial :

**$\alpha$ trop petit** :
- Convergence très lente
- Peut stagner avant le minimum

**$\alpha$ trop grand** :
- Divergence
- Oscillations autour du minimum

**$\alpha$ optimal** :
- Convergence rapide et stable

```python
# Visualisation de l'effet du learning rate
x0 = 6.0
alphas = [0.01, 0.1, 0.5, 1.0, 1.5]
max_iter = 50

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

x_vals = np.linspace(-5, 8, 1000)
y_vals = f(x_vals)

for idx, alpha in enumerate(alphas):
    ax = axes[idx]

    try:
        x_opt, _, history = gradient_descent(f, grad_f, x0, alpha, max_iter)

        ax.plot(x_vals, y_vals, 'b-', linewidth=2, alpha=0.5)
        ax.plot(history['x'], history['f'], 'ro-', markersize=4)
        ax.scatter([2], [0], color='green', s=100, marker='*', zorder=10)
        ax.set_title(f'α = {alpha}')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1, 20)

    except:
        ax.text(0.5, 0.5, 'DIVERGENCE', ha='center', va='center',
                transform=ax.transAxes, fontsize=20, color='red')
        ax.set_title(f'α = {alpha}')

axes[-1].axis('off')
plt.tight_layout()
plt.show()
```

### Learning Rate Scheduling

**Idée** : Réduire progressivement $\alpha$ au cours de l'entraînement.

**Stratégies** :

1. **Step Decay** :
   $$\alpha_k = \alpha_0 \cdot \gamma^{\lfloor k/s \rfloor}$$

2. **Exponential Decay** :
   $$\alpha_k = \alpha_0 \cdot e^{-\lambda k}$$

3. **1/t Decay** :
   $$\alpha_k = \frac{\alpha_0}{1 + \lambda k}$$

```python
def step_decay(alpha0, k, drop=0.5, epochs_drop=10):
    """Step decay"""
    return alpha0 * (drop ** np.floor(k / epochs_drop))

def exp_decay(alpha0, k, decay_rate=0.1):
    """Exponential decay"""
    return alpha0 * np.exp(-decay_rate * k)

def time_decay(alpha0, k, decay_rate=0.01):
    """1/t decay"""
    return alpha0 / (1 + decay_rate * k)

# Visualisation
iterations = np.arange(0, 100)
alpha0 = 0.1

plt.figure(figsize=(12, 5))
plt.plot(iterations, [step_decay(alpha0, k) for k in iterations],
         label='Step Decay', linewidth=2)
plt.plot(iterations, [exp_decay(alpha0, k) for k in iterations],
         label='Exponential Decay', linewidth=2)
plt.plot(iterations, [time_decay(alpha0, k) for k in iterations],
         label='1/t Decay', linewidth=2)
plt.xlabel('Itération')
plt.ylabel('Learning Rate')
plt.title('Stratégies de Learning Rate Scheduling')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Optimisation Sous Contraintes

### Problème avec Contraintes

$$
\begin{align}
\min_{\mathbf{x}} \quad & f(\mathbf{x}) \\
\text{sujet à} \quad & g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m \\
& h_j(\mathbf{x}) = 0, \quad j = 1, \ldots, p
\end{align}
$$

### Méthode de Lagrange

**Lagrangien** :
$$
\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = f(\mathbf{x}) + \sum_{i=1}^{m} \lambda_i g_i(\mathbf{x}) + \sum_{j=1}^{p} \mu_j h_j(\mathbf{x})
$$

**Conditions KKT (Karush-Kuhn-Tucker)** : Nécessaires pour un minimum.

### Exemple : Régression Ridge (L2)

$$
\min_{\mathbf{w}} \quad \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|^2
$$

Le terme $\lambda \|\mathbf{w}\|^2$ pénalise les poids trop grands (régularisation).

```python
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

# Générer données
X, y = make_regression(n_samples=100, n_features=10, noise=10, random_state=42)

# Ridge Regression
model = Ridge(alpha=1.0)
model.fit(X, y)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

---

## Applications au Machine Learning

### 1. Régression Linéaire

**Fonction de coût (MSE)** :
$$
J(\mathbf{w}) = \frac{1}{2N} \sum_{i=1}^{N} (h_{\mathbf{w}}(\mathbf{x}_i) - y_i)^2
$$

**Gradient** :
$$
\nabla J(\mathbf{w}) = \frac{1}{N} \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{y})
$$

### 2. Régression Logistique

**Fonction de coût (Cross-Entropy)** :
$$
J(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(h_{\mathbf{w}}(\mathbf{x}_i)) + (1-y_i) \log(1 - h_{\mathbf{w}}(\mathbf{x}_i)) \right]
$$

où $h_{\mathbf{w}}(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x})$ et $\sigma(z) = \frac{1}{1 + e^{-z}}$ (fonction sigmoïde).

### 3. Réseaux de Neurones

**Backpropagation** : Calcul efficace du gradient par la règle de la chaîne.

---

## Exercices Pratiques

### Exercice 1 : Optimisation Analytique

**Énoncé** : Résoudre analytiquement :
$$
\min_{x} f(x) = x^2 - 4x + 4
$$

**Solution** :

```python
# 1. Analytique
# f(x) = x² - 4x + 4 = (x-2)²
# f'(x) = 2x - 4
# f'(x) = 0 => x* = 2

print("Solution analytique:")
print("x* = 2")
print("f(x*) = 0")

# 2. Fonction et gradient
def f_ex1(x):
    return x**2 - 4*x + 4

def grad_f_ex1(x):
    return 2*x - 4

# 3. Vérification
x_opt = 2
print(f"\nVérification:")
print(f"f({x_opt}) = {f_ex1(x_opt)}")
print(f"∇f({x_opt}) = {grad_f_ex1(x_opt)}")
```

### Exercice 2 : Descente de Gradient

**Énoncé** : Implémenter une descente de gradient pour minimiser $f(x) = x^2 - 4x + 4$.

**Solution** :

```python
def gradient_descent_simple(f, grad_f, x0, alpha, max_iter):
    """Descente de gradient simple"""
    x = x0
    history = []

    for k in range(max_iter):
        grad = grad_f(x)
        x = x - alpha * grad
        history.append((x, f(x)))

    return x, f(x), history

# Test
x0 = 10.0
alpha = 0.1
max_iter = 50

x_opt, f_opt, history = gradient_descent_simple(f_ex1, grad_f_ex1, x0, alpha, max_iter)

print(f"Résultat:")
print(f"x* = {x_opt:.6f}")
print(f"f(x*) = {f_opt:.6f}")
print(f"Nombre d'itérations: {len(history)}")

# Visualisation
x_vals = np.linspace(-2, 12, 1000)
y_vals = f_ex1(x_vals)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, 'b-', linewidth=2)
history_x = [h[0] for h in history]
history_f = [h[1] for h in history]
plt.scatter(history_x, history_f, c=range(len(history)), cmap='Reds', s=50, zorder=5)
plt.plot(history_x, history_f, 'r--', alpha=0.5)
plt.scatter([2], [0], color='green', s=200, marker='*', zorder=10)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Trajectoire')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(len(history_f)), history_f, 'r-', linewidth=2, marker='o')
plt.xlabel('Itération')
plt.ylabel('f(x)')
plt.title('Convergence')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.show()
```

### Exercice 3 : Impact du Learning Rate

**Énoncé** : Tester différents learning rates et observer la convergence.

**Solution** :

```python
alphas_test = [0.01, 0.1, 0.5, 1.0, 1.2]
x0 = 10.0
max_iter = 30

results = {}

for alpha in alphas_test:
    try:
        x_opt, f_opt, history = gradient_descent_simple(f_ex1, grad_f_ex1,
                                                         x0, alpha, max_iter)
        results[alpha] = history
    except:
        results[alpha] = None

# Visualisation
plt.figure(figsize=(14, 6))

for alpha, history in results.items():
    if history is not None:
        history_f = [h[1] for h in history]
        plt.plot(range(len(history_f)), history_f, marker='o',
                 label=f'α = {alpha}', linewidth=2)

plt.xlabel('Itération')
plt.ylabel('f(x)')
plt.title('Impact du Learning Rate sur la Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()
```

### Exercice 4 : Régression Linéaire avec GD

**Énoncé** : Implémenter une régression linéaire avec descente de gradient.

**Solution** :

```python
# Générer données
np.random.seed(42)
N = 100
X = 2 * np.random.rand(N, 1)
y = 4 + 3 * X + np.random.randn(N, 1) * 0.5

# Ajouter colonne de 1 pour le biais
X_b = np.c_[np.ones((N, 1)), X]

# Fonction de coût
def mse_cost(theta, X, y):
    N = len(y)
    predictions = X @ theta
    return (1/(2*N)) * np.sum((predictions - y)**2)

# Gradient
def mse_gradient(theta, X, y):
    N = len(y)
    predictions = X @ theta
    return (1/N) * X.T @ (predictions - y)

# Descente de gradient
theta_init = np.zeros((2, 1))
alpha = 0.1
max_iter = 1000

theta = theta_init
cost_history = []

for i in range(max_iter):
    grad = mse_gradient(theta, X_b, y)
    theta = theta - alpha * grad
    cost_history.append(mse_cost(theta, X_b, y))

print("Paramètres optimaux:")
print(f"θ₀ (biais) = {theta[0, 0]:.4f} (attendu: ≈ 4)")
print(f"θ₁ (pente) = {theta[1, 0]:.4f} (attendu: ≈ 3)")

# Visualisation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Données et régression
ax1.scatter(X, y, alpha=0.6, label='Données')
X_plot = np.array([[0], [2]])
X_plot_b = np.c_[np.ones((2, 1)), X_plot]
y_pred = X_plot_b @ theta
ax1.plot(X_plot, y_pred, 'r-', linewidth=2, label='Régression')
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('Régression Linéaire')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Coût
ax2.plot(cost_history, 'b-', linewidth=2)
ax2.set_xlabel('Itération')
ax2.set_ylabel('Coût (MSE)')
ax2.set_title('Évolution du Coût')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.show()
```

---

## Résumé

### Points Clés à Retenir

#### 1. Optimisation

| Concept | Description |
|---------|-------------|
| **Problème** | $\min_{\mathbf{x}} f(\mathbf{x})$ |
| **Gradient** | Direction de plus forte croissance |
| **Minimum** | $\nabla f(\mathbf{x}^*) = \mathbf{0}$ |
| **Convexité** | Minimum local = minimum global |

#### 2. Descente de Gradient

**Formule** :
$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)
$$

**Paramètres** :
- $\alpha$ : Learning rate (crucial!)
- Critère d'arrêt : Nb itérations ou $\|\nabla f\| < \varepsilon$

#### 3. Variantes

| Méthode | Mise à jour | Avantage |
|---------|-------------|----------|
| **Batch GD** | Toutes les données | Stable |
| **SGD** | 1 donnée aléatoire | Rapide |
| **Mini-Batch** | Petit batch | Compromis |
| **Momentum** | Accumule vitesse | Accélère |
| **Adam** | Adaptatif | Performant |

#### 4. Learning Rate

- **Trop petit** : Convergence lente
- **Trop grand** : Divergence
- **Optimal** : Convergence rapide
- **Scheduling** : Réduire progressivement

#### 5. Applications ML

| Modèle | Fonction de Coût |
|--------|------------------|
| Régression linéaire | MSE : $\frac{1}{2N}\sum(h(\mathbf{x}_i) - y_i)^2$ |
| Régression logistique | Cross-Entropy |
| Réseaux de neurones | Backpropagation |

### Formules Essentielles

```
Gradient: ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ

Descente de gradient: xₖ₊₁ = xₖ - α·∇f(xₖ)

Momentum: vₖ₊₁ = β·vₖ + (1-β)·∇f(xₖ)
          xₖ₊₁ = xₖ - α·vₖ₊₁

MSE: J(w) = (1/2N)·Σ(ŷᵢ - yᵢ)²

Gradient MSE: ∇J(w) = (1/N)·Xᵀ(Xw - y)
```

### Bibliothèques Python

```python
import numpy as np                  # Calculs numériques
import matplotlib.pyplot as plt     # Visualisation
from scipy.optimize import minimize # Optimisation avancée
import torch                        # PyTorch (Deep Learning)
from sklearn.linear_model import *  # Modèles ML avec optimisation
```

### Checklist Optimisation

- [ ] Définir la fonction objectif $f(\mathbf{x})$
- [ ] Calculer le gradient $\nabla f(\mathbf{x})$
- [ ] Choisir le learning rate $\alpha$
- [ ] Initialiser les paramètres $\mathbf{x}_0$
- [ ] Itérer jusqu'à convergence
- [ ] Vérifier la convergence (gradient, coût)
- [ ] Ajuster les hyperparamètres si nécessaire

### Prochaine Étape

**Module 6 : Apprentissage Supervisé** - Régression, classification, arbres de décision, SVM

---

**Navigation :**
- [⬅️ Module 4 : Statistiques Descriptives](04_Statistiques_Descriptives.md)
- [🏠 Retour au Sommaire](README.md)
- [➡️ Module 6 : Apprentissage Supervisé](06_Apprentissage_Supervise.md)
