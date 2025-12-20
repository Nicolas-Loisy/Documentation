# Module 2 : Algèbre Linéaire pour le Machine Learning

## 📋 Table des Matières

1. [Introduction](#introduction)
2. [Vecteurs](#vecteurs)
3. [Opérations Vectorielles](#opérations-vectorielles)
4. [Normes et Produit Scalaire](#normes-et-produit-scalaire)
5. [Matrices](#matrices)
6. [Opérations Matricielles](#opérations-matricielles)
7. [Matrices Spéciales](#matrices-spéciales)
8. [Déterminant et Inversion](#déterminant-et-inversion)
9. [Valeurs et Vecteurs Propres](#valeurs-et-vecteurs-propres)
10. [Décomposition SVD](#décomposition-svd)
11. [Projections Orthogonales](#projections-orthogonales)
12. [Applications au Machine Learning](#applications-au-machine-learning)
13. [Exercices Pratiques](#exercices-pratiques)
14. [Résumé](#résumé)

---

## Introduction

L'**algèbre linéaire** est le fondement mathématique du Machine Learning. Elle permet de :

- Représenter et manipuler des datasets (matrices de données)
- Effectuer des transformations linéaires
- Résoudre des systèmes d'équations
- Réduire la dimensionnalité (PCA)
- Comprendre les réseaux de neurones (multiplication matricielle)

**Pourquoi est-ce crucial pour le ML ?**

- Les données sont représentées sous forme de vecteurs et matrices
- Les modèles ML effectuent des opérations matricielles
- L'optimisation repose sur le calcul vectoriel
- La décomposition matricielle permet la compression et l'interprétation

---

## Vecteurs

### Définition

Un **vecteur** $\mathbf{x}$ de dimension $p$ est un tableau ordonné de $p$ nombres réels :

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_p \end{bmatrix} \in \mathbb{R}^p
$$

**Notation** : On écrit $\mathbf{x} \in \mathbb{R}^p$ pour dire que $\mathbf{x}$ est un vecteur à $p$ dimensions.

### Transposée

La **transposée** d'un vecteur colonne est un vecteur ligne :

$$
\mathbf{x}^t = [x_1, x_2, \ldots, x_p]
$$

### Exemples Python

```python
import numpy as np

# Créer un vecteur colonne
x = np.array([1, 2, 3, 4, 5])
print("Vecteur x:")
print(x)
print(f"Dimension: {x.shape}")  # (5,)

# Vecteur ligne (transposée conceptuelle)
x_row = x.reshape(1, -1)
print("\nVecteur ligne:")
print(x_row)
print(f"Dimension: {x_row.shape}")  # (1, 5)

# Véritable vecteur colonne 2D
x_col = x.reshape(-1, 1)
print("\nVecteur colonne:")
print(x_col)
print(f"Dimension: {x_col.shape}")  # (5, 1)
```

**Résultat :**

```
Vecteur x:
[1 2 3 4 5]
Dimension: (5,)

Vecteur ligne:
[[1 2 3 4 5]]
Dimension: (1, 5)

Vecteur colonne:
[[1]
 [2]
 [3]
 [4]
 [5]]
Dimension: (5, 1)
```

---

## Opérations Vectorielles

### Addition Vectorielle

Soit $\mathbf{x}, \mathbf{y} \in \mathbb{R}^p$ :

$$
\mathbf{x} + \mathbf{y} = \begin{bmatrix} x_1 + y_1 \\ x_2 + y_2 \\ \vdots \\ x_p + y_p \end{bmatrix}
$$

### Multiplication par un Scalaire

Soit $\alpha \in \mathbb{R}$ :

$$
\alpha \mathbf{x} = \begin{bmatrix} \alpha x_1 \\ \alpha x_2 \\ \vdots \\ \alpha x_p \end{bmatrix}
$$

### Transformation Affine

$$
\mathbf{x} + \alpha \mathbf{y} = \begin{bmatrix} x_1 + \alpha y_1 \\ x_2 + \alpha y_2 \\ \vdots \\ x_p + \alpha y_p \end{bmatrix}
$$

### Exemples Python

```python
import numpy as np

# Définir deux vecteurs
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# Addition
print("x + y =", x + y)  # [5 7 9]

# Multiplication par scalaire
alpha = 2
print(f"{alpha}x =", alpha * x)  # [2 4 6]

# Transformation affine
print(f"x + {alpha}y =", x + alpha * y)  # [9 12 15]

# Soustraction
print("x - y =", x - y)  # [-3 -3 -3]
```

---

## Normes et Produit Scalaire

### Normes Vectorielles

Une **norme** $\|\cdot\| : \mathbb{R}^p \to \mathbb{R}_+$ satisfait :

1. **Positivité** : $\|\mathbf{x}\| = 0 \Leftrightarrow \mathbf{x} = \mathbf{0}$
2. **Inégalité triangulaire** : $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$
3. **Homogénéité absolue** : $\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|$

#### 1. Norme L1 (Manhattan)

$$
\|\mathbf{x}\|_1 = \sum_{i=1}^{p} |x_i|
$$

**Usage ML** : Régularisation Lasso, sélection de features

#### 2. Norme L2 (Euclidienne)

$$
\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{p} x_i^2}
$$

**Usage ML** : Distance euclidienne, régularisation Ridge

#### 3. Norme L∞ (Maximum)

$$
\|\mathbf{x}\|_\infty = \max_{i} |x_i|
$$

### Vecteur Unitaire

Le **vecteur unitaire** (normalisé) de $\mathbf{x}$ :

$$
\mathbf{u} = \frac{1}{\|\mathbf{x}\|_2} \mathbf{x}
$$

Propriété : $\|\mathbf{u}\|_2 = 1$

### Produit Scalaire (Dot Product)

Le **produit scalaire** entre $\mathbf{x}$ et $\mathbf{y}$ :

$$
\mathbf{x} \cdot \mathbf{y} = \mathbf{x}^t \mathbf{y} = \sum_{i=1}^{p} x_i y_i = \|\mathbf{x}\| \|\mathbf{y}\| \cos(\theta)
$$

où $\theta$ est l'angle entre les deux vecteurs.

**Propriété importante** :

$$
\|\mathbf{x}\|_2 = \sqrt{\mathbf{x}^t \mathbf{x}} = \sqrt{\mathbf{x} \cdot \mathbf{x}}
$$

### Inégalités Importantes

**Inégalité de Cauchy-Schwarz** :

$$
|\mathbf{x} \cdot \mathbf{y}| \leq \|\mathbf{x}\| \|\mathbf{y}\|
$$

**Relations entre normes** :

$$
\|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1 \leq \sqrt{p} \|\mathbf{x}\|_2
$$

$$
\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1 \leq p \|\mathbf{x}\|_\infty
$$

### Exemples Python

```python
import numpy as np

x = np.array([3, 4])
y = np.array([1, 2])

# Normes
print("Norme L1:", np.linalg.norm(x, ord=1))      # 7.0
print("Norme L2:", np.linalg.norm(x, ord=2))      # 5.0
print("Norme L∞:", np.linalg.norm(x, ord=np.inf)) # 4.0

# Vecteur unitaire
u = x / np.linalg.norm(x)
print("\nVecteur unitaire:", u)
print("Norme du vecteur unitaire:", np.linalg.norm(u))  # 1.0

# Produit scalaire
dot_product = np.dot(x, y)
print("\nProduit scalaire x·y:", dot_product)  # 11

# Vérification avec la formule
dot_formula = x[0]*y[0] + x[1]*y[1]
print("Vérification:", dot_formula)  # 11

# Angle entre vecteurs
cos_theta = dot_product / (np.linalg.norm(x) * np.linalg.norm(y))
theta_rad = np.arccos(cos_theta)
theta_deg = np.degrees(theta_rad)
print(f"\nAngle entre x et y: {theta_deg:.2f}°")
```

---

## Matrices

### Définition

Une **matrice** $\mathbf{X}$ de $n$ lignes et $p$ colonnes ($\mathbf{X} \in \mathbb{R}^{n \times p}$) :

$$
\mathbf{X} = \begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1p} \\
x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{np}
\end{bmatrix}
$$

**Notation compacte** : $\mathbf{X} = (x_{ij})_{\substack{i=1,\ldots,n \\ j=1,\ldots,p}}$

### Matrice Carrée

Quand $n = p$, la matrice est dite **carrée**.

### Transposée

La **transposée** $\mathbf{X}^t \in \mathbb{R}^{p \times n}$ intervertit lignes et colonnes :

$$
\mathbf{X}^t = \begin{bmatrix}
x_{11} & x_{21} & \cdots & x_{n1} \\
x_{12} & x_{22} & \cdots & x_{n2} \\
\vdots & \vdots & \ddots & \vdots \\
x_{1p} & x_{2p} & \cdots & x_{np}
\end{bmatrix}
$$

### Exemples Python

```python
import numpy as np

# Créer une matrice 3x4
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

print("Matrice A:")
print(A)
print(f"Dimension: {A.shape}")  # (3, 4)

# Transposée
A_t = A.T
print("\nTransposée A^t:")
print(A_t)
print(f"Dimension: {A_t.shape}")  # (4, 3)

# Accéder aux éléments
print(f"\nA[0,0] = {A[0,0]}")    # 1
print(f"A[2,3] = {A[2,3]}")      # 12

# Extraire une ligne
print(f"\nLigne 1: {A[1, :]}")   # [5 6 7 8]

# Extraire une colonne
print(f"Colonne 2: {A[:, 2]}")   # [3 7 11]

# Créer matrices spéciales
zeros = np.zeros((2, 3))
ones = np.ones((3, 2))
identity = np.eye(4)
diagonal = np.diag([1, 2, 3, 4])
random = np.random.randn(3, 3)

print("\nMatrice identité 4x4:")
print(identity)
```

---

## Opérations Matricielles

### Addition et Soustraction

Soit $\mathbf{X}, \mathbf{Y} \in \mathbb{R}^{n \times p}$ et $\alpha \in \mathbb{R}$ :

$$
\mathbf{X} + \alpha \mathbf{Y} = \begin{bmatrix}
x_{11} + \alpha y_{11} & \cdots & x_{1p} + \alpha y_{1p} \\
\vdots & \ddots & \vdots \\
x_{n1} + \alpha y_{n1} & \cdots & x_{np} + \alpha y_{np}
\end{bmatrix}
$$

### Multiplication Matrice-Vecteur

Soit $\mathbf{X} \in \mathbb{R}^{n \times p}$ et $\boldsymbol{\theta} \in \mathbb{R}^p$ :

$$
\mathbf{X} \boldsymbol{\theta} = \theta_1 \begin{bmatrix} x_{11} \\ \vdots \\ x_{n1} \end{bmatrix} + \cdots + \theta_p \begin{bmatrix} x_{1p} \\ \vdots \\ x_{np} \end{bmatrix} = \begin{bmatrix} \sum_{j=1}^{p} x_{1j} \theta_j \\ \vdots \\ \sum_{j=1}^{p} x_{nj} \theta_j \end{bmatrix} \in \mathbb{R}^n
$$

**Interprétation ML** : Calcul des prédictions d'un modèle linéaire $\mathbf{y} = \mathbf{X}\boldsymbol{\theta}$

### Multiplication Matrice-Matrice

Soit $\mathbf{X} \in \mathbb{R}^{n \times p}$ et $\mathbf{Z} \in \mathbb{R}^{p \times m}$ :

$$
(\mathbf{X} \times \mathbf{Z})_{ij} = \sum_{k=1}^{p} x_{ik} z_{kj}
$$

Résultat : $\mathbf{X} \times \mathbf{Z} \in \mathbb{R}^{n \times m}$

### Propriétés des Opérations

Pour des matrices de dimensions compatibles :

1. **Commutativité de l'addition** : $\mathbf{X} + \mathbf{Y} = \mathbf{Y} + \mathbf{X}$
2. **Commutativité scalaire** : $\alpha \mathbf{X} = \mathbf{X} \alpha$
3. **Distributivité** : $\mathbf{X}(\mathbf{Y} + \mathbf{Z}) = \mathbf{XY} + \mathbf{XZ}$
4. **Associativité** : $\mathbf{X}(\mathbf{YZ}) = (\mathbf{XY})\mathbf{Z}$
5. **NON-commutativité** : $\mathbf{XY} \neq \mathbf{YX}$ (en général)
6. **Transposée de transposée** : $(\mathbf{X}^t)^t = \mathbf{X}$
7. **Transposée d'une somme** : $(\mathbf{X} + \mathbf{Y})^t = \mathbf{X}^t + \mathbf{Y}^t$
8. **Transposée d'un produit** : $(\mathbf{XY})^t = \mathbf{Y}^t \mathbf{X}^t$

### Exemples Python

```python
import numpy as np

# Définir matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
v = np.array([1, 2])

# Addition
print("A + B =")
print(A + B)

# Multiplication par scalaire
print("\n2*A =")
print(2 * A)

# Produit matrice-vecteur
print("\nA @ v =")
print(A @ v)  # ou A.dot(v)

# Produit matrice-matrice
print("\nA @ B =")
print(A @ B)

# Vérification: AB ≠ BA
print("\nB @ A =")
print(B @ A)
print("\nAB == BA?", np.array_equal(A @ B, B @ A))  # False

# Produit élément par élément (Hadamard)
print("\nA * B (élément par élément) =")
print(A * B)

# Transposée d'un produit
print("\n(AB)^t =")
print((A @ B).T)
print("\nB^t @ A^t =")
print(B.T @ A.T)
print("\nSont-elles égales?", np.array_equal((A @ B).T, B.T @ A.T))  # True
```

---

## Matrices Spéciales

### 1. Matrice Diagonale

Tous les éléments hors diagonale sont nuls :

$$
\mathbf{D} = \begin{bmatrix}
d_{11} & 0 & \cdots & 0 \\
0 & d_{22} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & d_{nn}
\end{bmatrix}
$$

**Déterminant** : $\det(\mathbf{D}) = \prod_{i=1}^{n} d_{ii} = d_{11} \times d_{22} \times \cdots \times d_{nn}$

### 2. Matrice Identité

Matrice diagonale avec des 1 sur la diagonale :

$$
\mathbf{I}_n = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}
$$

**Propriété** : $\mathbf{XI} = \mathbf{IX} = \mathbf{X}$

### 3. Matrice Symétrique

Une matrice est **symétrique** si $\mathbf{X} = \mathbf{X}^t$

**Propriétés importantes** :

- Les valeurs propres sont réelles
- Les vecteurs propres sont orthogonaux
- Diagonalisable dans une base orthonormée

**Usage ML** : Matrices de covariance, matrices Gram

### 4. Matrice Orthogonale

Une matrice $\mathbf{Q} \in \mathbb{R}^{n \times n}$ est **orthogonale** si :

$$
\mathbf{Q}^t \mathbf{Q} = \mathbf{QQ}^t = \mathbf{I}
$$

**Conséquences** :

- $\mathbf{Q}^{-1} = \mathbf{Q}^t$ (inverse = transposée)
- Les lignes sont orthonormales
- Les colonnes sont orthonormales
- Préserve les normes : $\|\mathbf{Qx}\| = \|\mathbf{x}\|$

**Usage ML** : Rotations, PCA, décomposition SVD

### Exemples Python

```python
import numpy as np

# Matrice diagonale
D = np.diag([2, -1, 3])
print("Matrice diagonale D:")
print(D)
print(f"Déterminant: {np.linalg.det(D)}")  # 2*(-1)*3 = -6

# Matrice identité
I = np.eye(4)
print("\nMatrice identité 4x4:")
print(I)

# Matrice symétrique
M = np.random.randn(3, 3)
S = M + M.T  # Assure la symétrie
print("\nMatrice symétrique S:")
print(S)
print("S == S^t?", np.allclose(S, S.T))  # True

# Matrice orthogonale (rotation 90°)
Q = np.array([[0, -1],
              [1, 0]])
print("\nMatrice orthogonale Q:")
print(Q)
print("Q^t @ Q =")
print(Q.T @ Q)  # Doit donner I
print("Est orthogonale?", np.allclose(Q.T @ Q, np.eye(2)))  # True

# Vérifier préservation de la norme
x = np.array([3, 4])
Qx = Q @ x
print(f"\n||x|| = {np.linalg.norm(x):.4f}")
print(f"||Qx|| = {np.linalg.norm(Qx):.4f}")
```

---

## Déterminant et Inversion

### Déterminant

Le **déterminant** $\det(\mathbf{X})$ est un scalaire qui vaut 0 si au moins une colonne est linéairement dépendante des autres.

#### Matrice 2×2

$$
\mathbf{X} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \quad \Rightarrow \quad \det(\mathbf{X}) = ad - bc
$$

#### Matrice 3×3

$$
\mathbf{X} = \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix}
$$

$$
\det(\mathbf{X}) = a \det\begin{bmatrix} e & f \\ h & i \end{bmatrix} - b \det\begin{bmatrix} d & f \\ g & i \end{bmatrix} + c \det\begin{bmatrix} d & e \\ g & h \end{bmatrix}
$$

### Rang d'une Matrice

Le **rang** d'une matrice est le nombre maximal de lignes (ou colonnes) linéairement indépendantes.

**Propriété** : $\mathrm{rang}(\mathbf{X}) = n \Leftrightarrow \det(\mathbf{X}) \neq 0$ (pour matrice carrée $n \times n$)

### Inversion de Matrice

Pour $\mathbf{X} \in \mathbb{R}^{n \times n}$, l'**inverse** $\mathbf{X}^{-1}$ satisfait :

$$
\mathbf{X}^{-1} \mathbf{X} = \mathbf{X} \mathbf{X}^{-1} = \mathbf{I}
$$

**Condition d'existence** : $\det(\mathbf{X}) \neq 0$ (matrice inversible/non-singulière)

**Propriétés** :

- $(\mathbf{X}^{-1})^{-1} = \mathbf{X}$
- $(\mathbf{XY})^{-1} = \mathbf{Y}^{-1} \mathbf{X}^{-1}$
- $(\mathbf{X}^t)^{-1} = (\mathbf{X}^{-1})^t$

### Résolution de Systèmes Linéaires

Pour résoudre $\mathbf{Ax} = \mathbf{b}$ :

$$
\mathbf{x} = \mathbf{A}^{-1} \mathbf{b}
$$

**ATTENTION** : En pratique, on n'inverse jamais explicitement ! On utilise `np.linalg.solve()`.

### Exemples Python

```python
import numpy as np

# Matrice 2x2
A = np.array([[4, 7],
              [2, 6]])

# Déterminant
det_A = np.linalg.det(A)
print(f"Déterminant de A: {det_A}")  # 4*6 - 7*2 = 10

# Rang
rank_A = np.linalg.matrix_rank(A)
print(f"Rang de A: {rank_A}")  # 2

# Inverse
if det_A != 0:
    A_inv = np.linalg.inv(A)
    print("\nInverse de A:")
    print(A_inv)

    # Vérification: A @ A^(-1) = I
    print("\nA @ A^(-1) =")
    print(A @ A_inv)

# Résolution de système linéaire Ax = b
b = np.array([1, 2])

# Méthode 1: Avec inverse (À ÉVITER)
x1 = A_inv @ b
print(f"\nSolution (avec inverse): {x1}")

# Méthode 2: Avec solve (RECOMMANDÉ)
x2 = np.linalg.solve(A, b)
print(f"Solution (avec solve): {x2}")

# Vérification
print(f"Vérification Ax = b: {np.allclose(A @ x2, b)}")

# Exemple de matrice singulière (non inversible)
singular = np.array([[1, 2],
                     [2, 4]])
print(f"\nDéterminant matrice singulière: {np.linalg.det(singular)}")  # ≈ 0
```

---

## Valeurs et Vecteurs Propres

### Définitions

Pour une matrice carrée $\mathbf{A} \in \mathbb{R}^{n \times n}$ :

Un **vecteur propre** $\mathbf{v} \neq \mathbf{0}$ et sa **valeur propre** associée $\lambda$ satisfont :

$$
\mathbf{Av} = \lambda \mathbf{v}
$$

**Interprétation** : La transformation linéaire $\mathbf{A}$ se réduit à une simple multiplication scalaire le long de la direction $\mathbf{v}$.

### Équation Caractéristique

Les valeurs propres sont les racines de :

$$
\det(\mathbf{A} - \lambda \mathbf{I}) = 0
$$

### Propriétés Importantes

1. Une matrice $n \times n$ a $n$ valeurs propres (comptées avec multiplicité)
2. Pour matrice **symétrique** :

   - Toutes les valeurs propres sont **réelles**
   - Les vecteurs propres sont **orthogonaux**
   - La matrice est **diagonalisable**

3. **Trace** : $\mathrm{tr}(\mathbf{A}) = \sum_{i=1}^{n} \lambda_i$

4. **Déterminant** : $\det(\mathbf{A}) = \prod_{i=1}^{n} \lambda_i$

### Diagonalisation

Si $\mathbf{A}$ a $n$ vecteurs propres linéairement indépendants, alors :

$$
\mathbf{A} = \mathbf{P} \boldsymbol{\Lambda} \mathbf{P}^{-1}
$$

où :

- $\mathbf{P}$ : matrice des vecteurs propres (colonnes)
- $\boldsymbol{\Lambda}$ : matrice diagonale des valeurs propres

Pour matrice **symétrique** :

$$
\mathbf{A} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^t
$$

avec $\mathbf{Q}$ orthogonale.

### Rayon Spectral

Le **rayon spectral** est la plus grande valeur propre en valeur absolue :

$$
\rho(\mathbf{A}) = \max_i |\lambda_i|
$$

**Usage ML** : Analyse de convergence des algorithmes itératifs

### Exemples Python

```python
import numpy as np

# Matrice symétrique
A = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]])

print("Matrice A (symétrique):")
print(A)
print("Est symétrique?", np.allclose(A, A.T))

# Calcul valeurs et vecteurs propres
eigenvalues, eigenvectors = np.linalg.eig(A)

print("\nValeurs propres:")
print(eigenvalues)

print("\nVecteurs propres (colonnes):")
print(eigenvectors)

# Vérification: Av = λv
v1 = eigenvectors[:, 0]
lambda1 = eigenvalues[0]
print(f"\nVérification Av = λv pour λ1 = {lambda1:.4f}:")
print(f"Av =", A @ v1)
print(f"λv =", lambda1 * v1)
print(f"Sont égaux? {np.allclose(A @ v1, lambda1 * v1)}")

# Vérification orthogonalité des vecteurs propres
print("\nOrthogonalité des vecteurs propres:")
for i in range(3):
    for j in range(i+1, 3):
        dot = np.dot(eigenvectors[:, i], eigenvectors[:, j])
        print(f"v{i+1} · v{j+1} = {dot:.10f}")

# Diagonalisation
Q = eigenvectors
Lambda = np.diag(eigenvalues)

# Reconstruction: A = Q Λ Q^t
A_reconstructed = Q @ Lambda @ Q.T
print("\nReconstruction A = Q Λ Q^t:")
print(A_reconstructed)
print(f"Reconstruction réussie? {np.allclose(A, A_reconstructed)}")

# Rayon spectral
spectral_radius = np.max(np.abs(eigenvalues))
print(f"\nRayon spectral: {spectral_radius:.4f}")
```

### Algorithme Power Iteration

Méthode itérative pour trouver la valeur propre dominante :

```python
def power_iteration(A, precision=1e-6, max_iter=1000):
    """
    Calcule la plus grande valeur propre et son vecteur propre associé
    """
    n = A.shape[0]
    x = np.random.randn(n)  # Vecteur initial aléatoire

    for i in range(max_iter):
        # Multiplication matrice-vecteur
        Ax = A @ x

        # Normalisation
        x_new = Ax / np.linalg.norm(Ax)

        # Test de convergence
        if np.linalg.norm(x_new - x) < precision:
            break

        x = x_new

    # Calcul valeur propre: λ = v^t A v (pour vecteur normalisé)
    eigenvalue = x.T @ A @ x

    return eigenvalue, x

# Test
A = np.array([[0.5, 0.5],
              [0.2, 0.8]])

lambda_max, v_max = power_iteration(A, precision=1e-10)
print(f"Valeur propre dominante: {lambda_max:.6f}")
print(f"Vecteur propre associé: {v_max}")

# Comparaison avec numpy
eigenvalues_np = np.linalg.eigvals(A)
print(f"\nValeurs propres (numpy): {eigenvalues_np}")
print(f"Maximum: {np.max(eigenvalues_np):.6f}")
```

---

## Décomposition SVD

### Singular Value Decomposition

**Théorème SVD** : Toute matrice $\mathbf{X} \in \mathbb{R}^{n \times p}$ peut être décomposée comme :

$$
\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^t
$$

où :

- $\mathbf{U} \in \mathbb{R}^{n \times n}$ : matrice orthogonale (vecteurs singuliers à gauche)
- $\boldsymbol{\Sigma} \in \mathbb{R}^{n \times p}$ : matrice "diagonale" (valeurs singulières $\sigma_i \geq 0$)
- $\mathbf{V} \in \mathbb{R}^{p \times p}$ : matrice orthogonale (vecteurs singuliers à droite)

### Valeurs Singulières

Les **valeurs singulières** $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ (où $r = \mathrm{rang}(\mathbf{X})$) sont les racines carrées des valeurs propres de $\mathbf{X}^t \mathbf{X}$ :

$$
\sigma_i = \sqrt{\lambda_i(\mathbf{X}^t \mathbf{X})}
$$

### Applications en ML

1. **Réduction de dimensionnalité** (PCA)
2. **Compression d'images**
3. **Systèmes de recommandation** (matrix factorization)
4. **Débruitage de données**
5. **Pseudo-inverse de Moore-Penrose**

### Pseudo-Inverse

Pour résoudre des systèmes sur-déterminés ou sous-déterminés :

$$
\mathbf{X}^+ = \mathbf{V} \boldsymbol{\Sigma}^+ \mathbf{U}^t
$$

où $\boldsymbol{\Sigma}^+$ contient $1/\sigma_i$ pour $\sigma_i > 0$ et 0 sinon.

### Exemples Python

```python
import numpy as np
import matplotlib.pyplot as plt

# Matrice de données
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

print("Matrice X (4x3):")
print(X)

# Décomposition SVD
U, S, Vt = np.linalg.svd(X, full_matrices=True)

print(f"\nDimensions:")
print(f"U: {U.shape}, S: {S.shape}, V^t: {Vt.shape}")

print("\nValeurs singulières:")
print(S)

print("\nMatrice U (orthogonale):")
print(U)

print("\nMatrice V^t (orthogonale):")
print(Vt)

# Vérification orthogonalité
print(f"\nU^t @ U = I? {np.allclose(U.T @ U, np.eye(U.shape[0]))}")
print(f"V @ V^t = I? {np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0]))}")

# Reconstruction
Sigma = np.zeros((U.shape[0], Vt.shape[0]))
np.fill_diagonal(Sigma, S)

X_reconstructed = U @ Sigma @ Vt
print("\nReconstruction X = U Σ V^t:")
print(X_reconstructed)
print(f"Reconstruction réussie? {np.allclose(X, X_reconstructed)}")

# Approximation rang faible (k=2)
k = 2
X_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print(f"\nApproximation rang {k}:")
print(X_approx)

# Erreur de reconstruction
error = np.linalg.norm(X - X_approx, 'fro')
print(f"Erreur (norme de Frobenius): {error:.4f}")
```

### Application : Compression d'Image

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Charger image en niveaux de gris (simulé ici)
np.random.seed(42)
img = np.random.randint(0, 256, (100, 100))  # Image 100x100

# SVD
U, S, Vt = np.linalg.svd(img)

# Compression avec différents rangs
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
ranks = [5, 10, 20, 30, 50, 100]

for idx, k in enumerate(ranks):
    # Reconstruction avec k valeurs singulières
    img_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

    # Taux de compression
    original_size = img.size
    compressed_size = U[:, :k].size + S[:k].size + Vt[:k, :].size
    compression_ratio = compressed_size / original_size * 100

    # Affichage
    ax = axes[idx // 3, idx % 3]
    ax.imshow(img_compressed, cmap='gray')
    ax.set_title(f'Rang {k} ({compression_ratio:.1f}% de l\'original)')
    ax.axis('off')

plt.tight_layout()
plt.savefig('svd_compression.png', dpi=150)
print("Image sauvegardée: svd_compression.png")
```

---

## Projections Orthogonales

### Définition

La **projection orthogonale** d'un vecteur $\mathbf{x} \in \mathbb{R}^n$ sur un sous-espace $E \subset \mathbb{R}^n$ est le vecteur de $E$ le plus proche de $\mathbf{x}$.

### Projection sur une Base Orthonormée

Si $E$ est engendré par une base orthonormée $B = \{\mathbf{e}_1, \ldots, \mathbf{e}_p\}$ :

$$
\mathbf{p}_E(\mathbf{x}) = \sum_{i=1}^{p} \langle \mathbf{x}, \mathbf{e}_i \rangle \mathbf{e}_i
$$

### Matrice de Projection

Pour une matrice $\mathbf{A}$ dont les colonnes forment la base de $E$ :

$$
\mathbf{P} = \mathbf{A}(\mathbf{A}^t \mathbf{A})^{-1} \mathbf{A}^t
$$

**Propriétés** :

1. $\mathbf{P}$ est **symétrique** : $\mathbf{P}^t = \mathbf{P}$
2. $\mathbf{P}$ est **idempotente** : $\mathbf{P}^2 = \mathbf{P}$
3. $\mathbf{P}$ est **carrée** de dimension $n \times n$

### Exemples Python

```python
import numpy as np

# Base orthonormée en R^3
e1 = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
e2 = np.array([1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)])
B = np.array([e1, e2])

# Vérifier orthonormalité
print("Vérification orthonormalité:")
print(f"||e1|| = {np.linalg.norm(e1):.4f}")
print(f"||e2|| = {np.linalg.norm(e2):.4f}")
print(f"e1 · e2 = {np.dot(e1, e2):.10f}")

# Vecteur à projeter
x = np.array([1, 2, 3])

# Méthode 1: Formule avec base orthonormée
proj_x = np.dot(x, e1) * e1 + np.dot(x, e2) * e2
print(f"\nProjection (méthode 1): {proj_x}")

# Méthode 2: Matrice de projection
A = B.T  # Colonnes = vecteurs de base
P = A @ np.linalg.inv(A.T @ A) @ A.T
proj_x2 = P @ x
print(f"Projection (méthode 2): {proj_x2}")

# Vérifications propriétés matrice de projection
print(f"\nP est symétrique? {np.allclose(P, P.T)}")
print(f"P est idempotente? {np.allclose(P @ P, P)}")

# Composante orthogonale
orth_component = x - proj_x
print(f"\nComposante orthogonale: {orth_component}")
print(f"Orthogonale à e1? {np.abs(np.dot(orth_component, e1)) < 1e-10}")
print(f"Orthogonale à e2? {np.abs(np.dot(orth_component, e2)) < 1e-10}")

# Théorème de Pythagore: ||x||² = ||proj||² + ||orth||²
print(f"\nVérification Pythagore:")
print(f"||x||² = {np.linalg.norm(x)**2:.4f}")
print(f"||proj||² + ||orth||² = {np.linalg.norm(proj_x)**2 + np.linalg.norm(orth_component)**2:.4f}")
```

---

## Applications au Machine Learning

### 1. Représentation des Données

**Dataset** : matrice $\mathbf{X} \in \mathbb{R}^{n \times p}$

- $n$ : nombre d'exemples (lignes)
- $p$ : nombre de features (colonnes)

```python
# Exemple: Dataset Iris (simplifié)
X = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Exemple 1
    [4.9, 3.0, 1.4, 0.2],  # Exemple 2
    [6.2, 2.9, 4.3, 1.3],  # Exemple 3
])  # 3 exemples, 4 features
```

### 2. Régression Linéaire

**Modèle** : $\mathbf{y} = \mathbf{X}\boldsymbol{\theta}$

**Solution (moindres carrés)** :

$$
\boldsymbol{\theta}^* = (\mathbf{X}^t \mathbf{X})^{-1} \mathbf{X}^t \mathbf{y}
$$

```python
# Données
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])  # avec intercept
y = np.array([2, 4, 5, 4])

# Solution analytique
theta = np.linalg.inv(X.T @ X) @ X.T @ y
print("Paramètres θ:", theta)

# Prédictions
y_pred = X @ theta
print("Prédictions:", y_pred)
```

### 3. PCA (Principal Component Analysis)

**Objectif** : Réduire la dimensionnalité en trouvant les directions de variance maximale

**Algorithme** :

1. Centrer les données : $\mathbf{X}_c = \mathbf{X} - \bar{\mathbf{X}}$
2. Calculer matrice de covariance : $\mathbf{C} = \frac{1}{n-1} \mathbf{X}_c^t \mathbf{X}_c$
3. Trouver vecteurs propres de $\mathbf{C}$ (composantes principales)
4. Projeter : $\mathbf{Z} = \mathbf{X}_c \mathbf{V}_k$ ($k$ premières composantes)

```python
from sklearn.decomposition import PCA

# Données 3D
X = np.random.randn(100, 3)

# PCA vers 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"Forme originale: {X.shape}")
print(f"Forme réduite: {X_reduced.shape}")
print(f"Variance expliquée: {pca.explained_variance_ratio_}")
```

### 4. Similarité Cosinus

**Mesure de similarité** entre vecteurs :

$$
\text{sim}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \cos(\theta)
$$

**Usage** : Systèmes de recommandation, NLP (word embeddings)

```python
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

doc1 = np.array([1, 2, 3])
doc2 = np.array([2, 3, 4])
doc3 = np.array([-1, -2, -3])

print(f"sim(doc1, doc2) = {cosine_similarity(doc1, doc2):.4f}")
print(f"sim(doc1, doc3) = {cosine_similarity(doc1, doc3):.4f}")
```

### 5. Distance Euclidienne

$$
d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{i=1}^{p} (x_i - y_i)^2}
$$

**Usage** : k-NN, clustering k-means

```python
from scipy.spatial.distance import euclidean

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

dist = euclidean(x, y)
dist_manual = np.linalg.norm(x - y)

print(f"Distance euclidienne: {dist:.4f}")
print(f"Calcul manuel: {dist_manual:.4f}")
```

---

## Exercices Pratiques

### Exercice 1 : Opérations Matricielles

**Énoncé** : Soit

$$
\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad
\mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}, \quad
\mathbf{v} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}
$$

Calculer :

1. $\mathbf{A} + \mathbf{B}$
2. $3\mathbf{A}$
3. $\mathbf{A}\mathbf{v}$
4. $\mathbf{AB}$
5. $\mathbf{BA}$
6. $(\mathbf{AB})^t$

**Solution** :

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
v = np.array([1, 2])

print("1. A + B =")
print(A + B)

print("\n2. 3A =")
print(3 * A)

print("\n3. Av =")
print(A @ v)

print("\n4. AB =")
AB = A @ B
print(AB)

print("\n5. BA =")
BA = B @ A
print(BA)

print("\n6. (AB)^t =")
print(AB.T)

# Vérification: (AB)^t = B^t A^t
print("\nVérification: B^t A^t =")
print(B.T @ A.T)
print("Égal à (AB)^t?", np.array_equal(AB.T, B.T @ A.T))
```

### Exercice 2 : Valeurs Propres

**Énoncé** : Trouver les valeurs et vecteurs propres de

$$
\mathbf{A} = \begin{bmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{bmatrix}
$$

Vérifier que $\mathbf{A}$ est symétrique et que ses vecteurs propres sont orthogonaux.

**Solution** :

```python
import numpy as np

A = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]])

# Vérifier symétrie
print("A est symétrique?", np.allclose(A, A.T))

# Valeurs et vecteurs propres
eigenvalues, eigenvectors = np.linalg.eig(A)

print("\nValeurs propres:")
print(eigenvalues)

print("\nVecteurs propres:")
print(eigenvectors)

# Vérifier orthogonalité
print("\nOrthogonalité:")
for i in range(3):
    for j in range(i+1, 3):
        dot = np.dot(eigenvectors[:, i], eigenvectors[:, j])
        print(f"v{i} · v{j} = {dot:.10f}")

# Reconstruction
Lambda = np.diag(eigenvalues)
Q = eigenvectors
A_reconstructed = Q @ Lambda @ Q.T

print("\nReconstruction réussie?", np.allclose(A, A_reconstructed))
```

### Exercice 3 : Projection Orthogonale

**Énoncé** : Soit $E$ le sous-espace de $\mathbb{R}^3$ engendré par

$$
\mathbf{e}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \quad
\mathbf{e}_2 = \frac{1}{\sqrt{3}}\begin{bmatrix} 1 \\ -1 \\ 1 \end{bmatrix}
$$

Calculer la projection orthogonale de $\mathbf{x} = \begin{bmatrix} 1 \\ 2 \\ -1 \end{bmatrix}$ sur $E$.

**Solution** :

```python
import numpy as np

# Base orthonormée
e1 = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
e2 = np.array([1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)])

# Vérifier orthonormalité
print("||e1|| =", np.linalg.norm(e1))
print("||e2|| =", np.linalg.norm(e2))
print("e1 · e2 =", np.dot(e1, e2))

# Vecteur à projeter
x = np.array([1, 2, -1])

# Projection
proj_x = np.dot(x, e1) * e1 + np.dot(x, e2) * e2

print("\nProjection de x sur E:")
print(proj_x)

# Composante orthogonale
orth = x - proj_x
print("\nComposante orthogonale:")
print(orth)

# Vérification
print("\nVérification orthogonalité:")
print("orth · e1 =", np.dot(orth, e1))
print("orth · e2 =", np.dot(orth, e2))
```

### Exercice 4 : SVD et Compression

**Énoncé** : Créer une matrice aléatoire 10×10 et la compresser en utilisant SVD avec seulement 3 valeurs singulières. Calculer l'erreur de reconstruction.

**Solution** :

```python
import numpy as np

# Matrice aléatoire 10x10
np.random.seed(42)
X = np.random.randn(10, 10)

# SVD complète
U, S, Vt = np.linalg.svd(X)

print("Valeurs singulières:")
print(S)

# Compression rang k=3
k = 3
X_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Erreur
error_frobenius = np.linalg.norm(X - X_compressed, 'fro')
error_relative = error_frobenius / np.linalg.norm(X, 'fro')

print(f"\nErreur (norme de Frobenius): {error_frobenius:.4f}")
print(f"Erreur relative: {error_relative:.4%}")

# Taux de compression
original_elements = X.size
compressed_elements = U[:, :k].size + k + Vt[:k, :].size
compression_ratio = compressed_elements / original_elements

print(f"\nTaux de compression: {compression_ratio:.2%}")
print(f"Réduction: {(1 - compression_ratio):.2%}")
```

---

## Résumé

### Points Clés à Retenir

1. **Vecteurs et Matrices** : Représentation fondamentale des données en ML

2. **Opérations essentielles** :

   - Addition, multiplication scalaire
   - Produit scalaire et matriciel
   - Transposée

3. **Normes** :

   - L1 (Manhattan) : régularisation Lasso
   - L2 (Euclidienne) : distance, régularisation Ridge
   - L∞ (Maximum)

4. **Matrices spéciales** :

   - Identité : $\mathbf{I}$
   - Diagonale : calculs simplifiés
   - Symétrique : covariance, valeurs propres réelles
   - Orthogonale : rotations, PCA

5. **Déterminant et Rang** :

   - $\det(\mathbf{A}) \neq 0 \Leftrightarrow$ matrice inversible
   - Rang = nombre de lignes/colonnes indépendantes

6. **Valeurs et Vecteurs Propres** :

   - $\mathbf{Av} = \lambda \mathbf{v}$
   - Diagonalisation : $\mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$
   - Applications : PCA, analyse de stabilité

7. **SVD** :

   - $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^t$
   - Réduction de dimensionnalité, compression
   - Toujours possible (contrairement à la diagonalisation)

8. **Projections** :
   - $\mathbf{P} = \mathbf{A}(\mathbf{A}^t\mathbf{A})^{-1}\mathbf{A}^t$
   - Propriétés : symétrique, idempotente

### Applications ML Essentielles

| Concept                 | Application ML                               |
| ----------------------- | -------------------------------------------- |
| Produit matrice-vecteur | Prédictions modèle linéaire                  |
| Inversion matricielle   | Régression linéaire (moindres carrés)        |
| Valeurs propres         | PCA, analyse de stabilité                    |
| SVD                     | Réduction de dimensionnalité, recommandation |
| Normes                  | Régularisation (L1, L2), distance            |
| Produit scalaire        | Similarité cosinus, kernel methods           |
| Projection              | Réduction de dimension, features engineering |

### Checklist de Compétences

- [ ] Créer et manipuler vecteurs/matrices avec NumPy
- [ ] Effectuer opérations matricielles (addition, multiplication)
- [ ] Calculer normes vectorielles (L1, L2, L∞)
- [ ] Calculer produit scalaire et angle entre vecteurs
- [ ] Calculer déterminant et rang
- [ ] Inverser une matrice (et savoir quand ne PAS le faire)
- [ ] Résoudre systèmes linéaires avec `np.linalg.solve()`
- [ ] Calculer valeurs et vecteurs propres
- [ ] Effectuer décomposition SVD
- [ ] Calculer projections orthogonales
- [ ] Appliquer ces concepts à des problèmes ML réels

### Fonctions NumPy Essentielles

```python
# Création
np.array(), np.zeros(), np.ones(), np.eye(), np.diag(), np.random.randn()

# Opérations
A @ B, A.T, A.dot(B), A * B (Hadamard)

# Normes
np.linalg.norm(x, ord=1/2/np.inf)

# Algèbre linéaire
np.linalg.det(A)           # Déterminant
np.linalg.matrix_rank(A)   # Rang
np.linalg.inv(A)           # Inverse (à éviter)
np.linalg.solve(A, b)      # Résolution Ax=b
np.linalg.eig(A)           # Valeurs/vecteurs propres
np.linalg.svd(A)           # SVD
np.linalg.pinv(A)          # Pseudo-inverse
```

### Pièges à Éviter

1. ❌ **Ne jamais inverser explicitement** : Utilisez `np.linalg.solve()` au lieu de `np.linalg.inv()`
2. ❌ **Attention à la non-commutativité** : $\mathbf{AB} \neq \mathbf{BA}$
3. ❌ **Broadcast NumPy** : Vérifiez les dimensions avant multiplication
4. ❌ **Stabilité numérique** : Vérifiez le conditionnement des matrices
5. ✅ **Utilisez `np.allclose()`** au lieu de `==` pour comparaisons flottantes

### Ressources Complémentaires

**Livres** :

- "Introduction to Linear Algebra" - Gilbert Strang
- "Linear Algebra and Its Applications" - David Lay
- "Matrix Computations" - Golub & Van Loan

**Cours en ligne** :

- [MIT 18.06 - Linear Algebra (Gilbert Strang)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- [3Blue1Brown - Essence of Linear Algebra (YouTube)](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

**Documentation** :

- [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [SciPy Linear Algebra](https://docs.scipy.org/doc/scipy/reference/linalg.html)

### Prochaine Étape

**Module 3 : Probabilités** - Fondements statistiques pour le ML

---

**Navigation :**

- [⬅️ Module 1 : Introduction](01_Introduction_et_Motivation.md)
- [🏠 Retour au Sommaire](README_ML.md)
- [➡️ Module 3 : Probabilités](03_Probabilites.md)
