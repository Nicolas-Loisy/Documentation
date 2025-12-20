# Module 3 : Théorie des Probabilités pour le Machine Learning

## 📋 Table des Matières

1. [Introduction](#introduction)
2. [Pourquoi les Probabilités en ML ?](#pourquoi-les-probabilités-en-ml-)
3. [Fondements Mathématiques](#fondements-mathématiques)
4. [Probabilités Conditionnelles et Indépendance](#probabilités-conditionnelles-et-indépendance)
5. [Variables Aléatoires](#variables-aléatoires)
6. [Variables Discrètes](#variables-discrètes)
7. [Variables Continues](#variables-continues)
8. [Espérance et Variance](#espérance-et-variance)
9. [Vecteurs Aléatoires](#vecteurs-aléatoires)
10. [Covariance et Corrélation](#covariance-et-corrélation)
11. [Théorème de Bayes](#théorème-de-bayes)
12. [Applications au Machine Learning](#applications-au-machine-learning)
13. [Exercices Pratiques](#exercices-pratiques)
14. [Résumé](#résumé)

---

## Introduction

La **théorie des probabilités** est le langage mathématique de l'incertitude et du hasard. Elle constitue le fondement théorique du Machine Learning, permettant de modéliser l'incertitude inhérente aux données et aux prédictions.

**Domaines d'application** :

- Modélisation de l'incertitude dans les données
- Inférence statistique
- Apprentissage bayésien
- Évaluation de la confiance des prédictions
- Théorie de l'information

---

## Pourquoi les Probabilités en ML ?

### L'Incertitude en Machine Learning

Un concept clé en Machine Learning et Data Science est l'**incertitude** :

1. **Bruit dans les mesures** : Les données réelles sont toujours bruitées
2. **Information incomplète** : On ne dispose jamais de toutes les informations
3. **Variabilité naturelle** : Les phénomènes naturels sont intrinsèquement aléatoires

### Rôle de la Théorie des Probabilités

La théorie des probabilités fournit un cadre cohérent pour :

- **Quantifier l'incertitude** : Mesurer le degré de certitude/incertitude
- **Manipuler l'incertitude** : Combiner et propager les incertitudes
- **Prédictions optimales** : Faire des prédictions même avec information incomplète
- **Prise de décision** : Choisir l'action optimale sous incertitude

### Exemples Concrets

```python
import numpy as np
import matplotlib.pyplot as plt

# Exemple 1: Bruit dans les mesures
np.random.seed(42)
x = np.linspace(0, 10, 100)
y_true = 2 * x + 1
y_observed = y_true + np.random.normal(0, 2, 100)  # Données bruitées

plt.figure(figsize=(10, 5))
plt.scatter(x, y_observed, alpha=0.5, label='Données observées (bruitées)')
plt.plot(x, y_true, 'r-', linewidth=2, label='Vraie relation')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Incertitude due au bruit de mesure')
plt.legend()
plt.grid(True)
plt.show()

# Exemple 2: Classification probabiliste
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                           n_informative=2, random_state=42)
model = LogisticRegression()
model.fit(X, y)

# Probabilités de prédiction (pas juste 0 ou 1)
probabilities = model.predict_proba(X[:5])
print("Probabilités de classification:")
print(probabilities)
```

---

## Fondements Mathématiques

### Espace Probabilisé

Un **espace probabilisé** est un triplet $(\Omega, \mathcal{A}, \mathbb{P})$ où :

1. **$\Omega$** : Espace échantillonal (ensemble de tous les résultats possibles)
2. **$\mathcal{A}$** : σ-algèbre (ensemble des événements mesurables)
3. **$\mathbb{P}$** : Mesure de probabilité

### σ-Algèbre

Une famille $\mathcal{A}$ de sous-ensembles de $\Omega$ est une **σ-algèbre** si elle satisfait :

1. **$\Omega \in \mathcal{A}$** : L'espace total est un événement
2. **Stabilité par complémentaire** : Si $A \in \mathcal{A}$, alors $A^c \in \mathcal{A}$
3. **Stabilité par union** : Si $A, B \in \mathcal{A}$, alors $A \cup B \in \mathcal{A}$

**Les éléments de $\mathcal{A}$ sont appelés événements.**

### Mesure de Probabilité

Une **mesure de probabilité** $\mathbb{P} : \mathcal{A} \to [0,1]$ satisfait les **axiomes de Kolmogorov** :

1. **Non-négativité** : $0 \leq \mathbb{P}(A) \leq 1$ pour tout événement $A$
2. **Normalisation** : $\mathbb{P}(\Omega) = 1$
3. **Additivité** : Pour événements disjoints $A_1, A_2, A_3, \ldots$ :

$$
\mathbb{P}(A_1 \cup A_2 \cup A_3 \cup \cdots) = \mathbb{P}(A_1) + \mathbb{P}(A_2) + \mathbb{P}(A_3) + \cdots
$$

### Propriétés Fondamentales

**Complémentaire** :

$$
\mathbb{P}(A^c) = 1 - \mathbb{P}(A)
$$

**Union (formule d'inclusion-exclusion)** :

$$
\mathbb{P}(A \cup B) = \mathbb{P}(A) + \mathbb{P}(B) - \mathbb{P}(A \cap B)
$$

**Monotonie** :

$$
A \subseteq B \Rightarrow \mathbb{P}(A) \leq \mathbb{P}(B)
$$

### Exemple Python

```python
import numpy as np

# Simulation de lancers de dé
def simuler_de(n_lancers=1000):
    """Simule n lancers d'un dé équilibré"""
    return np.random.randint(1, 7, n_lancers)

# Expérience
lancers = simuler_de(10000)

# Événement A: obtenir un nombre pair
A = (lancers % 2 == 0)
prob_A = np.mean(A)
print(f"P(nombre pair) = {prob_A:.4f} (théorique = 0.5)")

# Événement B: obtenir un nombre ≥ 4
B = (lancers >= 4)
prob_B = np.mean(B)
print(f"P(≥4) = {prob_B:.4f} (théorique = 0.5)")

# Intersection A ∩ B
prob_A_inter_B = np.mean(A & B)
print(f"P(pair ET ≥4) = {prob_A_inter_B:.4f} (théorique = 1/3)")

# Union A ∪ B
prob_A_union_B = np.mean(A | B)
print(f"P(pair OU ≥4) = {prob_A_union_B:.4f}")

# Vérification: P(A∪B) = P(A) + P(B) - P(A∩B)
prob_union_calculee = prob_A + prob_B - prob_A_inter_B
print(f"Vérification formule: {prob_union_calculee:.4f}")
```

---

## Probabilités Conditionnelles et Indépendance

### Probabilité Conditionnelle

La **probabilité conditionnelle** de $A$ sachant $B$ (avec $\mathbb{P}(B) > 0$) :

$$
\mathbb{P}(A|B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)}
$$

**Interprétation** : Probabilité que $A$ se produise sachant que $B$ s'est produit.

**Formule des probabilités composées** :

$$
\mathbb{P}(A \cap B) = \mathbb{P}(A|B) \cdot \mathbb{P}(B) = \mathbb{P}(B|A) \cdot \mathbb{P}(A)
$$

### Indépendance

Deux événements $A$ et $B$ sont **indépendants** si et seulement si :

$$
\mathbb{P}(A \cap B) = \mathbb{P}(A) \cdot \mathbb{P}(B)
$$

**Équivalence** : $A$ et $B$ indépendants $\Leftrightarrow \mathbb{P}(A|B) = \mathbb{P}(A)$

**Interprétation** : La réalisation de $B$ n'apporte aucune information sur $A$.

### Théorème de Bayes (Forme Simple)

$$
\mathbb{P}(A|B) = \frac{\mathbb{P}(B|A) \cdot \mathbb{P}(A)}{\mathbb{P}(B)}
$$

### Formule des Probabilités Totales

Soit $A_1, \ldots, A_n$ une partition de $\Omega$ (événements disjoints dont l'union est $\Omega$), alors :

$$
\mathbb{P}(B) = \sum_{i=1}^{n} \mathbb{P}(B|A_i) \cdot \mathbb{P}(A_i)
$$

### Théorème de Bayes (Forme Générale)

$$
\mathbb{P}(A_i|B) = \frac{\mathbb{P}(B|A_i) \cdot \mathbb{P}(A_i)}{\sum_{j=1}^{n} \mathbb{P}(B|A_j) \cdot \mathbb{P}(A_j)}
$$

### Exemple Pratique : Test Médical

```python
import numpy as np

# Paramètres
P_maladie = 0.01          # P(Malade) = 1%
P_pos_sachant_malade = 0.95   # Sensibilité (True Positive Rate)
P_neg_sachant_sain = 0.90     # Spécificité (True Negative Rate)

# P(Test+|Sain) = 1 - Spécificité (Faux positif)
P_pos_sachant_sain = 1 - P_neg_sachant_sain

# P(Sain)
P_sain = 1 - P_maladie

# Probabilité totale: P(Test+)
P_test_positif = (P_pos_sachant_malade * P_maladie +
                  P_pos_sachant_sain * P_sain)

# Théorème de Bayes: P(Malade|Test+)
P_malade_sachant_pos = (P_pos_sachant_malade * P_maladie) / P_test_positif

print(f"Prévalence de la maladie: {P_maladie:.1%}")
print(f"Sensibilité du test: {P_pos_sachant_malade:.1%}")
print(f"Spécificité du test: {P_neg_sachant_sain:.1%}")
print(f"\nP(Test positif) = {P_test_positif:.4f}")
print(f"P(Malade | Test+) = {P_malade_sachant_pos:.4f} = {P_malade_sachant_pos:.1%}")

# Simulation
n_population = 100000
malade = np.random.random(n_population) < P_maladie

# Test
test_result = np.zeros(n_population, dtype=bool)
test_result[malade] = np.random.random(malade.sum()) < P_pos_sachant_malade
test_result[~malade] = np.random.random((~malade).sum()) < P_pos_sachant_sain

# Vérification empirique
positifs = test_result
malades_parmi_positifs = malade[positifs]
prob_empirique = malades_parmi_positifs.mean()

print(f"\nVérification par simulation: {prob_empirique:.4f} = {prob_empirique:.1%}")
```

---

## Variables Aléatoires

### Définition

Une **variable aléatoire** $X$ est une fonction qui associe à chaque résultat de l'espace $\Omega$ une valeur dans un ensemble $E$ :

$$
X : \Omega \to E
$$

**Notations** :

- $\mathbb{P}(X = x)$ désigne $\mathbb{P}(\{\omega \in \Omega : X(\omega) = x\})$
- $\mathbb{P}(X \in I)$ désigne $\mathbb{P}(\{\omega \in \Omega : X(\omega) \in I\})$

### Fonction de Répartition (CDF)

La **fonction de répartition** (Cumulative Distribution Function) de $X$ :

$$
F_X(t) = \mathbb{P}(X \leq t)
$$

**Propriétés** :

- $F_X$ est croissante
- $\lim_{t \to -\infty} F_X(t) = 0$ et $\lim_{t \to +\infty} F_X(t) = 1$
- $F_X$ est continue à droite

### Distribution de Probabilité

La **distribution de probabilité** de $X$ est l'ensemble :

$$
\{\mathbb{P}(X = x), \, x \in E\}
$$

Elle décrit comment la probabilité est répartie sur les valeurs possibles de $X$.

---

## Variables Discrètes

Une variable aléatoire est **discrète** si elle prend un nombre fini ou dénombrable de valeurs.

### Règle de Somme

Pour toute variable discrète $X$ :

$$
\sum_{x \in E} \mathbb{P}(X = x) = 1
$$

### Distribution Marginale

Soit $X$ et $Y$ deux variables discrètes :

$$
\mathbb{P}(X = x) = \sum_{y} \mathbb{P}(X = x, Y = y)
$$

### Distribution Conditionnelle

$$
\mathbb{P}(X = x | Y = y) = \frac{\mathbb{P}(X = x, Y = y)}{\mathbb{P}(Y = y)}
$$

### 1. Loi de Bernoulli $\mathcal{B}(p)$

**Définition** : Modélise une expérience à deux issues (succès/échec).

$$
X \sim \mathcal{B}(p) \Leftrightarrow
\begin{cases}
\mathbb{P}(X = 1) = p \\
\mathbb{P}(X = 0) = 1 - p
\end{cases}
$$

**Paramètres** : $p \in [0, 1]$ (probabilité de succès)

**Propriétés** :

- **Espérance** : $\mathbb{E}[X] = p$
- **Variance** : $\mathbb{V}[X] = p(1-p)$

**Exemple** : Lancer de pièce, réussite/échec d'un test

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Loi de Bernoulli avec p=0.3
p = 0.3
X_bern = stats.bernoulli(p)

# Simulation
n_simulations = 10000
echantillon = X_bern.rvs(n_simulations)

print(f"Bernoulli(p={p})")
print(f"Espérance théorique: {p}")
print(f"Espérance empirique: {echantillon.mean():.4f}")
print(f"Variance théorique: {p*(1-p):.4f}")
print(f"Variance empirique: {echantillon.var():.4f}")

# Visualisation
valeurs = [0, 1]
probas = [1-p, p]

plt.figure(figsize=(8, 5))
plt.bar(valeurs, probas, width=0.3, alpha=0.7, edgecolor='black')
plt.xlabel('Valeur')
plt.ylabel('Probabilité')
plt.title(f'Loi de Bernoulli B({p})')
plt.xticks([0, 1])
plt.grid(axis='y', alpha=0.3)
plt.show()
```

### 2. Loi Binomiale $\mathcal{B}(n, p)$

**Définition** : Nombre de succès dans $n$ répétitions indépendantes d'une expérience de Bernoulli.

$$
X \sim \mathcal{B}(n, p) \Rightarrow \mathbb{P}(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

où $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ est le coefficient binomial.

**Paramètres** :

- $n \in \mathbb{N}$ : nombre d'essais
- $p \in [0, 1]$ : probabilité de succès

**Propriétés** :

- **Espérance** : $\mathbb{E}[X] = np$
- **Variance** : $\mathbb{V}[X] = np(1-p)$

**Exemple** : Nombre de faces dans 10 lancers de pièce

```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# Loi binomiale B(n=10, p=0.5)
n, p = 10, 0.5
X_binom = stats.binom(n, p)

# PMF (Probability Mass Function)
k_values = np.arange(0, n+1)
probas = X_binom.pmf(k_values)

# Visualisation
plt.figure(figsize=(10, 6))
plt.bar(k_values, probas, alpha=0.7, edgecolor='black')
plt.xlabel('Nombre de succès (k)')
plt.ylabel('Probabilité P(X=k)')
plt.title(f'Loi Binomiale B(n={n}, p={p})')
plt.xticks(k_values)
plt.grid(axis='y', alpha=0.3)
plt.axvline(n*p, color='r', linestyle='--', label=f'Espérance = {n*p}')
plt.legend()
plt.show()

# Simulation
echantillon = X_binom.rvs(10000)
print(f"Espérance théorique: {n*p}")
print(f"Espérance empirique: {echantillon.mean():.4f}")
print(f"Variance théorique: {n*p*(1-p):.4f}")
print(f"Variance empirique: {echantillon.var():.4f}")
```

### 3. Loi Géométrique $\mathcal{G}(p)$

**Définition** : Rang du premier succès dans une suite d'essais de Bernoulli indépendants.

$$
X \sim \mathcal{G}(p) \Rightarrow \mathbb{P}(X = k) = p(1-p)^{k-1}, \quad k \in \mathbb{N}^*
$$

**Paramètres** : $p \in ]0, 1]$ (probabilité de succès)

**Propriétés** :

- **Espérance** : $\mathbb{E}[X] = \frac{1}{p}$
- **Variance** : $\mathbb{V}[X] = \frac{1-p}{p^2}$

**Propriété sans mémoire** : $\mathbb{P}(X > n + m | X > n) = \mathbb{P}(X > m)$

**Exemple** : Nombre de lancers avant le premier 6

```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# Loi géométrique G(p=1/6)
p = 1/6
X_geom = stats.geom(p)

# PMF
k_values = np.arange(1, 21)
probas = X_geom.pmf(k_values)

# Visualisation
plt.figure(figsize=(10, 6))
plt.bar(k_values, probas, alpha=0.7, edgecolor='black')
plt.xlabel('Nombre d\'essais jusqu\'au premier succès')
plt.ylabel('Probabilité')
plt.title(f'Loi Géométrique G(p={p:.4f})')
plt.grid(axis='y', alpha=0.3)
plt.axvline(1/p, color='r', linestyle='--', label=f'Espérance = {1/p:.2f}')
plt.legend()
plt.show()

# Simulation
echantillon = X_geom.rvs(10000)
print(f"Espérance théorique: {1/p:.4f}")
print(f"Espérance empirique: {echantillon.mean():.4f}")
```

### 4. Loi de Poisson $\mathcal{P}(\lambda)$

**Définition** : Nombre d'événements se produisant dans un intervalle de temps fixé, sachant qu'ils se produisent en moyenne $\lambda$ fois.

$$
X \sim \mathcal{P}(\lambda) \Rightarrow \mathbb{P}(X = k) = \frac{\lambda^k}{k!} e^{-\lambda}, \quad k \in \mathbb{N}
$$

**Paramètres** : $\lambda > 0$ (taux moyen d'occurrence)

**Propriétés** :

- **Espérance** : $\mathbb{E}[X] = \lambda$
- **Variance** : $\mathbb{V}[X] = \lambda$

**Exemple** : Nombre d'appels téléphoniques par heure, nombre d'erreurs par page

```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# Loi de Poisson P(λ=3)
lambda_param = 3
X_poisson = stats.poisson(lambda_param)

# PMF
k_values = np.arange(0, 15)
probas = X_poisson.pmf(k_values)

# Visualisation
plt.figure(figsize=(10, 6))
plt.bar(k_values, probas, alpha=0.7, edgecolor='black')
plt.xlabel('Nombre d\'événements (k)')
plt.ylabel('Probabilité P(X=k)')
plt.title(f'Loi de Poisson P(λ={lambda_param})')
plt.grid(axis='y', alpha=0.3)
plt.axvline(lambda_param, color='r', linestyle='--',
            label=f'Espérance = Variance = {lambda_param}')
plt.legend()
plt.show()

# Comparaison de différentes valeurs de λ
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
lambdas = [1, 4, 10]

for ax, lam in zip(axes, lambdas):
    X = stats.poisson(lam)
    k_vals = np.arange(0, lam*3)
    ax.bar(k_vals, X.pmf(k_vals), alpha=0.7, edgecolor='black')
    ax.set_title(f'Poisson(λ={lam})')
    ax.set_xlabel('k')
    ax.set_ylabel('P(X=k)')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Variables Continues

Une variable aléatoire est **continue** si elle prend ses valeurs dans un ensemble continu (typiquement $\mathbb{R}$ ou un intervalle).

### Fonction de Densité de Probabilité (PDF)

Une fonction $f : \mathbb{R} \to \mathbb{R}_+$ est une **densité de probabilité** si :

$$
\int_{\mathbb{R}} f(x) \, dx = 1
$$

Pour une variable continue $X$ de densité $f$ :

$$
\mathbb{P}(a \leq X \leq b) = \int_a^b f(x) \, dx
$$

**Remarque importante** : $\mathbb{P}(X = x) = 0$ pour tout $x$ (probabilité nulle en un point).

### Fonction de Répartition

$$
F_X(t) = \mathbb{P}(X \leq t) = \int_{-\infty}^{t} f(x) \, dx
$$

**Relation** : $f(x) = F_X'(x)$ (la densité est la dérivée de la fonction de répartition)

### 1. Loi Uniforme $\mathcal{U}([a, b])$

**Définition** : Tous les points de l'intervalle $[a, b]$ ont la même "densité de probabilité".

**Densité** :

$$
f(x) = \begin{cases}
\frac{1}{b-a} & \text{si } x \in [a, b] \\
0 & \text{sinon}
\end{cases}
$$

**Fonction de répartition** :

$$
F_X(t) = \begin{cases}
0 & \text{si } t < a \\
\frac{t-a}{b-a} & \text{si } t \in [a, b] \\
1 & \text{si } t > b
\end{cases}
$$

**Propriétés** :

- **Espérance** : $\mathbb{E}[X] = \frac{a+b}{2}$
- **Variance** : $\mathbb{V}[X] = \frac{(b-a)^2}{12}$

**Exemple** : Nombre aléatoire entre 0 et 1

```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# Loi uniforme U([0, 5])
a, b = 0, 5
X_unif = stats.uniform(a, b-a)

# Visualisation
x = np.linspace(-1, 6, 1000)
pdf = X_unif.pdf(x)
cdf = X_unif.cdf(x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# PDF
ax1.plot(x, pdf, 'b-', linewidth=2)
ax1.fill_between(x, 0, pdf, alpha=0.3)
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_title(f'Densité de probabilité - Uniforme([{a}, {b}])')
ax1.grid(True, alpha=0.3)
ax1.axvline((a+b)/2, color='r', linestyle='--', label='Espérance')
ax1.legend()

# CDF
ax2.plot(x, cdf, 'g-', linewidth=2)
ax2.set_xlabel('x')
ax2.set_ylabel('F(x)')
ax2.set_title(f'Fonction de répartition - Uniforme([{a}, {b}])')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Génération d'échantillons
echantillon = X_unif.rvs(10000)
print(f"Espérance théorique: {(a+b)/2}")
print(f"Espérance empirique: {echantillon.mean():.4f}")
print(f"Variance théorique: {(b-a)**2/12:.4f}")
print(f"Variance empirique: {echantillon.var():.4f}")
```

### 2. Loi Exponentielle $\mathcal{E}(\lambda)$

**Définition** : Temps d'attente avant le premier événement dans un processus de Poisson.

**Densité** :

$$
f(x) = \lambda e^{-\lambda x}, \quad x \geq 0
$$

**Fonction de répartition** :

$$
F_X(t) = 1 - e^{-\lambda t}, \quad t \geq 0
$$

**Propriétés** :

- **Espérance** : $\mathbb{E}[X] = \frac{1}{\lambda}$
- **Variance** : $\mathbb{V}[X] = \frac{1}{\lambda^2}$

**Propriété sans mémoire** : $\mathbb{P}(X > s + t | X > s) = \mathbb{P}(X > t)$

**Exemple** : Durée de vie d'un composant électronique

```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# Loi exponentielle E(λ=0.5)
lambda_param = 0.5
X_exp = stats.expon(scale=1/lambda_param)

# Visualisation pour différentes valeurs de λ
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

lambdas = [0.5, 1.0, 1.5]
x = np.linspace(0, 5, 1000)
colors = ['b', 'g', 'r']

for lam, color in zip(lambdas, colors):
    X = stats.expon(scale=1/lam)
    ax1.plot(x, X.pdf(x), color=color, linewidth=2, label=f'λ={lam}')
    ax2.plot(x, X.cdf(x), color=color, linewidth=2, label=f'λ={lam}')

ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_title('Densité - Loi Exponentielle')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('x')
ax2.set_ylabel('F(x)')
ax2.set_title('Fonction de répartition - Loi Exponentielle')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Propriété sans mémoire
lambda_param = 1.0
X = stats.expon(scale=1/lambda_param)
s, t = 2.0, 1.0

# P(X > s+t | X > s) = P(X > t)
prob_sans_memoire = 1 - X.cdf(t)
prob_conditionnelle = (1 - X.cdf(s+t)) / (1 - X.cdf(s))

print(f"Propriété sans mémoire:")
print(f"P(X > {t}) = {prob_sans_memoire:.4f}")
print(f"P(X > {s+t} | X > {s}) = {prob_conditionnelle:.4f}")
```

### 3. Loi Normale (Gaussienne) $\mathcal{N}(\mu, \sigma^2)$

**Définition** : Distribution la plus importante en statistiques (Théorème Central Limite).

**Densité** :

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

**Propriétés** :

- **Espérance** : $\mathbb{E}[X] = \mu$
- **Variance** : $\mathbb{V}[X] = \sigma^2$
- **Symétrie** : Symétrique autour de $\mu$
- **Forme en cloche**

**Loi normale standard** $\mathcal{N}(0, 1)$ :

Si $X \sim \mathcal{N}(\mu, \sigma^2)$, alors $Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$

**Règle empirique (68-95-99.7)** :

- 68% des valeurs dans $[\mu - \sigma, \mu + \sigma]$
- 95% des valeurs dans $[\mu - 2\sigma, \mu + 2\sigma]$
- 99.7% des valeurs dans $[\mu - 3\sigma, \mu + 3\sigma]$

```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# Différentes lois normales
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Effet de μ (moyenne)
x = np.linspace(-10, 10, 1000)
means = [0, 2, -2]
ax = axes[0, 0]
for mu in means:
    X = stats.norm(mu, 1)
    ax.plot(x, X.pdf(x), linewidth=2, label=f'μ={mu}, σ²=1')
ax.set_title('Effet de μ (moyenne)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Effet de σ² (variance)
x = np.linspace(-10, 10, 1000)
stds = [0.5, 1, 2]
ax = axes[0, 1]
for sigma in stds:
    X = stats.norm(0, sigma)
    ax.plot(x, X.pdf(x), linewidth=2, label=f'μ=0, σ²={sigma**2}')
ax.set_title('Effet de σ² (variance)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Fonction de répartition
x = np.linspace(-4, 4, 1000)
X_std = stats.norm(0, 1)
ax = axes[1, 0]
ax.plot(x, X_std.cdf(x), 'b-', linewidth=2)
ax.set_title('Fonction de répartition - N(0,1)')
ax.set_xlabel('x')
ax.set_ylabel('F(x)')
ax.grid(True, alpha=0.3)
ax.axhline(0.5, color='r', linestyle='--', alpha=0.5)
ax.axvline(0, color='r', linestyle='--', alpha=0.5)

# 4. Règle empirique 68-95-99.7
x = np.linspace(-4, 4, 1000)
ax = axes[1, 1]
ax.plot(x, X_std.pdf(x), 'b-', linewidth=2)
ax.fill_between(x, 0, X_std.pdf(x), where=(np.abs(x) <= 1),
                 alpha=0.3, color='green', label='68% (±σ)')
ax.fill_between(x, 0, X_std.pdf(x), where=(np.abs(x) <= 2),
                 alpha=0.2, color='blue', label='95% (±2σ)')
ax.fill_between(x, 0, X_std.pdf(x), where=(np.abs(x) <= 3),
                 alpha=0.1, color='red', label='99.7% (±3σ)')
ax.set_title('Règle empirique 68-95-99.7')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Vérification numérique
mu, sigma = 0, 1
X = stats.norm(mu, sigma)

print("Règle empirique:")
print(f"P(μ-σ ≤ X ≤ μ+σ) = {X.cdf(1) - X.cdf(-1):.4f} ≈ 0.68")
print(f"P(μ-2σ ≤ X ≤ μ+2σ) = {X.cdf(2) - X.cdf(-2):.4f} ≈ 0.95")
print(f"P(μ-3σ ≤ X ≤ μ+3σ) = {X.cdf(3) - X.cdf(-3):.4f} ≈ 0.997")
```

### Théorème Central Limite

**Énoncé simplifié** : La somme (ou moyenne) d'un grand nombre de variables aléatoires indépendantes tend vers une loi normale.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Démonstration du TCL avec des uniformes
n_samples = 10000
sample_sizes = [1, 2, 5, 30]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, n in enumerate(sample_sizes):
    # Générer n variables uniformes et calculer leur moyenne
    moyennes = np.mean(np.random.uniform(0, 1, (n_samples, n)), axis=1)

    ax = axes[idx]
    ax.hist(moyennes, bins=50, density=True, alpha=0.7, edgecolor='black')

    # Superposer la loi normale théorique
    mu = 0.5  # E[Unif(0,1)] = 0.5
    sigma = np.sqrt(1/12 / n)  # Var[Unif(0,1)] / n
    x = np.linspace(moyennes.min(), moyennes.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
            label='Normale théorique')

    ax.set_title(f'Moyenne de {n} variables uniformes')
    ax.set_xlabel('Valeur de la moyenne')
    ax.set_ylabel('Densité')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Espérance et Variance

### Espérance (Valeur Moyenne)

**Variable discrète** :

$$
\mathbb{E}[X] = \sum_{x \in E} x \cdot \mathbb{P}(X = x)
$$

**Variable continue** :

$$
\mathbb{E}[X] = \int_{-\infty}^{+\infty} x \cdot f(x) \, dx
$$

**Interprétation** : Valeur moyenne que prendrait $X$ sur un grand nombre de répétitions.

### Propriétés de l'Espérance

1. **Linéarité** : $\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$
2. **Additivité** : $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$
3. **Constante** : $\mathbb{E}[c] = c$
4. **Positivité** : Si $X \geq 0$ alors $\mathbb{E}[X] \geq 0$
5. **Monotonie** : Si $X \geq Y$ alors $\mathbb{E}[X] \geq \mathbb{E}[Y]$

**Si $X$ et $Y$ sont indépendantes** :

$$
\mathbb{E}[XY] = \mathbb{E}[X] \cdot \mathbb{E}[Y]
$$

### Variance (Dispersion)

$$
\mathbb{V}[X] = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

**Interprétation** : Mesure de la dispersion autour de la moyenne.

### Propriétés de la Variance

1. **Translation** : $\mathbb{V}[X + b] = \mathbb{V}[X]$
2. **Homothétie** : $\mathbb{V}[aX] = a^2 \mathbb{V}[X]$
3. **Positivité** : $\mathbb{V}[X] \geq 0$
4. **Nullité** : $\mathbb{V}[X] = 0 \Leftrightarrow X$ est constante

**Si $X$ et $Y$ sont indépendantes** :

$$
\mathbb{V}[X + Y] = \mathbb{V}[X] + \mathbb{V}[Y]
$$

### Écart-Type

$$
\sigma(X) = \sqrt{\mathbb{V}[X]}
$$

**Avantage** : Même unité que $X$ (contrairement à la variance).

### Exemple Python

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Comparer différentes distributions avec même espérance
mu = 5

# Distribution 1: Normale avec faible variance
X1 = stats.norm(mu, 0.5)

# Distribution 2: Normale avec forte variance
X2 = stats.norm(mu, 2)

# Distribution 3: Uniforme centrée sur mu
a, b = mu - 3, mu + 3
X3 = stats.uniform(a, b-a)

# Visualisation
x = np.linspace(0, 10, 1000)

plt.figure(figsize=(12, 6))
plt.plot(x, X1.pdf(x), label=f'N({mu}, 0.25) - σ=0.5', linewidth=2)
plt.plot(x, X2.pdf(x), label=f'N({mu}, 4) - σ=2', linewidth=2)
plt.plot(x, X3.pdf(x), label=f'U([{a},{b}]) - σ={np.sqrt((b-a)**2/12):.2f}', linewidth=2)

plt.axvline(mu, color='r', linestyle='--', linewidth=2, label='Espérance commune')
plt.xlabel('x')
plt.ylabel('Densité')
plt.title('Distributions avec même espérance mais variances différentes')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Simulation
n_samples = 100000
samples1 = X1.rvs(n_samples)
samples2 = X2.rvs(n_samples)
samples3 = X3.rvs(n_samples)

print("Espérance empirique:")
print(f"X1: {samples1.mean():.4f}")
print(f"X2: {samples2.mean():.4f}")
print(f"X3: {samples3.mean():.4f}")

print("\nVariance empirique:")
print(f"X1: {samples1.var():.4f} (théorique: 0.25)")
print(f"X2: {samples2.var():.4f} (théorique: 4)")
print(f"X3: {samples3.var():.4f} (théorique: {(b-a)**2/12:.4f})")
```

---

## Vecteurs Aléatoires

### Définition

Un **vecteur aléatoire** $\mathbf{X} = (X_1, \ldots, X_n)$ est un vecteur dont les composantes sont des variables aléatoires.

### Fonction de Répartition Jointe

$$
F_{\mathbf{X}}(t_1, \ldots, t_n) = \mathbb{P}(X_1 \leq t_1, \ldots, X_n \leq t_n)
$$

### Densité Jointe

Pour un vecteur aléatoire continu :

$$
\mathbb{P}(\mathbf{X} \in A) = \int_A f(x_1, \ldots, x_n) \, dx_1 \cdots dx_n
$$

### Espérance d'un Vecteur

$$
\mathbb{E}[\mathbf{X}] = (\mathbb{E}[X_1], \ldots, \mathbb{E}[X_n]) \in \mathbb{R}^n
$$

### Indépendance

Les variables $X_1, \ldots, X_n$ sont **indépendantes** si et seulement si :

$$
f_{\mathbf{X}}(x_1, \ldots, x_n) = \prod_{i=1}^{n} f_{X_i}(x_i)
$$

**Conséquence** : Si $X_1, \ldots, X_n$ indépendantes alors :

$$
\forall i \neq j, \quad \text{Cov}(X_i, X_j) = 0
$$

### Loi Normale Multivariée

$$
\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})
$$

**Densité** :

$$
f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d \det(\boldsymbol{\Sigma})}} e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})}
$$

où :

- $\boldsymbol{\mu} \in \mathbb{R}^d$ : vecteur des moyennes
- $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$ : matrice de covariance (symétrique définie positive)

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Loi normale bivariée
mu = np.array([0, 0])
Sigma = np.array([[1, 0.8],
                   [0.8, 1]])

# Création de la distribution
X = stats.multivariate_normal(mu, Sigma)

# Grille pour visualisation
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
pos = np.dstack((X1, X2))

# Densité
Z = X.pdf(pos)

# Visualisation 3D
fig = plt.figure(figsize=(14, 6))

# Surface 3D
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('X₁')
ax1.set_ylabel('X₂')
ax1.set_zlabel('Densité')
ax1.set_title('Densité normale bivariée')

# Contours
ax2 = fig.add_subplot(122)
contour = ax2.contour(X1, X2, Z, levels=10, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('X₁')
ax2.set_ylabel('X₂')
ax2.set_title('Lignes de niveau')
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

plt.tight_layout()
plt.show()

# Échantillonnage
samples = X.rvs(1000)
plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.xlabel('X₁')
plt.ylabel('X₂')
plt.title('1000 échantillons de la loi normale bivariée')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

---

## Covariance et Corrélation

### Covariance

La **covariance** entre deux variables $X$ et $Y$ :

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
$$

**Interprétation** :

- $\text{Cov}(X, Y) > 0$ : $X$ et $Y$ varient dans le même sens
- $\text{Cov}(X, Y) < 0$ : $X$ et $Y$ varient en sens opposé
- $\text{Cov}(X, Y) = 0$ : $X$ et $Y$ ne sont pas linéairement liées

### Propriétés de la Covariance

1. **Symétrie** : $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
2. **Avec soi-même** : $\text{Cov}(X, X) = \mathbb{V}[X]$
3. **Bilinéarité** : $\text{Cov}(aX + b, cY + d) = ac \cdot \text{Cov}(X, Y)$
4. **Variance d'une somme** : $\mathbb{V}[X + Y] = \mathbb{V}[X] + \mathbb{V}[Y] + 2\text{Cov}(X, Y)$

### Coefficient de Corrélation

$$
\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sqrt{\mathbb{V}[X]} \sqrt{\mathbb{V}[Y]}} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

**Propriétés** :

- $\rho(X, Y) \in [-1, 1]$
- $|\rho(X, Y)| = 1$ : relation linéaire parfaite
- $\rho(X, Y) = 0$ : variables non corrélées (mais pas nécessairement indépendantes !)

**Interprétation** :

- $\rho = 1$ : corrélation positive parfaite ($Y = aX + b$ avec $a > 0$)
- $\rho = -1$ : corrélation négative parfaite ($Y = aX + b$ avec $a < 0$)
- $\rho = 0$ : pas de corrélation linéaire

### Matrice de Covariance

Pour un vecteur aléatoire $\mathbf{X} = (X_1, \ldots, X_n)$ :

$$
\boldsymbol{\Sigma} = \text{Cov}(\mathbf{X}) = (\text{Cov}(X_i, X_j))_{1 \leq i,j \leq n}
$$

**Propriétés** :

- $\boldsymbol{\Sigma}$ est **symétrique** : $\boldsymbol{\Sigma}^T = \boldsymbol{\Sigma}$
- $\boldsymbol{\Sigma}$ est **semi-définie positive**

### Exemples Python

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Générer des données avec différentes corrélations
n = 500
correlations = [0.9, 0.5, 0, -0.5, -0.9]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax, rho in zip(axes, correlations):
    # Matrice de covariance
    Sigma = np.array([[1, rho],
                       [rho, 1]])

    # Générer échantillon
    samples = np.random.multivariate_normal([0, 0], Sigma, n)

    # Calcul corrélation empirique
    rho_empirique = np.corrcoef(samples.T)[0, 1]

    # Visualisation
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    ax.set_title(f'ρ = {rho} (empirique: {rho_empirique:.2f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

plt.tight_layout()
plt.show()

# Matrice de covariance pour 3 variables
n = 1000
mu = np.array([0, 0, 0])
Sigma = np.array([[1.0, 0.8, 0.3],
                   [0.8, 1.0, 0.5],
                   [0.3, 0.5, 1.0]])

samples = np.random.multivariate_normal(mu, Sigma, n)

# Matrice de covariance empirique
cov_empirique = np.cov(samples.T)

# Matrice de corrélation empirique
corr_empirique = np.corrcoef(samples.T)

print("Matrice de covariance théorique:")
print(Sigma)
print("\nMatrice de covariance empirique:")
print(cov_empirique)
print("\nMatrice de corrélation empirique:")
print(corr_empirique)

# Visualisation avec heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.imshow(cov_empirique, cmap='coolwarm', vmin=-1, vmax=1)
ax1.set_title('Matrice de covariance')
for i in range(3):
    for j in range(3):
        text = ax1.text(j, i, f'{cov_empirique[i, j]:.2f}',
                       ha="center", va="center", color="black")
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(corr_empirique, cmap='coolwarm', vmin=-1, vmax=1)
ax2.set_title('Matrice de corrélation')
for i in range(3):
    for j in range(3):
        text = ax2.text(j, i, f'{corr_empirique[i, j]:.2f}',
                       ha="center", va="center", color="black")
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()
```

**Attention** : Corrélation nulle n'implique PAS indépendance !

```python
# Exemple: variables non corrélées mais dépendantes
n = 1000
X = np.random.uniform(-2, 2, n)
Y = X**2 + np.random.normal(0, 0.1, n)

# Corrélation
rho = np.corrcoef(X, Y)[0, 1]

plt.figure(figsize=(8, 6))
plt.scatter(X, Y, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y = X²')
plt.title(f'Variables dépendantes mais non corrélées (ρ = {rho:.3f})')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Corrélation: {rho:.4f} (proche de 0)")
print("Pourtant Y dépend clairement de X (relation quadratique) !")
```

---

## Théorème de Bayes

### Formulation Générale

Soit $A_1, \ldots, A_n$ une partition de $\Omega$ (événements disjoints avec $\bigcup_{i=1}^n A_i = \Omega$).

**Théorème de Bayes** :

$$
\mathbb{P}(A_i | B) = \frac{\mathbb{P}(B|A_i) \cdot \mathbb{P}(A_i)}{\sum_{j=1}^{n} \mathbb{P}(B|A_j) \cdot \mathbb{P}(A_j)}
$$

**Terminologie** :

- $\mathbb{P}(A_i)$ : **Probabilité a priori** (avant observation de $B$)
- $\mathbb{P}(A_i|B)$ : **Probabilité a posteriori** (après observation de $B$)
- $\mathbb{P}(B|A_i)$ : **Vraisemblance** (likelihood)
- $\mathbb{P}(B)$ : **Évidence** (probabilité marginale)

### Formulation pour Variables Continues

Soit $\Theta$ un paramètre et $X$ une observation :

$$
f(\theta | x) = \frac{f(x|\theta) \cdot f(\theta)}{\int f(x|\theta') \cdot f(\theta') \, d\theta'}
$$

où :

- $f(\theta)$ : **distribution a priori**
- $f(x|\theta)$ : **vraisemblance**
- $f(\theta|x)$ : **distribution a posteriori**

### Application : Classification Bayésienne Naïve

**Hypothèse naïve** : Les features sont conditionnellement indépendantes sachant la classe.

$$
\mathbb{P}(C_k | x_1, \ldots, x_n) \propto \mathbb{P}(C_k) \prod_{i=1}^{n} \mathbb{P}(x_i | C_k)
$$

### Exemple Complet : Diagnostic Médical

```python
import numpy as np
import matplotlib.pyplot as plt

def diagnostic_bayesien(prevalence, sensibilite, specificite):
    """
    Calcul de P(Malade|Test+) en utilisant le théorème de Bayes

    Parameters:
    - prevalence: P(Malade)
    - sensibilite: P(Test+|Malade) (True Positive Rate)
    - specificite: P(Test-|Sain) (True Negative Rate)
    """
    # P(Test+|Sain) = 1 - Spécificité (Faux positif)
    P_pos_sain = 1 - specificite

    # P(Sain)
    P_sain = 1 - prevalence

    # P(Test+) = P(Test+|Malade)P(Malade) + P(Test+|Sain)P(Sain)
    P_test_pos = sensibilite * prevalence + P_pos_sain * P_sain

    # Bayes: P(Malade|Test+)
    P_malade_test_pos = (sensibilite * prevalence) / P_test_pos

    return P_malade_test_pos

# Étude de l'effet de la prévalence
prevalences = np.linspace(0.001, 0.1, 100)
sensibilite = 0.95
specificite = 0.90

probs_posteriori = [diagnostic_bayesien(p, sensibilite, specificite)
                     for p in prevalences]

plt.figure(figsize=(10, 6))
plt.plot(prevalences * 100, np.array(probs_posteriori) * 100,
         linewidth=2, label='P(Malade|Test+)')
plt.axhline(sensibilite * 100, color='r', linestyle='--',
            label=f'Sensibilité = {sensibilite:.0%}')
plt.xlabel('Prévalence (%)')
plt.ylabel('P(Malade|Test+) (%)')
plt.title(f'Probabilité d\'être malade sachant test positif\n(Sensibilité={sensibilite:.0%}, Spécificité={specificite:.0%})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Cas concret
print("="*60)
print("CAS PRATIQUE: Test COVID-19")
print("="*60)
prevalence = 0.05  # 5% de la population infectée
sensibilite = 0.95  # 95% de détection des vrais positifs
specificite = 0.98  # 98% de détection des vrais négatifs

P_malade_test_pos = diagnostic_bayesien(prevalence, sensibilite, specificite)

print(f"\nPrévalence: {prevalence:.1%}")
print(f"Sensibilité: {sensibilite:.1%}")
print(f"Spécificité: {specificite:.1%}")
print(f"\nSi le test est positif:")
print(f"P(Malade|Test+) = {P_malade_test_pos:.1%}")
print(f"P(Sain|Test+) = {1-P_malade_test_pos:.1%}")

# Validation par simulation
n = 100000
malades = np.random.random(n) < prevalence

# Résultats des tests
tests = np.zeros(n, dtype=bool)
tests[malades] = np.random.random(malades.sum()) < sensibilite
tests[~malades] = np.random.random((~malades).sum()) < (1-specificite)

# P(Malade|Test+)
pos_et_malade = malades[tests].sum()
total_pos = tests.sum()
prob_simulation = pos_et_malade / total_pos

print(f"\nValidation par simulation (n={n}):")
print(f"P(Malade|Test+) = {prob_simulation:.1%}")
```

### Naive Bayes Classifier

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Générer données
X, y = make_classification(n_samples=1000, n_features=4, n_informative=3,
                           n_redundant=0, n_classes=3, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modèle Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Évaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Exemple de probabilités a posteriori
print("\nExemples de probabilités a posteriori:")
for i in range(5):
    print(f"Exemple {i+1}: P(Classe 0)={y_proba[i,0]:.3f}, "
          f"P(Classe 1)={y_proba[i,1]:.3f}, P(Classe 2)={y_proba[i,2]:.3f}")
```

---

## Applications au Machine Learning

### 1. Maximum de Vraisemblance (MLE)

**Principe** : Trouver les paramètres qui maximisent la probabilité d'observer les données.

$$
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \mathbb{P}(X_1, \ldots, X_n | \theta)
$$

```python
import numpy as np
from scipy import stats

# Données observées (supposées Gaussiennes)
data = np.array([1.2, 2.3, 1.8, 2.1, 1.9, 2.5, 1.7, 2.2])

# MLE pour μ et σ² d'une loi normale
mu_mle = np.mean(data)
sigma2_mle = np.var(data, ddof=0)  # ddof=0 pour MLE (pas estimateur non biaisé)

print(f"MLE: μ̂ = {mu_mle:.4f}, σ̂² = {sigma2_mle:.4f}")

# Comparaison avec scipy
params = stats.norm.fit(data)
print(f"Scipy: μ̂ = {params[0]:.4f}, σ̂ = {params[1]:.4f}")
```

### 2. Maximum A Posteriori (MAP)

**Principe** : Trouver les paramètres qui maximisent la probabilité a posteriori.

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} \mathbb{P}(\theta | X_1, \ldots, X_n) \propto \arg\max_{\theta} \mathbb{P}(X_1, \ldots, X_n | \theta) \cdot \mathbb{P}(\theta)
$$

### 3. Intervalles de Confiance

```python
from scipy import stats
import numpy as np

# Données
data = np.random.normal(5, 2, 100)

# Intervalle de confiance à 95% pour la moyenne
conf_level = 0.95
mean = np.mean(data)
se = stats.sem(data)  # Standard error
interval = stats.t.interval(conf_level, len(data)-1, loc=mean, scale=se)

print(f"Moyenne: {mean:.4f}")
print(f"Intervalle de confiance à 95%: [{interval[0]:.4f}, {interval[1]:.4f}]")
```

### 4. Tests d'Hypothèses

```python
from scipy import stats

# Deux échantillons
sample1 = np.random.normal(5, 1, 100)
sample2 = np.random.normal(5.3, 1, 100)

# Test t de Student (H0: les moyennes sont égales)
t_stat, p_value = stats.ttest_ind(sample1, sample2)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Rejet de H0: Les moyennes sont significativement différentes")
else:
    print("On ne rejette pas H0: Pas de différence significative")
```

### 5. Modèle Génératif vs Discriminatif

**Modèle génératif** (Naive Bayes, GMM) : Modélise $P(X, Y)$ ou $P(X|Y)$ et $P(Y)$

**Modèle discriminatif** (Régression logistique, SVM) : Modélise directement $P(Y|X)$

---

## Exercices Pratiques

### Exercice 1 : Lois de Probabilité

**Énoncé** : On lance un dé équilibré 20 fois. Quelle est la probabilité d'obtenir exactement 5 fois le chiffre 6 ?

**Solution** :

```python
from scipy import stats

# Paramètres
n = 20  # Nombre de lancers
p = 1/6  # Probabilité d'obtenir un 6
k = 5    # Nombre de 6 souhaités

# Loi binomiale
X = stats.binom(n, p)

# Probabilité
prob = X.pmf(k)
print(f"P(X = {k}) = {prob:.6f} = {prob:.2%}")

# Vérification par simulation
n_simulations = 100000
resultats = np.random.binomial(n, p, n_simulations)
prob_simulation = (resultats == k).mean()
print(f"Simulation: {prob_simulation:.6f}")
```

### Exercice 2 : Théorème de Bayes

**Énoncé** : Une usine a 3 machines A, B, C qui produisent respectivement 50%, 30%, 20% de la production totale. Les taux de défauts sont 2%, 3%, 4% respectivement. Un produit est tiré au hasard et est défectueux. Quelle est la probabilité qu'il provienne de la machine A ?

**Solution** :

```python
# Probabilités a priori
P_A = 0.50
P_B = 0.30
P_C = 0.20

# Vraisemblances
P_D_A = 0.02  # P(Défaut|A)
P_D_B = 0.03
P_D_C = 0.04

# P(Défaut) - Probabilité totale
P_D = P_D_A * P_A + P_D_B * P_B + P_D_C * P_C

# Bayes: P(A|Défaut)
P_A_D = (P_D_A * P_A) / P_D

print(f"P(Machine A | Défectueux) = {P_A_D:.4f} = {P_A_D:.1%}")
print(f"P(Machine B | Défectueux) = {(P_D_B * P_B) / P_D:.1%}")
print(f"P(Machine C | Défectueux) = {(P_D_C * P_C) / P_D:.1%}")
```

### Exercice 3 : Covariance et Corrélation

**Énoncé** : Générer deux variables aléatoires normales avec une corrélation de 0.7 et vérifier empiriquement.

**Solution** :

```python
import numpy as np

# Paramètres
n = 10000
rho = 0.7

# Matrice de covariance
Sigma = np.array([[1, rho],
                   [rho, 1]])

# Génération
samples = np.random.multivariate_normal([0, 0], Sigma, n)

# Corrélation empirique
corr_empirique = np.corrcoef(samples.T)[0, 1]
cov_empirique = np.cov(samples.T)[0, 1]

print(f"Corrélation théorique: {rho}")
print(f"Corrélation empirique: {corr_empirique:.4f}")
print(f"Covariance empirique: {cov_empirique:.4f}")

# Visualisation
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Variables normales corrélées (ρ = {rho})')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

---

## Résumé

### Points Clés à Retenir

1. **Fondements** :

   - Espace probabilisé $(\Omega, \mathcal{A}, \mathbb{P})$
   - Axiomes de Kolmogorov
   - Probabilités conditionnelles

2. **Variables Aléatoires** :

   - **Discrètes** : PMF $\mathbb{P}(X = x)$
   - **Continues** : PDF $f(x)$, CDF $F_X(t)$

3. **Lois Discrètes Importantes** :

   - **Bernoulli** $\mathcal{B}(p)$ : $\mathbb{E}=p$, $\mathbb{V}=p(1-p)$
   - **Binomiale** $\mathcal{B}(n,p)$ : $\mathbb{E}=np$, $\mathbb{V}=np(1-p)$
   - **Poisson** $\mathcal{P}(\lambda)$ : $\mathbb{E}=\mathbb{V}=\lambda$

4. **Lois Continues Importantes** :

   - **Uniforme** $\mathcal{U}([a,b])$ : $\mathbb{E}=\frac{a+b}{2}$
   - **Exponentielle** $\mathcal{E}(\lambda)$ : $\mathbb{E}=\frac{1}{\lambda}$
   - **Normale** $\mathcal{N}(\mu, \sigma^2)$ : $\mathbb{E}=\mu$, $\mathbb{V}=\sigma^2$

5. **Moments** :

   - **Espérance** : $\mathbb{E}[X]$ (valeur moyenne)
   - **Variance** : $\mathbb{V}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$

6. **Vecteurs Aléatoires** :

   - **Covariance** : $\text{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$
   - **Corrélation** : $\rho(X,Y) = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y} \in [-1,1]$
   - **Matrice de covariance** : symétrique, semi-définie positive

7. **Théorème de Bayes** :

$$
\mathbb{P}(A|B) = \frac{\mathbb{P}(B|A) \cdot \mathbb{P}(A)}{\mathbb{P}(B)}
$$

### Applications ML Essentielles

| Concept                  | Application ML                                  |
| ------------------------ | ----------------------------------------------- |
| Loi Normale              | Hypothèse dans de nombreux modèles (régression) |
| Bernoulli/Binomiale      | Classification binaire                          |
| Poisson                  | Modélisation d'événements rares                 |
| Théorème de Bayes        | Classificateurs bayésiens, filtres spam         |
| Covariance               | PCA, analyse de corrélation                     |
| Maximum de vraisemblance | Estimation de paramètres                        |
| Loi normale multivariée  | Modèles génératifs gaussiens                    |

### Checklist de Compétences

- [ ] Comprendre les axiomes de probabilité
- [ ] Calculer probabilités conditionnelles
- [ ] Utiliser le théorème de Bayes
- [ ] Identifier et utiliser les lois de probabilité courantes
- [ ] Calculer espérance et variance
- [ ] Manipuler vecteurs aléatoires
- [ ] Calculer et interpréter covariance et corrélation
- [ ] Appliquer ces concepts en ML (MLE, MAP, Naive Bayes)

### Formules Essentielles à Retenir

```
P(A|B) = P(A∩B) / P(B)

E[X] = Σ x·P(X=x)  (discret)
E[X] = ∫ x·f(x)dx  (continu)

Var[X] = E[X²] - (E[X])²

Cov(X,Y) = E[XY] - E[X]E[Y]

ρ(X,Y) = Cov(X,Y) / (σ_X σ_Y)
```

### Bibliothèques Python Essentielles

```python
import numpy as np                 # Calculs numériques
from scipy import stats            # Distributions, tests statistiques
import matplotlib.pyplot as plt    # Visualisation
import seaborn as sns              # Visualisation statistique
from sklearn.naive_bayes import *  # Classificateurs bayésiens
```

### Prochaine Étape

**Module 4 : Statistiques Descriptives** - Analyse exploratoire des données

---

**Navigation :**

- [⬅️ Module 2 : Algèbre Linéaire](02_Algebre_Lineaire.md)
- [🏠 Retour au Sommaire](README_ML.md)
- [➡️ Module 4 : Statistiques Descriptives](04_Statistiques_Descriptives.md)
