# Module 4 : Statistiques Descriptives

## 📋 Table des Matières
1. [Introduction](#introduction)
2. [Manipulation de Données avec Pandas](#manipulation-de-données-avec-pandas)
3. [Mesures de Tendance Centrale](#mesures-de-tendance-centrale)
4. [Mesures de Dispersion](#mesures-de-dispersion)
5. [Mesures de Forme](#mesures-de-forme)
6. [Analyse de Corrélation](#analyse-de-corrélation)
7. [Visualisation de Données](#visualisation-de-données)
8. [Analyse Exploratoire de Données (EDA)](#analyse-exploratoire-de-données-eda)
9. [Exercices Pratiques](#exercices-pratiques)
10. [Résumé](#résumé)

---

## Introduction

Les **statistiques descriptives** constituent la première étape essentielle dans toute analyse de données. Elles permettent de résumer, organiser et présenter les données de manière compréhensible et informative.

### Objectifs des Statistiques Descriptives

1. **Résumer les données** : Extraire l'information essentielle de grandes quantités de données
2. **Détecter les patterns** : Identifier les tendances et structures dans les données
3. **Identifier les anomalies** : Repérer les valeurs aberrantes (outliers)
4. **Préparer la modélisation** : Comprendre les caractéristiques des données avant le ML

### Deux Grands Types de Statistiques

**Statistiques descriptives** :
- Résument et décrivent les données observées
- Mesures de tendance centrale, dispersion, forme
- Visualisations (histogrammes, boxplots, etc.)

**Statistiques inférentielles** :
- Font des inférences sur une population à partir d'un échantillon
- Tests d'hypothèses, intervalles de confiance
- Prédictions et généralisations

### Importance en Machine Learning

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration pour de belles visualisations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
```

---

## Manipulation de Données avec Pandas

### Introduction à Pandas

**Pandas** est la bibliothèque Python incontournable pour la manipulation et l'analyse de données structurées.

**Structures de données principales** :
- **Series** : Tableau 1D indexé
- **DataFrame** : Tableau 2D (lignes × colonnes), similaire à une table SQL ou un tableur Excel

### Création de DataFrames

#### 1. DataFrame Vide

```python
import pandas as pd

# Créer un DataFrame vide
empty_df = pd.DataFrame()
print(type(empty_df))  # <class 'pandas.core.frame.DataFrame'>
```

#### 2. À partir d'un Dictionnaire

```python
# Dictionnaire avec des listes
data = {
    "Nom": ["Alice", "Bob", "Charlie", "Diana"],
    "Age": [25, 30, 35, 28],
    "Ville": ["Paris", "Lyon", "Marseille", "Paris"],
    "Salaire": [45000, 52000, 48000, 51000]
}

df = pd.DataFrame(data)
print(df)
```

**Sortie** :
```
       Nom  Age      Ville  Salaire
0    Alice   25      Paris    45000
1      Bob   30       Lyon    52000
2  Charlie   35  Marseille    48000
3    Diana   28      Paris    51000
```

#### 3. À partir d'un Array NumPy

```python
import numpy as np

# Array aléatoire 5×3
array = np.random.randn(5, 3)

# Créer DataFrame avec noms de colonnes
df = pd.DataFrame(array, columns=['A', 'B', 'C'])
print(df)
```

### Opérations de Base sur les DataFrames

#### Dimensions et Informations

```python
# Forme du DataFrame (lignes, colonnes)
print(df.shape)  # (4, 4)

# Nombre de lignes
print(f"Nombre de lignes : {df.shape[0]}")

# Nombre de colonnes
print(f"Nombre de colonnes : {df.shape[1]}")

# Noms des colonnes
print(df.columns.tolist())  # ['Nom', 'Age', 'Ville', 'Salaire']

# Informations générales
df.info()

# Premières lignes
df.head()  # 5 premières par défaut
df.head(3)  # 3 premières

# Dernières lignes
df.tail(2)  # 2 dernières
```

#### Accès aux Colonnes

```python
# Accéder à une colonne (retourne une Series)
ages = df["Age"]
print(type(ages))  # <class 'pandas.core.series.Series'>

# Accéder à plusieurs colonnes (retourne un DataFrame)
subset = df[["Nom", "Salaire"]]
print(subset)
```

#### Accès aux Lignes

```python
# Accès par position avec iloc
premiere_ligne = df.iloc[0]  # Première ligne
print(premiere_ligne)

# Plusieurs lignes
lignes_1_et_3 = df.iloc[[1, 3]]  # Lignes d'index 1 et 3

# Plage de lignes (slicing)
lignes_1_a_3 = df.iloc[1:4]  # Lignes 1, 2, 3 (4 exclu)

# Accès par label avec loc
df_indexed = df.set_index('Nom')
alice_data = df_indexed.loc['Alice']
```

#### Filtrage et Sélection Conditionnelle

```python
# Filtrer les personnes de plus de 28 ans
seniors = df[df["Age"] > 28]
print(seniors)

# Conditions multiples avec &amp; (ET) et | (OU)
# Personnes de Paris avec salaire > 46000
parisiens_riches = df[(df["Ville"] == "Paris") &amp; (df["Salaire"] > 46000)]

# Personnes de Paris OU avec salaire > 50000
paris_ou_riche = df[(df["Ville"] == "Paris") | (df["Salaire"] > 50000)]

# NOT avec ~
pas_paris = df[~(df["Ville"] == "Paris")]
```

#### Ajouter et Supprimer des Colonnes

```python
# Ajouter une colonne
df["Bonus"] = df["Salaire"] * 0.1
print(df)

# Ajouter une colonne calculée
df["Salaire_Annuel"] = df["Salaire"] * 12

# Supprimer une colonne (sans modifier l'original)
df_sans_bonus = df.drop("Bonus", axis=1)

# Supprimer plusieurs colonnes
df_reduit = df.drop(["Bonus", "Salaire_Annuel"], axis=1)

# Supprimer une colonne en place (modifie l'original)
df.drop("Bonus", axis=1, inplace=True)
```

#### Supprimer des Lignes

```python
# Supprimer une ligne par index
df_sans_ligne_2 = df.drop(2)  # Supprime la ligne d'index 2

# Supprimer plusieurs lignes
df_sans_0_et_3 = df.drop([0, 3])
```

#### Valeurs Manquantes

```python
# Créer un DataFrame avec valeurs manquantes
data_with_nan = {
    "A": [1, 2, np.nan, 4],
    "B": [5, np.nan, 7, 8],
    "C": [9, 10, 11, 12]
}
df_nan = pd.DataFrame(data_with_nan)

# Détecter les valeurs manquantes
print(df_nan.isna())  # Retourne DataFrame de booléens

# Compter les valeurs manquantes par colonne
print(df_nan.isna().sum())

# Supprimer les lignes avec valeurs manquantes
df_clean = df_nan.dropna()

# Remplir les valeurs manquantes
df_filled = df_nan.fillna(0)  # Remplacer par 0
df_filled_mean = df_nan.fillna(df_nan.mean())  # Remplacer par la moyenne

# Remplir en place
df_nan.fillna(0, inplace=True)
```

#### Tri

```python
# Trier par une colonne
df_sorted = df.sort_values(by="Age")  # Ordre croissant
df_sorted_desc = df.sort_values(by="Age", ascending=False)  # Décroissant

# Trier par plusieurs colonnes
df_multi_sort = df.sort_values(by=["Ville", "Age"])
```

#### Sauvegarde et Chargement

```python
# Sauvegarder en CSV
df.to_csv("data.csv", index=False)  # index=False pour ne pas sauvegarder l'index

# Charger depuis CSV
df_loaded = pd.read_csv("data.csv")

# Autres formats
df.to_excel("data.xlsx", index=False)  # Excel
df.to_json("data.json")  # JSON
df.to_parquet("data.parquet")  # Parquet (efficace pour gros volumes)
```

---

## Mesures de Tendance Centrale

Les mesures de tendance centrale donnent une **valeur typique** qui représente le centre de la distribution des données.

### 1. Moyenne (Mean)

**Définition** : Somme de toutes les valeurs divisée par le nombre de valeurs.

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

**Propriétés** :
- Sensible aux valeurs extrêmes (outliers)
- Utilisée pour des données numériques continues
- Minimise la somme des écarts au carré

```python
import numpy as np
import pandas as pd

# Données
data = np.array([10, 20, 30, 40, 50])

# Moyenne avec NumPy
mean_np = np.mean(data)
print(f"Moyenne (NumPy): {mean_np}")  # 30.0

# Moyenne avec Pandas
df = pd.DataFrame({"valeurs": data})
mean_pd = df["valeurs"].mean()
print(f"Moyenne (Pandas): {mean_pd}")  # 30.0

# Effet d'une valeur extrême
data_with_outlier = np.array([10, 20, 30, 40, 1000])
print(f"Moyenne avec outlier: {np.mean(data_with_outlier)}")  # 220.0
```

### 2. Médiane (Median)

**Définition** : Valeur centrale qui divise les données ordonnées en deux parties égales.

- Si $n$ est impair : médiane = valeur centrale
- Si $n$ est pair : médiane = moyenne des deux valeurs centrales

**Propriétés** :
- **Robuste** aux valeurs extrêmes
- Utilisée pour des données ordinales ou continues
- 50ème percentile

```python
# Médiane avec NumPy
median_np = np.median(data)
print(f"Médiane: {median_np}")  # 30.0

# Médiane avec Pandas
median_pd = df["valeurs"].median()

# Comparaison médiane vs moyenne avec outlier
data_with_outlier = np.array([10, 20, 30, 40, 1000])
print(f"Moyenne: {np.mean(data_with_outlier)}")   # 220.0
print(f"Médiane: {np.median(data_with_outlier)}")  # 30.0 (robuste!)
```

### 3. Mode

**Définition** : Valeur qui apparaît le plus fréquemment.

**Propriétés** :
- Peut être utilisé pour données catégorielles
- Peut avoir plusieurs modes (distribution multimodale)

```python
from scipy import stats

# Données avec mode
data_mode = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])

# Mode avec scipy
mode_result = stats.mode(data_mode, keepdims=True)
print(f"Mode: {mode_result.mode[0]}")  # 3
print(f"Fréquence: {mode_result.count[0]}")  # 3

# Mode avec Pandas
df_mode = pd.DataFrame({"valeurs": data_mode})
mode_pd = df_mode["valeurs"].mode()[0]
print(f"Mode (Pandas): {mode_pd}")  # 3
```

### Comparaison des Mesures

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution asymétrique
np.random.seed(42)
data_skewed = np.concatenate([
    np.random.normal(50, 10, 900),
    np.random.normal(100, 5, 100)  # Quelques valeurs élevées
])

# Calculer les mesures
mean_val = np.mean(data_skewed)
median_val = np.median(data_skewed)
mode_val = stats.mode(data_skewed, keepdims=True).mode[0]

# Visualisation
plt.figure(figsize=(12, 6))
plt.hist(data_skewed, bins=50, density=True, alpha=0.7, edgecolor='black')
plt.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Moyenne = {mean_val:.2f}')
plt.axvline(median_val, color='g', linestyle='--', linewidth=2, label=f'Médiane = {median_val:.2f}')
plt.xlabel('Valeur')
plt.ylabel('Densité')
plt.title('Distribution asymétrique : Comparaison Moyenne vs Médiane')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Moyenne: {mean_val:.2f}")
print(f"Médiane: {median_val:.2f}")
print(f"Écart: {abs(mean_val - median_val):.2f}")
```

**Interprétation** :
- Distribution **symétrique** : moyenne ≈ médiane
- Distribution **asymétrique à droite** (skew positif) : moyenne > médiane
- Distribution **asymétrique à gauche** (skew négatif) : moyenne < médiane

---

## Mesures de Dispersion

Les mesures de dispersion quantifient la **variabilité** ou l'**étalement** des données autour de la tendance centrale.

### 1. Étendue (Range)

**Définition** : Différence entre la valeur maximale et minimale.

$$
\text{Étendue} = \max(x) - \min(x)
$$

```python
data = np.array([10, 15, 20, 25, 30, 100])

range_val = np.max(data) - np.min(data)
print(f"Étendue: {range_val}")  # 90

# Avec Pandas
df = pd.DataFrame({"valeurs": data})
range_pd = df["valeurs"].max() - df["valeurs"].min()
```

**Limite** : Très sensible aux valeurs extrêmes.

### 2. Variance

**Définition** : Moyenne des écarts au carré par rapport à la moyenne.

**Variance de la population** :
$$
\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2
$$

**Variance de l'échantillon** (estimateur non biaisé) :
$$
s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

```python
data = np.array([10, 20, 30, 40, 50])

# Variance de la population (ddof=0)
var_population = np.var(data, ddof=0)
print(f"Variance population: {var_population}")  # 200.0

# Variance de l'échantillon (ddof=1)
var_sample = np.var(data, ddof=1)
print(f"Variance échantillon: {var_sample}")  # 250.0

# Avec Pandas (ddof=1 par défaut)
var_pd = pd.Series(data).var()
print(f"Variance (Pandas): {var_pd}")  # 250.0
```

**Propriétés** :
- Toujours positive ou nulle
- Unité : carré de l'unité des données
- Sensible aux outliers

### 3. Écart-Type (Standard Deviation)

**Définition** : Racine carrée de la variance.

$$
\sigma = \sqrt{\sigma^2}
$$

```python
# Écart-type
std_population = np.std(data, ddof=0)
std_sample = np.std(data, ddof=1)

print(f"Écart-type population: {std_population:.2f}")  # 14.14
print(f"Écart-type échantillon: {std_sample:.2f}")  # 15.81

# Avec Pandas
std_pd = pd.Series(data).std()
```

**Avantage** : Même unité que les données (contrairement à la variance).

**Interprétation** :
- Écart-type faible : données concentrées autour de la moyenne
- Écart-type élevé : données dispersées

### 4. Coefficient de Variation (CV)

**Définition** : Ratio de l'écart-type sur la moyenne (en pourcentage).

$$
CV = \frac{\sigma}{\mu} \times 100\%
$$

```python
mean = np.mean(data)
std = np.std(data, ddof=1)
cv = (std / mean) * 100

print(f"Coefficient de variation: {cv:.2f}%")
```

**Utilité** : Comparer la variabilité relative de distributions avec des moyennes différentes.

### 5. Quartiles et Quantiles

**Quartiles** :
- **Q1** (1er quartile, 25ème percentile) : 25% des données sont en dessous
- **Q2** (2ème quartile, médiane, 50ème percentile) : 50% en dessous
- **Q3** (3ème quartile, 75ème percentile) : 75% en dessous

**Interquartile Range (IQR)** :
$$
IQR = Q3 - Q1
$$

```python
data = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])

# Quartiles
q1 = np.percentile(data, 25)
q2 = np.percentile(data, 50)  # Médiane
q3 = np.percentile(data, 75)

print(f"Q1 (25%): {q1}")
print(f"Q2 (50%, médiane): {q2}")
print(f"Q3 (75%): {q3}")

# IQR
iqr = q3 - q1
print(f"IQR: {iqr}")

# Avec Pandas
df = pd.DataFrame({"valeurs": data})
print(df["valeurs"].quantile([0.25, 0.5, 0.75]))
```

**Détection d'outliers avec IQR** :
- **Outliers modérés** : valeurs < Q1 - 1.5×IQR ou > Q3 + 1.5×IQR
- **Outliers extrêmes** : valeurs < Q1 - 3×IQR ou > Q3 + 3×IQR

```python
# Détection d'outliers
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = data[(data < lower_bound) | (data > upper_bound)]
print(f"Valeurs aberrantes: {outliers}")
```

### 6. Résumé Statistique

```python
# Avec NumPy
data = np.random.randn(1000)

print(f"Moyenne: {np.mean(data):.4f}")
print(f"Médiane: {np.median(data):.4f}")
print(f"Écart-type: {np.std(data, ddof=1):.4f}")
print(f"Min: {np.min(data):.4f}")
print(f"Max: {np.max(data):.4f}")

# Avec Pandas (plus simple)
df = pd.DataFrame({"valeurs": data})
print(df.describe())
```

**Output de `describe()`** :
```
       valeurs
count  1000.000000
mean      0.014234
std       1.012345
min      -3.123456
25%      -0.678901
50%       0.012345
75%       0.701234
max       3.234567
```

---

## Mesures de Forme

Les mesures de forme caractérisent la **forme de la distribution** des données.

### 1. Asymétrie (Skewness)

**Définition** : Mesure de l'asymétrie de la distribution par rapport à la moyenne.

$$
\text{Skewness} = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{\sigma}\right)^3
$$

**Interprétation** :
- **Skewness = 0** : Distribution symétrique (normale)
- **Skewness > 0** : Distribution asymétrique à droite (queue à droite, valeurs élevées)
- **Skewness < 0** : Distribution asymétrique à gauche (queue à gauche, valeurs faibles)

**Règles pratiques** :
- $|\text{Skewness}| < 0.5$ : Approximativement symétrique
- $0.5 < |\text{Skewness}| < 1$ : Modérément asymétrique
- $|\text{Skewness}| > 1$ : Fortement asymétrique

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Générer différentes distributions
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
right_skewed = np.random.exponential(2, 1000)
left_skewed = -np.random.exponential(2, 1000)

# Calculer skewness
skew_normal = stats.skew(normal_data)
skew_right = stats.skew(right_skewed)
skew_left = stats.skew(left_skewed)

print(f"Skewness normale: {skew_normal:.4f}")
print(f"Skewness à droite: {skew_right:.4f}")
print(f"Skewness à gauche: {skew_left:.4f}")

# Avec Pandas
df = pd.DataFrame({
    "normale": normal_data,
    "droite": right_skewed,
    "gauche": left_skewed
})
print(df.skew())

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(normal_data, bins=30, edgecolor='black', alpha=0.7)
axes[0].set_title(f'Symétrique\nSkew = {skew_normal:.2f}')
axes[0].axvline(np.mean(normal_data), color='r', linestyle='--', label='Moyenne')
axes[0].axvline(np.median(normal_data), color='g', linestyle='--', label='Médiane')
axes[0].legend()

axes[1].hist(right_skewed, bins=30, edgecolor='black', alpha=0.7)
axes[1].set_title(f'Asymétrique à droite\nSkew = {skew_right:.2f}')
axes[1].axvline(np.mean(right_skewed), color='r', linestyle='--', label='Moyenne')
axes[1].axvline(np.median(right_skewed), color='g', linestyle='--', label='Médiane')
axes[1].legend()

axes[2].hist(left_skewed, bins=30, edgecolor='black', alpha=0.7)
axes[2].set_title(f'Asymétrique à gauche\nSkew = {skew_left:.2f}')
axes[2].axvline(np.mean(left_skewed), color='r', linestyle='--', label='Moyenne')
axes[2].axvline(np.median(left_skewed), color='g', linestyle='--', label='Médiane')
axes[2].legend()

plt.tight_layout()
plt.show()
```

### 2. Aplatissement (Kurtosis)

**Définition** : Mesure de l'aplatissement ou de la concentration des valeurs dans les queues de distribution.

$$
\text{Kurtosis} = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{\sigma}\right)^4 - 3
$$

**Interprétation** (kurtosis "excess", -3 pour centrer sur 0) :
- **Kurtosis = 0** : Distribution mesokurtique (normale)
- **Kurtosis > 0** : Distribution leptokurtique (queues lourdes, pic pointu)
- **Kurtosis < 0** : Distribution platykurtique (queues légères, pic aplati)

```python
# Calculer kurtosis
kurt_normal = stats.kurtosis(normal_data)  # Excess kurtosis (Fisher)
print(f"Kurtosis normale: {kurt_normal:.4f}")  # ≈ 0

# Avec Pandas
print(df.kurtosis())

# Distribution avec kurtosis élevée (Student t, df=3)
high_kurt = stats.t.rvs(df=3, size=1000)
kurt_high = stats.kurtosis(high_kurt)

# Distribution avec kurtosis faible (Uniforme)
low_kurt = np.random.uniform(-2, 2, 1000)
kurt_low = stats.kurtosis(low_kurt)

print(f"Kurtosis élevée: {kurt_high:.4f}")
print(f"Kurtosis faible: {kurt_low:.4f}")

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(normal_data, bins=30, density=True, edgecolor='black', alpha=0.7)
axes[0].set_title(f'Mesokurtique (Normale)\nKurt = {kurt_normal:.2f}')

axes[1].hist(high_kurt, bins=30, density=True, edgecolor='black', alpha=0.7)
axes[1].set_title(f'Leptokurtique (Queues lourdes)\nKurt = {kurt_high:.2f}')

axes[2].hist(low_kurt, bins=30, density=True, edgecolor='black', alpha=0.7)
axes[2].set_title(f'Platykurtique (Queues légères)\nKurt = {kurt_low:.2f}')

plt.tight_layout()
plt.show()
```

---

## Analyse de Corrélation

La **corrélation** mesure la force et la direction de la relation linéaire entre deux variables.

### Coefficient de Corrélation de Pearson

**Définition** :
$$
r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
$$

**Propriétés** :
- $r \in [-1, 1]$
- $r = 1$ : Corrélation positive parfaite
- $r = -1$ : Corrélation négative parfaite
- $r = 0$ : Pas de corrélation linéaire
- $|r| > 0.7$ : Corrélation forte
- $0.3 < |r| < 0.7$ : Corrélation modérée
- $|r| < 0.3$ : Corrélation faible

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Générer des données corrélées
np.random.seed(42)
n = 100

# Corrélation positive forte
x1 = np.random.randn(n)
y1 = 2 * x1 + np.random.randn(n) * 0.5

# Corrélation négative forte
x2 = np.random.randn(n)
y2 = -1.5 * x2 + np.random.randn(n) * 0.5

# Pas de corrélation
x3 = np.random.randn(n)
y3 = np.random.randn(n)

# Calculer corrélations
corr1 = np.corrcoef(x1, y1)[0, 1]
corr2 = np.corrcoef(x2, y2)[0, 1]
corr3 = np.corrcoef(x3, y3)[0, 1]

print(f"Corrélation positive: r = {corr1:.3f}")
print(f"Corrélation négative: r = {corr2:.3f}")
print(f"Pas de corrélation: r = {corr3:.3f}")

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(x1, y1, alpha=0.6)
axes[0].set_title(f'Corrélation positive\nr = {corr1:.3f}')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(x2, y2, alpha=0.6)
axes[1].set_title(f'Corrélation négative\nr = {corr2:.3f}')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].grid(True, alpha=0.3)

axes[2].scatter(x3, y3, alpha=0.6)
axes[2].set_title(f'Pas de corrélation\nr = {corr3:.3f}')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Matrice de Corrélation

```python
# Créer un DataFrame avec plusieurs variables
data = {
    'A': np.random.randn(100),
    'B': np.random.randn(100),
    'C': np.random.randn(100),
    'D': np.random.randn(100)
}
data['E'] = data['A'] * 2 + np.random.randn(100) * 0.5  # Corrélée avec A
data['F'] = -data['B'] * 1.5 + np.random.randn(100) * 0.5  # Anti-corrélée avec B

df = pd.DataFrame(data)

# Matrice de corrélation
corr_matrix = df.corr()
print(corr_matrix)

# Visualisation avec heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, square=True, linewidths=1)
plt.title('Matrice de Corrélation')
plt.tight_layout()
plt.show()
```

### Attention : Corrélation ≠ Causalité

```python
# Exemple : Corrélation sans causalité
np.random.seed(42)
n = 100

# Variable cachée (température)
temperature = np.random.randn(n) * 10 + 25

# Deux variables influencées par la température
ventes_glaces = temperature * 10 + np.random.randn(n) * 20
noyades = temperature * 2 + np.random.randn(n) * 5

# Corrélation entre glaces et noyades
corr_spurious = np.corrcoef(ventes_glaces, noyades)[0, 1]
print(f"Corrélation glaces-noyades: r = {corr_spurious:.3f}")
print("⚠️ Corrélation ne signifie PAS causalité!")

plt.figure(figsize=(8, 6))
plt.scatter(ventes_glaces, noyades, alpha=0.6)
plt.xlabel('Ventes de glaces')
plt.ylabel('Nombre de noyades')
plt.title(f'Corrélation spurieuse (r = {corr_spurious:.2f})')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Visualisation de Données

La visualisation est essentielle pour comprendre les données et communiquer les résultats.

### 1. Histogramme

**Utilité** : Distribution d'une variable continue.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Données
np.random.seed(42)
data = np.random.normal(100, 15, 1000)

# Histogramme simple
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Valeur')
plt.ylabel('Fréquence')
plt.title('Histogramme')
plt.grid(True, alpha=0.3)
plt.show()

# Histogramme avec densité (Seaborn)
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=30, kde=True, stat='density', color='darkblue')
plt.xlabel('Valeur')
plt.ylabel('Densité')
plt.title('Histogramme avec Courbe de Densité')
plt.show()
```

### 2. Boxplot (Boîte à Moustaches)

**Utilité** : Résumé visuel de la distribution (quartiles, médiane, outliers).

**Composantes** :
- **Boîte** : De Q1 à Q3 (IQR)
- **Ligne dans la boîte** : Médiane
- **Moustaches** : Jusqu'à 1.5×IQR
- **Points** : Outliers

```python
# Données
data_groups = {
    'Groupe A': np.random.normal(50, 10, 100),
    'Groupe B': np.random.normal(60, 15, 100),
    'Groupe C': np.random.normal(55, 8, 100)
}
df = pd.DataFrame(data_groups)

# Boxplot avec Matplotlib
plt.figure(figsize=(10, 6))
plt.boxplot(df.values, labels=df.columns)
plt.ylabel('Valeur')
plt.title('Boxplot')
plt.grid(True, alpha=0.3)
plt.show()

# Boxplot avec Seaborn (plus joli)
plt.figure(figsize=(10, 6))
df_melted = df.melt(var_name='Groupe', value_name='Valeur')
sns.boxplot(data=df_melted, x='Groupe', y='Valeur', palette='Set2')
plt.title('Boxplot avec Seaborn')
plt.show()
```

### 3. Scatter Plot (Nuage de Points)

**Utilité** : Relation entre deux variables continues.

```python
# Données
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

# Scatter plot simple
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.grid(True, alpha=0.3)
plt.show()

# Avec Seaborn (avec régression linéaire)
plt.figure(figsize=(8, 6))
sns.regplot(x=x, y=y, scatter_kws={'alpha':0.6})
plt.title('Scatter Plot avec Régression')
plt.show()
```

### 4. Pairplot

**Utilité** : Visualiser toutes les relations par paires dans un DataFrame.

```python
# Créer DataFrame
df = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randn(100),
    'C': np.random.randn(100),
    'Catégorie': np.random.choice(['X', 'Y'], 100)
})
df['D'] = df['A'] * 2 + np.random.randn(100) * 0.3

# Pairplot
sns.pairplot(df, hue='Catégorie', diag_kind='kde')
plt.suptitle('Pairplot', y=1.02)
plt.show()
```

### 5. Heatmap

**Utilité** : Visualiser matrices (corrélations, données tabulaires).

```python
# Matrice de corrélation
corr = df[['A', 'B', 'C', 'D']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1)
plt.title('Heatmap de Corrélation')
plt.show()
```

### 6. Violin Plot

**Utilité** : Combine boxplot et densité de probabilité.

```python
plt.figure(figsize=(10, 6))
sns.violinplot(data=df_melted, x='Groupe', y='Valeur', palette='muted')
plt.title('Violin Plot')
plt.show()
```

---

## Analyse Exploratoire de Données (EDA)

L'**Analyse Exploratoire de Données** (Exploratory Data Analysis, EDA) est une étape cruciale avant toute modélisation.

### Checklist EDA Complète

1. **Charger et inspecter les données**
2. **Comprendre la structure**
3. **Vérifier les types de données**
4. **Détecter et traiter les valeurs manquantes**
5. **Analyser les statistiques descriptives**
6. **Visualiser les distributions**
7. **Détecter les outliers**
8. **Analyser les corrélations**
9. **Identifier les patterns**

### Exemple Complet d'EDA

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Charger les données
# Simulons un dataset
np.random.seed(42)
n = 500

data = {
    'age': np.random.randint(18, 70, n),
    'revenu': np.random.lognormal(10.5, 0.5, n),
    'education': np.random.choice(['Lycée', 'Licence', 'Master', 'Doctorat'], n),
    'experience': np.random.randint(0, 40, n),
    'score_credit': np.random.randint(300, 850, n)
}

# Ajouter corrélation
data['revenu'] = data['revenu'] + data['experience'] * 500 + np.random.randn(n) * 1000

df = pd.DataFrame(data)

# 2. Inspection initiale
print("=" * 60)
print("APERÇU DES DONNÉES")
print("=" * 60)
print(df.head(10))
print(f"\nDimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")

# 3. Informations sur les colonnes
print("\n" + "=" * 60)
print("INFORMATIONS SUR LES COLONNES")
print("=" * 60)
df.info()

# 4. Types de données
print("\n" + "=" * 60)
print("TYPES DE DONNÉES")
print("=" * 60)
print(df.dtypes)

# 5. Valeurs manquantes
print("\n" + "=" * 60)
print("VALEURS MANQUANTES")
print("=" * 60)
print(df.isna().sum())
print(f"Pourcentage total: {df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%")

# 6. Statistiques descriptives
print("\n" + "=" * 60)
print("STATISTIQUES DESCRIPTIVES")
print("=" * 60)
print(df.describe())

# Variables numériques
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\nVariables numériques: {list(numeric_cols)}")

# Variables catégorielles
cat_cols = df.select_dtypes(include=['object']).columns
print(f"Variables catégorielles: {list(cat_cols)}")

# 7. Distribution des variables catégorielles
print("\n" + "=" * 60)
print("DISTRIBUTION DES VARIABLES CATÉGORIELLES")
print("=" * 60)
for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())
    print(f"Pourcentages:")
    print(df[col].value_counts(normalize=True) * 100)

# 8. Skewness et Kurtosis
print("\n" + "=" * 60)
print("ASYMÉTRIE (SKEWNESS) ET APLATISSEMENT (KURTOSIS)")
print("=" * 60)
print("\nSkewness:")
print(df[numeric_cols].skew())
print("\nKurtosis:")
print(df[numeric_cols].kurtosis())

# 9. Visualisations

# Histogrammes des variables numériques
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numeric_cols):
    axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution de {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Fréquence')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Boxplots pour détecter outliers
fig, axes = plt.subplots(1, len(numeric_cols), figsize=(15, 5))

for idx, col in enumerate(numeric_cols):
    axes[idx].boxplot(df[col])
    axes[idx].set_title(f'Boxplot: {col}')
    axes[idx].set_ylabel('Valeur')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 10. Matrice de corrélation
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matrice de Corrélation', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# 11. Pairplot
sns.pairplot(df[numeric_cols], diag_kind='kde', corner=True)
plt.suptitle('Pairplot des Variables Numériques', y=1.02)
plt.show()

# 12. Détection d'outliers (méthode IQR)
print("\n" + "=" * 60)
print("DÉTECTION D'OUTLIERS (MÉTHODE IQR)")
print("=" * 60)

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"\n{col}:")
    print(f"  Bornes: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Nombre d'outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

print("\n" + "=" * 60)
print("FIN DE L'ANALYSE EXPLORATOIRE")
print("=" * 60)
```

---

## Exercices Pratiques

### Exercice 1 : Analyse d'un Dataset Médical

**Dataset** : Assurance médicale (Medical Insurance)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données (simulées)
np.random.seed(42)
n = 1000

medical_data = pd.DataFrame({
    'age': np.random.randint(18, 65, n),
    'sex': np.random.choice(['male', 'female'], n),
    'bmi': np.random.normal(28, 6, n),
    'children': np.random.randint(0, 5, n),
    'smoker': np.random.choice(['yes', 'no'], n, p=[0.2, 0.8]),
    'region': np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], n),
})

# Charges calculées avec une formule réaliste
base_charge = 3000
medical_data['charges'] = (
    base_charge +
    medical_data['age'] * 250 +
    medical_data['bmi'] * 400 +
    medical_data['children'] * 500 +
    (medical_data['smoker'] == 'yes') * 23000 +
    np.random.normal(0, 3000, n)
)
medical_data['charges'] = medical_data['charges'].clip(lower=1000)
```

**Questions** :

1. Afficher les 5 premières lignes
2. Quelle est la forme du dataset ?
3. Combien d'individus fument ?
4. Sélectionner les hommes de la région "southeast" qui ne fument pas
5. Sélectionner les personnes avec exactement 1 enfant
6. Calculer la moyenne des charges pour les 25-40 ans sans enfants
7. Calculer moyenne et écart-type des charges par région pour les fumeurs
8. Calculer skewness et kurtosis des variables numériques
9. Visualiser charges en fonction de l'âge (scatter plot, couleur par fumeur)
10. Créer un boxplot des charges par région

**Solutions** :

```python
# 1. Premières lignes
print(medical_data.head(5))

# 2. Forme
print(f"Forme: {medical_data.shape}")

# 3. Nombre de fumeurs
n_smokers = (medical_data['smoker'] == 'yes').sum()
print(f"Nombre de fumeurs: {n_smokers}")

# 4. Hommes du southeast non-fumeurs
selection = medical_data[
    (medical_data['sex'] == 'male') &
    (medical_data['region'] == 'southeast') &
    (medical_data['smoker'] == 'no')
]
print(f"Hommes southeast non-fumeurs: {len(selection)}")

# 5. Personnes avec 1 enfant
one_child = medical_data[medical_data['children'] == 1]
print(f"Personnes avec 1 enfant: {len(one_child)}")

# 6. Moyenne charges 25-40 ans sans enfants
charges_25_40_no_kids = medical_data[
    (medical_data['age'] >= 25) &
    (medical_data['age'] <= 40) &
    (medical_data['children'] == 0)
]['charges'].mean()
print(f"Moyenne charges (25-40, 0 enfants): {charges_25_40_no_kids:.2f}")

# 7. Moyenne et std des charges par région pour fumeurs
smokers = medical_data[medical_data['smoker'] == 'yes']
stats_by_region = smokers.groupby('region')['charges'].agg(['mean', 'std'])
print("\nStatistiques charges fumeurs par région:")
print(stats_by_region)

# 8. Skewness et kurtosis
numeric_cols = ['age', 'bmi', 'children', 'charges']
print("\nSkewness:")
print(medical_data[numeric_cols].skew())
print("\nKurtosis:")
print(medical_data[numeric_cols].kurtosis())

# 9. Scatter plot charges vs age (couleur par fumeur)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=medical_data, x='age', y='charges',
                hue='smoker', palette=['green', 'red'], alpha=0.6)
plt.title('Charges en fonction de l\'âge')
plt.xlabel('Âge')
plt.ylabel('Charges ($)')
plt.legend(title='Fumeur')
plt.grid(True, alpha=0.3)
plt.show()

# 10. Boxplot charges par région
plt.figure(figsize=(10, 6))
sns.boxplot(data=medical_data, x='region', y='charges', palette='Set2')
plt.title('Distribution des charges par région')
plt.xlabel('Région')
plt.ylabel('Charges ($)')
plt.xticks(rotation=45)
plt.show()
```

### Exercice 2 : Analyse de Prix de Maisons

```python
# Créer un dataset de prix de maisons
np.random.seed(42)
n = 500

house_data = pd.DataFrame({
    'LotArea': np.random.randint(5000, 20000, n),
    'OverallQual': np.random.randint(1, 11, n),
    'YearBuilt': np.random.randint(1950, 2023, n),
    'GrLivArea': np.random.randint(800, 4000, n),
    'FullBath': np.random.randint(1, 4, n),
    'BedroomAbvGr': np.random.randint(1, 6, n),
    'GarageCars': np.random.randint(0, 4, n)
})

# Prix calculé
house_data['SalePrice'] = (
    house_data['GrLivArea'] * 100 +
    house_data['OverallQual'] * 10000 +
    house_data['GarageCars'] * 5000 +
    (2023 - house_data['YearBuilt']) * (-200) +
    np.random.normal(0, 20000, n)
).clip(lower=50000)

# Ajouter quelques valeurs manquantes
house_data.loc[np.random.choice(n, 20, replace=False), 'LotArea'] = np.nan
house_data.loc[np.random.choice(n, 15, replace=False), 'GarageCars'] = np.nan
```

**Questions** :

1. Afficher les informations du dataset
2. Compter les valeurs manquantes par colonne
3. Remplir les valeurs manquantes avec 0
4. Statistiques descriptives de SalePrice
5. Histogramme et densité de SalePrice
6. Matrice de corrélation
7. Scatter plot SalePrice vs GrLivArea
8. Boxplot SalePrice par OverallQual

**Solutions** :

```python
# 1. Informations
print(house_data.info())

# 2. Valeurs manquantes
print("\nValeurs manquantes:")
print(house_data.isna().sum())

# 3. Remplir par 0
house_data.fillna(0, inplace=True)
print("\nAprès remplissage:")
print(house_data.isna().sum())

# 4. Stats SalePrice
print("\nStatistiques SalePrice:")
print(house_data['SalePrice'].describe())
print(f"Skewness: {house_data['SalePrice'].skew():.4f}")
print(f"Kurtosis: {house_data['SalePrice'].kurtosis():.4f}")

# 5. Histogramme
plt.figure(figsize=(10, 6))
sns.histplot(house_data['SalePrice'], bins=30, kde=True, color='darkblue')
plt.xlabel('Prix de vente ($)')
plt.ylabel('Densité')
plt.title('Distribution des prix de vente')
plt.show()

# 6. Matrice de corrélation
corr = house_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1)
plt.title('Matrice de Corrélation')
plt.tight_layout()
plt.show()

# 7. Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=house_data, x='GrLivArea', y='SalePrice', alpha=0.6)
plt.xlabel('Surface habitable (sq ft)')
plt.ylabel('Prix de vente ($)')
plt.title('Prix en fonction de la surface')
plt.grid(True, alpha=0.3)
plt.show()

# 8. Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=house_data, x='OverallQual', y='SalePrice', palette='viridis')
plt.xlabel('Qualité globale (1-10)')
plt.ylabel('Prix de vente ($)')
plt.title('Prix par qualité globale')
plt.show()
```

---

## Résumé

### Points Clés à Retenir

#### 1. Pandas DataFrame
- Structure de données 2D pour analyse de données
- Opérations : `head()`, `info()`, `describe()`, `shape`, `columns`
- Sélection : `df["col"]`, `df.iloc[]`, `df.loc[]`, filtrage conditionnel
- Valeurs manquantes : `isna()`, `dropna()`, `fillna()`

#### 2. Mesures de Tendance Centrale
| Mesure | Formule | Propriété |
|--------|---------|-----------|
| **Moyenne** | $\bar{x} = \frac{1}{n}\sum x_i$ | Sensible aux outliers |
| **Médiane** | Valeur centrale | Robuste aux outliers |
| **Mode** | Valeur la plus fréquente | Applicable au catégoriel |

#### 3. Mesures de Dispersion
| Mesure | Formule | Interprétation |
|--------|---------|----------------|
| **Étendue** | max - min | Sensible aux extrêmes |
| **Variance** | $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$ | Unité au carré |
| **Écart-type** | $s = \sqrt{s^2}$ | Même unité que données |
| **IQR** | Q3 - Q1 | Robuste aux outliers |

#### 4. Mesures de Forme
- **Skewness** : Asymétrie
  - > 0 : Queue à droite
  - < 0 : Queue à gauche
  - ≈ 0 : Symétrique
- **Kurtosis** : Aplatissement
  - > 0 : Queues lourdes
  - < 0 : Queues légères
  - ≈ 0 : Normale

#### 5. Corrélation
- **Pearson** : $r \in [-1, 1]$
- Corrélation ≠ Causalité
- Matrice de corrélation : visualiser toutes les relations

#### 6. Visualisations Essentielles
| Type | Utilité |
|------|---------|
| **Histogramme** | Distribution d'une variable |
| **Boxplot** | Quartiles, médiane, outliers |
| **Scatter plot** | Relation entre 2 variables |
| **Heatmap** | Matrice de corrélation |
| **Pairplot** | Relations multiples |

### Bibliothèques Python

```python
import numpy as np                    # Calculs numériques
import pandas as pd                   # Manipulation de données
import matplotlib.pyplot as plt       # Visualisation de base
import seaborn as sns                 # Visualisation statistique
from scipy import stats               # Fonctions statistiques
```

### Checklist EDA

- [ ] Charger et inspecter les données (`head()`, `info()`, `shape`)
- [ ] Identifier types de variables (numériques, catégorielles)
- [ ] Vérifier valeurs manquantes (`isna().sum()`)
- [ ] Statistiques descriptives (`describe()`)
- [ ] Visualiser distributions (histogrammes, boxplots)
- [ ] Calculer skewness et kurtosis
- [ ] Analyser corrélations (matrice, heatmap)
- [ ] Détecter outliers (IQR, z-score)
- [ ] Identifier patterns et relations

### Formules Essentielles

```
Moyenne: x̄ = Σx / n
Variance: s² = Σ(x - x̄)² / (n-1)
Écart-type: s = √s²
Corrélation: r = Cov(X,Y) / (σₓ σᵧ)
IQR = Q₃ - Q₁
Outliers: < Q₁ - 1.5×IQR ou > Q₃ + 1.5×IQR
```

### Prochaine Étape

**Module 5 : Optimisation Numérique** - Gradient, descente de gradient, optimisation

---

**Navigation :**
- [⬅️ Module 3 : Probabilités](03_Probabilites.md)
- [🏠 Retour au Sommaire](README.md)
- [➡️ Module 5 : Optimisation Numérique](05_Optimisation_Numerique.md)
