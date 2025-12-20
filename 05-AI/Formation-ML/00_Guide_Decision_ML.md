# Guide de Décision : Quel Modèle ML pour Quel Problème ?

## 📋 Table des Matières
1. [Arbre de Décision Global](#arbre-de-décision-global)
2. [Classification](#classification)
3. [Régression](#régression)
4. [Clustering](#clustering)
5. [Réduction de Dimensionnalité](#réduction-de-dimensionnalité)
6. [Détection d'Anomalies](#détection-danomalies)
7. [Séries Temporelles](#séries-temporelles)
8. [Traitement d'Images](#traitement-dimages)
9. [Traitement du Langage Naturel (NLP)](#traitement-du-langage-naturel-nlp)
10. [Guide des Techniques d'Optimisation](#guide-des-techniques-doptimisation)
11. [Quand Utiliser Quoi ?](#quand-utiliser-quoi-)

---

## Arbre de Décision Global

```
┌─────────────────────────────────────────────────────────────┐
│              QUEL EST VOTRE TYPE DE DONNÉES ?               │
└─────────────────────────────────────────────────────────────┘
                              │
                  ┌───────────┼───────────┐
                  │           │           │
            ┌─────▼─────┐ ┌──▼───┐ ┌────▼────┐
            │ Tabulaires│ │Images│ │  Texte  │
            └─────┬─────┘ └──┬───┘ └────┬────┘
                  │          │           │
        ┌─────────┼─────┐    │           │
        │         │     │    │           │
    ┌───▼──┐ ┌───▼──┐ ┌▼────▼──┐   ┌───▼─────┐
    │Labels│ │Sans  │ │  CNN   │   │   NLP   │
    │      │ │Labels│ │ResNet  │   │Transform│
    └───┬──┘ └───┬──┘ │EfficNet│   │  BERT   │
        │        │    └────────┘   └─────────┘
   ┌────┼────┐   │
   │    │    │   │
 ┌─▼─┐┌─▼─┐┌─▼──▼──┐
 │Cls││Reg││Cluster│
 └───┘└───┘└───────┘
```

---

## Classification

### 📊 Arbre de Décision pour Classification

```
Vous avez un problème de CLASSIFICATION
│
├─ Combien de classes ?
│  ├─ 2 classes → Classification BINAIRE
│  └─ >2 classes → Classification MULTI-CLASSE
│
├─ Quelle est la taille de vos données ?
│  ├─ <1,000 samples
│  │  ├─ Linéaire → Logistic Regression, Naive Bayes
│  │  └─ Non-linéaire → Decision Tree, KNN
│  │
│  ├─ 1,000 - 100,000 samples
│  │  ├─ Interprétabilité requise → Logistic Regression, Decision Tree
│  │  ├─ Performance max → Random Forest, XGBoost, LightGBM
│  │  └─ Données texte → Naive Bayes, SVM
│  │
│  └─ >100,000 samples
│     ├─ Données tabulaires → XGBoost, LightGBM, CatBoost
│     ├─ Données images → CNN (ResNet, EfficientNet)
│     ├─ Données texte → Transformers (BERT, RoBERTa)
│     └─ Temps réel → Linear models, small trees
│
└─ Contraintes spécifiques ?
   ├─ Interprétabilité → Logistic Regression, Decision Tree, Linear SVM
   ├─ Temps d'inférence court → Linear models, small Decision Trees
   ├─ Classes déséquilibrées → XGBoost, Random Forest + class_weight
   └─ Peu de features → SVM (kernel), Neural Networks
```

### 📋 Tableau de Décision Classification

| Critère | Modèle Recommandé | Raison |
|---------|-------------------|--------|
| **Données linéairement séparables** | Logistic Regression, Linear SVM | Simple, rapide, interprétable |
| **Données non-linéaires, petites** | SVM (RBF kernel), Decision Tree | Capture non-linéarité, peu de données |
| **Données non-linéaires, moyennes** | Random Forest, XGBoost | Meilleure performance, robuste |
| **Données non-linéaires, grandes** | XGBoost, LightGBM, Neural Networks | Scalable, haute performance |
| **Classes déséquilibrées** | XGBoost + scale_pos_weight, Random Forest + class_weight | Gère déséquilibre nativement |
| **Interprétabilité critique** | Logistic Regression, Decision Tree | Coefficients/règles clairs |
| **Haute dimensionnalité** | Linear SVM, Logistic Regression + régularisation | Évite overfitting |
| **Données catégorielles** | CatBoost, LightGBM | Gestion native des catégories |
| **Temps d'entraînement court** | Naive Bayes, Logistic Regression | Très rapides |
| **Temps d'inférence court** | Linear models, small trees | Prédictions instantanées |

### 🔍 Détail des Modèles de Classification

#### 1. Logistic Regression

**Quand l'utiliser ?**
- ✅ Classes linéairement séparables
- ✅ Besoin d'interprétabilité (coefficients)
- ✅ Baseline rapide
- ✅ Probabilités calibrées nécessaires
- ✅ Haute dimensionnalité (avec régularisation)

**Quand ne PAS l'utiliser ?**
- ❌ Relations fortement non-linéaires
- ❌ Interactions complexes entre features
- ❌ Besoin de performance maximale sur données complexes

**Exemple d'usage :**
```python
from sklearn.linear_model import LogisticRegression

# Standard
model = LogisticRegression(max_iter=1000)

# Avec régularisation L1 (sélection de features)
model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)

# Avec régularisation L2 (Ridge)
model = LogisticRegression(penalty='l2', C=1.0)

# Classes déséquilibrées
model = LogisticRegression(class_weight='balanced')
```

#### 2. Decision Tree

**Quand l'utiliser ?**
- ✅ Besoin d'interprétabilité visuelle
- ✅ Relations non-linéaires
- ✅ Pas besoin de normalisation
- ✅ Gère valeurs manquantes naturellement
- ✅ Variables catégorielles et continues mélangées

**Quand ne PAS l'utiliser ?**
- ❌ Données bruitées (overfitting facile)
- ❌ Besoin de performance maximale
- ❌ Extrapolation nécessaire

**Exemple d'usage :**
```python
from sklearn.tree import DecisionTreeClassifier

# Standard
model = DecisionTreeClassifier(random_state=42)

# Limiter la profondeur pour éviter overfitting
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
```

#### 3. Random Forest

**Quand l'utiliser ?**
- ✅ Données non-linéaires
- ✅ Peu de preprocessing nécessaire
- ✅ Importance des features requise
- ✅ Robustesse aux outliers
- ✅ Taille moyenne à grande

**Quand ne PAS l'utiliser ?**
- ❌ Très grandes données (préférer LightGBM)
- ❌ Temps d'inférence critique
- ❌ Mémoire limitée
- ❌ Interprétabilité au niveau individuel requise

**Exemple d'usage :**
```python
from sklearn.ensemble import RandomForestClassifier

# Standard
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# Classes déséquilibrées
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

# Performance optimale
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
```

#### 4. XGBoost / LightGBM / CatBoost

**Quand l'utiliser ?**
- ✅ Performance maximale requise
- ✅ Données tabulaires
- ✅ Compétitions Kaggle
- ✅ Grandes données
- ✅ Tuning d'hyperparamètres possible

**Différences :**
- **XGBoost** : Standard, très performant, bien documenté
- **LightGBM** : Plus rapide, gère grandes données, économe en mémoire
- **CatBoost** : Gère catégories nativement, peu de tuning nécessaire

**Exemple d'usage :**
```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# XGBoost
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# LightGBM (plus rapide)
lgbm = LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# CatBoost (gère catégories)
catboost = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=0
)
```

#### 5. Support Vector Machine (SVM)

**Quand l'utiliser ?**
- ✅ Haute dimensionnalité (features >> samples)
- ✅ Données non-linéaires (kernel RBF)
- ✅ Classes bien séparées
- ✅ Petites/moyennes données

**Quand ne PAS l'utiliser ?**
- ❌ Très grandes données (lent, O(n²))
- ❌ Classes déséquilibrées (nécessite tuning)
- ❌ Besoin de probabilités calibrées

**Exemple d'usage :**
```python
from sklearn.svm import SVC

# Linéaire (haute dimensionnalité)
model = SVC(kernel='linear', C=1.0)

# RBF (non-linéaire)
model = SVC(kernel='rbf', C=1.0, gamma='scale')

# Avec probabilités
model = SVC(kernel='rbf', probability=True)

# Classes déséquilibrées
model = SVC(kernel='rbf', class_weight='balanced')
```

#### 6. Naive Bayes

**Quand l'utiliser ?**
- ✅ Données texte (NLP)
- ✅ Besoin de rapidité
- ✅ Baseline simple
- ✅ Features indépendantes
- ✅ Streaming/online learning

**Quand ne PAS l'utiliser ?**
- ❌ Features corrélées
- ❌ Besoin de performance maximale

**Exemple d'usage :**
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Gaussian (features continues, distribution normale)
model = GaussianNB()

# Multinomial (comptages, ex: TF-IDF)
model = MultinomialNB(alpha=1.0)

# Bernoulli (features binaires)
model = BernoulliNB(alpha=1.0)
```

#### 7. K-Nearest Neighbors (KNN)

**Quand l'utiliser ?**
- ✅ Patterns locaux importants
- ✅ Données peu volumineuses
- ✅ Pas de phase d'entraînement nécessaire
- ✅ Données non-linéaires

**Quand ne PAS l'utiliser ?**
- ❌ Grandes données (lent à prédire)
- ❌ Haute dimensionnalité (curse of dimensionality)
- ❌ Features de différentes échelles (nécessite normalisation)

**Exemple d'usage :**
```python
from sklearn.neighbors import KNeighborsClassifier

# Standard
model = KNeighborsClassifier(n_neighbors=5)

# Pondération par distance
model = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Métrique personnalisée
model = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
```

### 🎯 Stratégie de Sélection Rapide

**Workflow recommandé :**

1. **Baseline rapide** : Logistic Regression
2. **Amélioration** : Random Forest ou XGBoost
3. **Optimisation** : Tuning d'hyperparamètres du meilleur modèle
4. **Si insatisfait** : Tester SVM, Neural Networks

---

## Régression

### 📊 Arbre de Décision pour Régression

```
Vous avez un problème de RÉGRESSION
│
├─ La relation est-elle linéaire ?
│  │
│  ├─ OUI → Modèles Linéaires
│  │  ├─ Peu de features → Linear Regression
│  │  ├─ Beaucoup de features → Ridge, Lasso
│  │  └─ Sélection de features → Lasso, ElasticNet
│  │
│  └─ NON → Modèles Non-Linéaires
│     ├─ <10k samples → SVR, Decision Tree
│     ├─ 10k-100k samples → Random Forest, XGBoost
│     └─ >100k samples → XGBoost, LightGBM, Neural Networks
│
├─ Y a-t-il des outliers ?
│  ├─ OUI → Ridge, Random Forest, Huber Regression
│  └─ NON → Tous modèles OK
│
└─ Contraintes ?
   ├─ Interprétabilité → Linear Regression, Decision Tree
   ├─ Régularisation → Ridge (L2), Lasso (L1), ElasticNet
   └─ Performance max → XGBoost, LightGBM
```

### 📋 Tableau de Décision Régression

| Critère | Modèle Recommandé | Raison |
|---------|-------------------|--------|
| **Relation linéaire** | Linear Regression | Simple, interprétable |
| **Relation linéaire + multicollinéarité** | Ridge Regression | Stabilise les coefficients |
| **Relation linéaire + beaucoup de features** | Lasso, ElasticNet | Sélection de features |
| **Relation non-linéaire, petites données** | SVR, Decision Tree | Capture non-linéarité |
| **Relation non-linéaire, moyennes données** | Random Forest, XGBoost | Performance, robustesse |
| **Relation non-linéaire, grandes données** | XGBoost, LightGBM | Scalable, performant |
| **Outliers présents** | Huber Regression, Random Forest | Robuste aux outliers |
| **Interprétabilité** | Linear Regression, Decision Tree | Coefficients/règles clairs |

### 🔍 Détail des Modèles de Régression

#### 1. Linear Regression

**Quand l'utiliser ?**
- ✅ Relation linéaire claire
- ✅ Besoin d'interprétabilité
- ✅ Baseline rapide
- ✅ Peu de features
- ✅ Pas de multicollinéarité

**Quand ne PAS l'utiliser ?**
- ❌ Relation non-linéaire
- ❌ Multicollinéarité forte
- ❌ Beaucoup de features inutiles

**Exemple d'usage :**
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Interpréter les coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
```

#### 2. Ridge Regression (L2)

**Quand l'utiliser ?**
- ✅ Multicollinéarité présente
- ✅ Beaucoup de features
- ✅ Prévention de l'overfitting
- ✅ Garder toutes les features

**Paramètre clé :** `alpha` (force de régularisation)
- Petit alpha → proche Linear Regression
- Grand alpha → coefficients plus petits

**Exemple d'usage :**
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import RidgeCV

# Avec alpha fixe
model = Ridge(alpha=1.0)

# Avec cross-validation pour choisir alpha
model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
model.fit(X_train, y_train)
print(f"Best alpha: {model.alpha_}")
```

#### 3. Lasso Regression (L1)

**Quand l'utiliser ?**
- ✅ Beaucoup de features inutiles
- ✅ Sélection automatique de features
- ✅ Features parcimonieuses souhaitées
- ✅ Interprétabilité avec peu de features

**Avantage :** Met certains coefficients à 0 (sélection de features)

**Exemple d'usage :**
```python
from sklearn.linear_model import Lasso, LassoCV

# Avec alpha fixe
model = Lasso(alpha=0.1)

# Avec cross-validation
model = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5)
model.fit(X_train, y_train)

# Features sélectionnées
selected_features = X_train.columns[model.coef_ != 0]
print(f"Features sélectionnées: {len(selected_features)}/{len(X_train.columns)}")
```

#### 4. ElasticNet (L1 + L2)

**Quand l'utiliser ?**
- ✅ Compromis entre Ridge et Lasso
- ✅ Beaucoup de features corrélées
- ✅ Sélection de groupes de features corrélées

**Paramètres :**
- `alpha` : Force de régularisation
- `l1_ratio` : Mix L1/L2 (0=Ridge, 1=Lasso, 0.5=équilibre)

**Exemple d'usage :**
```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

model = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
    alphas=[0.001, 0.01, 0.1, 1.0],
    cv=5
)
model.fit(X_train, y_train)
print(f"Best l1_ratio: {model.l1_ratio_}, Best alpha: {model.alpha_}")
```

#### 5. Decision Tree / Random Forest / XGBoost

**Quand l'utiliser ?** (même logique que classification)
- ✅ Relation non-linéaire
- ✅ Interactions complexes
- ✅ Pas besoin de normalisation
- ✅ Performance maximale (XGBoost)

**Exemple d'usage :**
```python
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Random Forest
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

# XGBoost
xgb = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
```

### 🎯 Stratégie de Sélection Rapide

1. **Baseline** : Linear Regression
2. **Si multicollinéarité** : Ridge
3. **Si beaucoup de features** : Lasso ou ElasticNet
4. **Si non-linéaire** : XGBoost ou Random Forest

---

## Clustering

### 📊 Arbre de Décision pour Clustering

```
Vous voulez faire du CLUSTERING
│
├─ Connaissez-vous le nombre de clusters K ?
│  │
│  ├─ OUI
│  │  ├─ Clusters sphériques ? → K-Means
│  │  └─ Formes arbitraires ? → Spectral Clustering
│  │
│  └─ NON
│     ├─ Densité variable ? → DBSCAN, HDBSCAN
│     ├─ Hiérarchie ? → Hierarchical Clustering
│     └─ Automatique ? → HDBSCAN, Gaussian Mixture
│
├─ Quelle est la taille des données ?
│  ├─ <10k → Tous modèles OK
│  ├─ 10k-100k → K-Means, DBSCAN
│  └─ >100k → K-Means, Mini-Batch K-Means
│
└─ Contraintes ?
   ├─ Vitesse → K-Means, Mini-Batch K-Means
   ├─ Outliers importants → DBSCAN
   └─ Clusters imbriqués → Hierarchical
```

### 📋 Comparaison des Algorithmes de Clustering

| Algorithme | Avantages | Inconvénients | Usage |
|------------|-----------|---------------|-------|
| **K-Means** | Rapide, scalable | K fixe, clusters sphériques | Grandes données, clusters bien définis |
| **DBSCAN** | Détecte outliers, K automatique | Sensible à eps et min_samples | Densité variable, outliers |
| **HDBSCAN** | DBSCAN + hiérarchie | Lent sur grandes données | Meilleure alternative à DBSCAN |
| **Hierarchical** | Dendrogramme, K flexible | Lent O(n²), pas scalable | Petites données, hiérarchie |
| **Gaussian Mixture** | Clusters probabilistes | K fixe, suppose Gaussiennes | Clusters ellipsoïdes, incertitude |
| **Spectral** | Formes complexes | Lent, K fixe | Clusters non-convexes |

### 🔍 Détail des Algorithmes

#### 1. K-Means

**Quand l'utiliser ?**
- ✅ Clusters sphériques et de taille similaire
- ✅ Grandes données
- ✅ K connu ou estimable
- ✅ Besoin de rapidité

**Quand ne PAS l'utiliser ?**
- ❌ Clusters de formes arbitraires
- ❌ Clusters de densités différentes
- ❌ Outliers nombreux

**Exemple d'usage :**
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Méthode du coude pour choisir K
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inertie')
plt.title('Méthode du Coude')
plt.show()

# K-Means final
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
```

#### 2. DBSCAN

**Quand l'utiliser ?**
- ✅ Clusters de formes arbitraires
- ✅ Outliers à détecter
- ✅ K inconnu
- ✅ Densité variable

**Paramètres critiques :**
- `eps` : Rayon du voisinage
- `min_samples` : Nombre minimum de points pour un cluster

**Exemple d'usage :**
```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Trouver eps optimal avec k-distance graph
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X)
distances, indices = nn.kneighbors(X)
distances = np.sort(distances[:, -1])

plt.plot(distances)
plt.ylabel('5-NN distance')
plt.xlabel('Points')
plt.title('K-distance Graph (chercher le coude)')
plt.show()

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers = list(labels).count(-1)
print(f"Clusters: {n_clusters}, Outliers: {n_outliers}")
```

#### 3. Hierarchical Clustering

**Quand l'utiliser ?**
- ✅ Hiérarchie de clusters importante
- ✅ Petites données (<10k)
- ✅ Dendrogramme souhaité
- ✅ K flexible après coup

**Méthodes de linkage :**
- `ward` : Minimise variance (recommandé)
- `average` : Distance moyenne
- `complete` : Distance maximale
- `single` : Distance minimale

**Exemple d'usage :**
```python
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Créer dendrogramme
linkage_matrix = linkage(X, method='ward')

plt.figure(figsize=(12, 5))
dendrogram(linkage_matrix)
plt.title('Dendrogramme')
plt.show()

# Clustering agglomératif
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)
```

---

## Réduction de Dimensionnalité

### 📊 Guide de Décision

```
Objectif de RÉDUCTION DE DIMENSIONNALITÉ
│
├─ Quel est l'objectif ?
│  │
│  ├─ Visualisation (2D/3D)
│  │  ├─ Structure globale → PCA
│  │  ├─ Structure locale → t-SNE
│  │  └─ Structure globale+locale → UMAP
│  │
│  ├─ Compression de données
│  │  ├─ Linéaire → PCA
│  │  ├─ Non-linéaire → Autoencoder
│  │  └─ Données images → CNN Autoencoder
│  │
│  ├─ Preprocessing avant ML
│  │  ├─ Linéaire → PCA
│  │  ├─ Sélection de features → Feature Selection (Lasso, RFE)
│  │  └─ Non-linéaire → Kernel PCA
│  │
│  └─ Génération de données
│     └─ Variational Autoencoder (VAE)
│
└─ Taille des données ?
   ├─ <10k → Tous algorithmes OK
   ├─ 10k-100k → PCA, UMAP, Autoencoder
   └─ >100k → PCA, UMAP, Autoencoder
```

### 📋 Comparaison des Techniques

| Technique | Type | Usage Principal | Préserve | Vitesse |
|-----------|------|-----------------|----------|---------|
| **PCA** | Linéaire | Compression, preprocessing | Variance globale | Très rapide |
| **t-SNE** | Non-linéaire | Visualisation | Structure locale | Lent |
| **UMAP** | Non-linéaire | Visualisation, général | Structure locale+globale | Rapide |
| **Autoencoder** | Non-linéaire | Compression, génération | Features apprises | Moyen |
| **Kernel PCA** | Non-linéaire | Preprocessing | Variance (kernel space) | Lent |

### 🔍 Quand Utiliser Quoi ?

#### PCA (Principal Component Analysis)

**Utiliser POUR :**
- ✅ Compression de données
- ✅ Réduction de bruit
- ✅ Visualisation rapide
- ✅ Preprocessing avant ML
- ✅ Données avec variance linéaire

**NE PAS utiliser pour :**
- ❌ Données avec structure non-linéaire complexe
- ❌ Visualisation fine de clusters
- ❌ Interprétation sémantique des composantes

```python
from sklearn.decomposition import PCA

# PCA standard
pca = PCA(n_components=0.95)  # 95% de variance
X_reduced = pca.fit_transform(X)

print(f"Dimensions: {X.shape} → {X_reduced.shape}")
print(f"Variance expliquée: {pca.explained_variance_ratio_.sum():.2%}")
```

#### t-SNE

**Utiliser POUR :**
- ✅ Visualisation 2D/3D de clusters
- ✅ Exploration de données
- ✅ Structure locale importante

**NE PAS utiliser pour :**
- ❌ Preprocessing pour ML (non déterministe)
- ❌ Nouvelles données (pas de transform)
- ❌ Grandes données (>50k, très lent)
- ❌ Interprétation des distances globales

```python
from sklearn.manifold import TSNE

# t-SNE pour visualisation
tsne = TSNE(
    n_components=2,
    perplexity=30,  # 5-50 typiquement
    n_iter=1000,
    random_state=42
)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
plt.title('t-SNE Visualization')
plt.show()
```

#### UMAP

**Utiliser POUR :**
- ✅ Visualisation (alternative à t-SNE)
- ✅ Preprocessing pour ML
- ✅ Structure globale ET locale
- ✅ Plus rapide que t-SNE
- ✅ Peut transformer nouvelles données

**NE PAS utiliser pour :**
- ❌ Besoin d'interprétabilité des axes

```python
import umap

# UMAP
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)
X_umap = reducer.fit_transform(X)

# Peut transformer nouvelles données
X_new_umap = reducer.transform(X_new)
```

---

## Guide des Techniques d'Optimisation

### À Quoi Sert la Descente de Gradient ?

**Définition :** Algorithme d'optimisation pour minimiser une fonction de coût.

**Principe :**
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

où :
- $\theta$ = paramètres du modèle
- $\eta$ = learning rate
- $\nabla_\theta \mathcal{L}$ = gradient de la fonction de coût

### Quand Utiliser la Descente de Gradient ?

#### Modèles qui l'utilisent :

| Modèle | Utilise Gradient Descent ? | Algorithme |
|--------|----------------------------|------------|
| **Linear Regression** | Non (solution analytique) | Normal Equation |
| **Logistic Regression** | Oui | Gradient Descent, LBFGS, Newton |
| **Neural Networks** | Oui | SGD, Adam, RMSprop |
| **SVM** | Oui (si SGD) | SGD, SMO |
| **XGBoost** | Non | Gradient Boosting (différent) |
| **Decision Tree** | Non | Algorithme glouton |

### Types de Descente de Gradient

```python
# 1. Batch Gradient Descent (tout le dataset)
for epoch in range(n_epochs):
    gradients = compute_gradients(X, y, theta)
    theta = theta - learning_rate * gradients

# 2. Stochastic Gradient Descent (1 sample à la fois)
for epoch in range(n_epochs):
    for i in range(n_samples):
        gradients = compute_gradients(X[i], y[i], theta)
        theta = theta - learning_rate * gradients

# 3. Mini-Batch Gradient Descent (batch de samples)
for epoch in range(n_epochs):
    for batch in get_batches(X, y, batch_size):
        gradients = compute_gradients(batch_X, batch_y, theta)
        theta = theta - learning_rate * gradients
```

### Optimiseurs pour Neural Networks

```python
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad

# 1. SGD (simple)
optimizer = SGD(learning_rate=0.01)

# 2. SGD with Momentum
optimizer = SGD(learning_rate=0.01, momentum=0.9)

# 3. Adam (recommandé par défaut)
optimizer = Adam(learning_rate=0.001)

# 4. RMSprop
optimizer = RMSprop(learning_rate=0.001)
```

**Guide de choix :**
- **SGD** : Baseline simple
- **SGD + Momentum** : Accélère la convergence
- **Adam** : Recommandé par défaut (adaptatif)
- **RMSprop** : Bon pour RNN

---

## Quand Utiliser Quoi ?

### Tableau Récapitulatif Global

| Situation | Technique Recommandée | Raison |
|-----------|----------------------|--------|
| **Baseline rapide** | Logistic Regression / Linear Regression | Simple, rapide, interprétable |
| **Performance maximale sur tabulaire** | XGBoost / LightGBM | État de l'art pour données tabulaires |
| **Images** | CNN (ResNet, EfficientNet) | Spécialisé pour images |
| **Texte** | Transformers (BERT) / RNN | Spécialisé pour NLP |
| **Séries temporelles** | LSTM, GRU, Prophet | Capture dépendances temporelles |
| **Interprétabilité requise** | Logistic Regression, Decision Tree | Coefficients/règles clairs |
| **Peu de données** | Linear models, Decision Tree, SVM | Évite overfitting |
| **Beaucoup de données** | XGBoost, LightGBM, Deep Learning | Scalable, performant |
| **Temps d'inférence critique** | Linear models, small trees | Prédictions instantanées |
| **Classes déséquilibrées** | XGBoost + class_weight, SMOTE | Gère déséquilibre |
| **Clustering avec K inconnu** | DBSCAN, HDBSCAN | K automatique |
| **Visualisation** | t-SNE, UMAP | Excellente visualisation 2D/3D |
| **Compression** | PCA, Autoencoder | Réduit dimensionnalité |
| **Détection d'anomalies** | Isolation Forest, Autoencoder | Spécialisé pour outliers |

---

**🎯 Ce guide vous aide à choisir le bon modèle dans toutes les situations !**

---

**Navigation :**
- [⬅️ Guide Projet ML](00_Guide_Projet_ML.md)
- [➡️ Workflows ML](00_Workflows_ML.md)
- [🏠 Retour au Sommaire](README.md)
