# Guide Complet : Démarrer un Projet Machine Learning

## 📋 Table des Matières
1. [Checklist Complète d'un Projet ML](#checklist-complète-dun-projet-ml)
2. [Phase 1 : Compréhension du Problème](#phase-1--compréhension-du-problème)
3. [Phase 2 : Collecte et Exploration des Données](#phase-2--collecte-et-exploration-des-données)
4. [Phase 3 : Préparation des Données](#phase-3--préparation-des-données)
5. [Phase 4 : Modélisation](#phase-4--modélisation)
6. [Phase 5 : Évaluation](#phase-5--évaluation)
7. [Phase 6 : Déploiement](#phase-6--déploiement)
8. [Questions Critiques à se Poser](#questions-critiques-à-se-poser)
9. [Templates de Documentation](#templates-de-documentation)

---

## Checklist Complète d'un Projet ML

### ✅ Phase 1 : Compréhension du Problème
- [ ] Définir la problématique métier clairement
- [ ] Identifier les objectifs mesurables
- [ ] Déterminer le type de problème ML
- [ ] Évaluer la faisabilité du projet
- [ ] Définir les critères de succès
- [ ] Identifier les contraintes (temps, budget, ressources)
- [ ] Comprendre l'impact business

### ✅ Phase 2 : Collecte et Exploration des Données
- [ ] Identifier les sources de données disponibles
- [ ] Collecter les données nécessaires
- [ ] Vérifier la qualité des données
- [ ] Analyser les statistiques descriptives
- [ ] Visualiser les distributions
- [ ] Identifier les valeurs manquantes
- [ ] Détecter les outliers
- [ ] Analyser les corrélations entre variables

### ✅ Phase 3 : Préparation des Données
- [ ] Traiter les valeurs manquantes
- [ ] Gérer les outliers
- [ ] Encoder les variables catégorielles
- [ ] Normaliser/standardiser les variables numériques
- [ ] Créer de nouvelles features (feature engineering)
- [ ] Sélectionner les features pertinentes
- [ ] Diviser les données (train/validation/test)
- [ ] Gérer le déséquilibre des classes (si nécessaire)

### ✅ Phase 4 : Modélisation
- [ ] Choisir les modèles candidats
- [ ] Définir la baseline
- [ ] Entraîner les modèles
- [ ] Optimiser les hyperparamètres
- [ ] Valider avec cross-validation
- [ ] Comparer les performances
- [ ] Sélectionner le meilleur modèle
- [ ] Analyser les erreurs

### ✅ Phase 5 : Évaluation
- [ ] Évaluer sur le test set
- [ ] Calculer les métriques appropriées
- [ ] Analyser la matrice de confusion (classification)
- [ ] Vérifier l'overfitting/underfitting
- [ ] Tester sur cas limites
- [ ] Interpréter les prédictions
- [ ] Documenter les résultats

### ✅ Phase 6 : Déploiement
- [ ] Préparer le modèle pour production
- [ ] Créer une API ou interface
- [ ] Mettre en place le monitoring
- [ ] Tester en environnement réel
- [ ] Former les utilisateurs
- [ ] Planifier la maintenance
- [ ] Prévoir le retraining

---

## Phase 1 : Compréhension du Problème

### Questions Essentielles

#### 1. Quelle est la problématique métier ?

**Template de problématique :**
```
Contexte : [Décrire la situation actuelle]
Problème : [Quel problème cherche-t-on à résoudre ?]
Impact : [Quelles sont les conséquences du problème ?]
Solution envisagée : [Comment le ML peut-il aider ?]
```

**Exemple :**
```
Contexte : Une banque reçoit des milliers de demandes de crédit par jour
Problème : Le processus d'évaluation manuel est lent et coûteux
Impact : Perte de clients, coûts opérationnels élevés
Solution envisagée : Système automatisé de prédiction de défaut de paiement
```

#### 2. Quel est le type de problème ML ?

| Type | Description | Exemples |
|------|-------------|----------|
| **Classification binaire** | 2 classes | Spam/Non-spam, Fraude/Légitime |
| **Classification multi-classe** | >2 classes | Reconnaissance de chiffres, Catégorisation de produits |
| **Régression** | Prédiction valeur continue | Prix immobilier, Température |
| **Clustering** | Groupement sans labels | Segmentation client |
| **Détection d'anomalies** | Identifier les outliers | Fraude, Défauts industriels |
| **Séries temporelles** | Prédiction temporelle | Prévision des ventes, Prix boursiers |
| **NLP** | Traitement du langage | Analyse de sentiment, Traduction |
| **Vision** | Traitement d'images | Détection d'objets, Classification d'images |

#### 3. Quels sont les objectifs mesurables ?

**Template d'objectifs :**
```
Objectif principal : [Métrique cible]
  - Actuel : [Valeur baseline]
  - Cible : [Valeur à atteindre]
  - Délai : [Quand ?]

Objectifs secondaires :
  - [Autre métrique 1]
  - [Autre métrique 2]
```

**Exemple :**
```
Objectif principal : Réduire le taux de défaut de paiement
  - Actuel : 15% des crédits accordés
  - Cible : <8% des crédits accordés
  - Délai : 6 mois

Objectifs secondaires :
  - Réduire le temps de traitement de 5 jours à 1 heure
  - Maintenir un taux d'approbation >70%
```

#### 4. Quelles sont les contraintes ?

**Contraintes à identifier :**

| Type | Questions |
|------|-----------|
| **Temps** | Quelle est la deadline ? Temps d'inférence acceptable ? |
| **Budget** | Ressources de calcul disponibles ? Budget cloud ? |
| **Données** | Quantité de données disponibles ? Qualité ? Labels ? |
| **Interprétabilité** | Le modèle doit-il être explicable ? (médical, finance) |
| **Précision** | Quelle précision minimale ? Quel type d'erreur est acceptable ? |
| **Déploiement** | Edge device ? Cloud ? On-premise ? |
| **Légal** | RGPD ? Autres réglementations ? |

#### 5. Définir les critères de succès

**Critères techniques :**
- Métriques de performance (accuracy, F1, RMSE, etc.)
- Temps d'inférence
- Taille du modèle
- Robustesse

**Critères business :**
- ROI attendu
- Réduction des coûts
- Amélioration de l'expérience utilisateur
- Gain de productivité

---

## Phase 2 : Collecte et Exploration des Données

### Questions sur les Données

#### 1. Quelles données sont disponibles ?

**Checklist des données :**
```python
# Template d'inventaire des données
donnees_disponibles = {
    'source_1': {
        'type': 'Base de données SQL',
        'volume': '1M lignes',
        'periode': '2020-2024',
        'format': 'Structuré',
        'qualite': 'Bonne',
        'acces': 'API',
        'cout': 'Gratuit'
    },
    'source_2': {
        'type': 'Fichiers CSV',
        'volume': '500K lignes',
        'periode': '2022-2024',
        'format': 'Semi-structuré',
        'qualite': 'Moyenne (valeurs manquantes)',
        'acces': 'FTP',
        'cout': 'Gratuit'
    }
}
```

#### 2. Quel est le type de données ?

| Type | Exemples | Préparation |
|------|----------|-------------|
| **Numériques continues** | Prix, température, âge | Normalisation, standardisation |
| **Numériques discrètes** | Nombre de produits, compteurs | Binning possible |
| **Catégorielles ordinales** | Niveau d'éducation, taille (S/M/L) | Ordinal encoding |
| **Catégorielles nominales** | Couleur, ville, catégorie | One-hot encoding, target encoding |
| **Temporelles** | Date, heure, timestamp | Feature engineering (jour, mois, etc.) |
| **Texte** | Avis, descriptions | TF-IDF, embeddings |
| **Images** | Photos, scans | Normalisation, augmentation |
| **Audio** | Voix, sons | Spectrogrammes, MFCC |

#### 3. Analyse Exploratoire des Données (EDA)

**Script EDA Standard :**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyse_exploratoire(df):
    """
    Analyse exploratoire complète d'un DataFrame
    """
    print("="*80)
    print("1. INFORMATIONS GÉNÉRALES")
    print("="*80)
    print(f"Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"\nMémoire utilisée : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\n" + "="*80)
    print("2. TYPES DE DONNÉES")
    print("="*80)
    print(df.dtypes.value_counts())

    print("\n" + "="*80)
    print("3. VALEURS MANQUANTES")
    print("="*80)
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_table = pd.DataFrame({
        'Manquantes': missing,
        'Pourcentage': missing_pct
    })
    print(missing_table[missing_table['Manquantes'] > 0].sort_values('Pourcentage', ascending=False))

    print("\n" + "="*80)
    print("4. STATISTIQUES DESCRIPTIVES")
    print("="*80)
    print(df.describe())

    print("\n" + "="*80)
    print("5. DOUBLONS")
    print("="*80)
    duplicates = df.duplicated().sum()
    print(f"Nombre de doublons : {duplicates} ({100*duplicates/len(df):.2f}%)")

    print("\n" + "="*80)
    print("6. CARDINALITÉ DES VARIABLES CATÉGORIELLES")
    print("="*80)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        print(f"{col}: {df[col].nunique()} valeurs uniques")
        if df[col].nunique() <= 10:
            print(df[col].value_counts())
        print()

    # Visualisations
    print("\n" + "="*80)
    print("7. VISUALISATIONS")
    print("="*80)

    # Distribution des variables numériques
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        fig, axes = plt.subplots(len(num_cols), 2, figsize=(15, 5*len(num_cols)))
        if len(num_cols) == 1:
            axes = axes.reshape(1, -1)

        for idx, col in enumerate(num_cols):
            # Histogramme
            axes[idx, 0].hist(df[col].dropna(), bins=30, edgecolor='black')
            axes[idx, 0].set_title(f'Distribution de {col}')
            axes[idx, 0].set_xlabel(col)
            axes[idx, 0].set_ylabel('Fréquence')

            # Boxplot
            axes[idx, 1].boxplot(df[col].dropna())
            axes[idx, 1].set_title(f'Boxplot de {col}')
            axes[idx, 1].set_ylabel(col)

        plt.tight_layout()
        plt.show()

    # Matrice de corrélation
    if len(num_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation = df[num_cols].corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                    square=True, linewidths=0.5)
        plt.title('Matrice de Corrélation')
        plt.tight_layout()
        plt.show()

    return {
        'shape': df.shape,
        'missing': missing_table[missing_table['Manquantes'] > 0],
        'duplicates': duplicates,
        'dtypes': df.dtypes
    }

# Utilisation
# resultats = analyse_exploratoire(df)
```

#### 4. Questions sur la Qualité des Données

**Checklist qualité :**

- [ ] **Complétude** : Taux de valeurs manquantes acceptable ?
- [ ] **Cohérence** : Les valeurs sont-elles cohérentes ? (ex: âge négatif)
- [ ] **Précision** : Les données sont-elles exactes ?
- [ ] **Actualité** : Les données sont-elles à jour ?
- [ ] **Unicité** : Y a-t-il des doublons ?
- [ ] **Représentativité** : Les données reflètent-elles la population cible ?
- [ ] **Équilibre** : Les classes sont-elles équilibrées (classification) ?

---

## Phase 3 : Préparation des Données

### 1. Traitement des Valeurs Manquantes

#### Stratégies selon le contexte

```python
# Décision : Comment traiter les valeurs manquantes ?

def strategie_valeurs_manquantes(df, col):
    """
    Guide de décision pour valeurs manquantes
    """
    missing_pct = df[col].isnull().sum() / len(df) * 100

    print(f"Colonne : {col}")
    print(f"Valeurs manquantes : {missing_pct:.2f}%")

    if missing_pct > 50:
        print("→ RECOMMANDATION : Supprimer la colonne (trop de valeurs manquantes)")
    elif missing_pct > 20:
        print("→ RECOMMANDATION : Imputation avancée ou créer feature 'is_missing'")
    else:
        if df[col].dtype in ['int64', 'float64']:
            print("→ OPTIONS :")
            print("  - Imputation par la moyenne (si distribution normale)")
            print("  - Imputation par la médiane (si outliers)")
            print("  - Imputation par régression/KNN (si corrélé à autres features)")
        else:
            print("→ OPTIONS :")
            print("  - Imputation par le mode")
            print("  - Imputation par 'Unknown' / 'Missing'")
            print("  - Imputation par modèle (classification)")
    print()

# Méthodes d'imputation
from sklearn.impute import SimpleImputer, KNNImputer

# 1. Imputation simple
imputer_mean = SimpleImputer(strategy='mean')  # moyenne
imputer_median = SimpleImputer(strategy='median')  # médiane
imputer_mode = SimpleImputer(strategy='most_frequent')  # mode

# 2. Imputation KNN
imputer_knn = KNNImputer(n_neighbors=5)

# 3. Imputation par régression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer_iter = IterativeImputer(max_iter=10, random_state=42)
```

### 2. Gestion des Outliers

#### Détection

```python
def detecter_outliers(df, col):
    """
    Détecte les outliers par plusieurs méthodes
    """
    print(f"Analyse des outliers pour : {col}")
    print("="*60)

    # Méthode 1 : IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_iqr = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Méthode IQR : {len(outliers_iqr)} outliers ({len(outliers_iqr)/len(df)*100:.2f}%)")
    print(f"  Bornes : [{lower_bound:.2f}, {upper_bound:.2f}]")

    # Méthode 2 : Z-score
    from scipy import stats
    z_scores = np.abs(stats.zscore(df[col].dropna()))
    outliers_z = df[np.abs(z_scores) > 3]
    print(f"Méthode Z-score (>3) : {len(outliers_z)} outliers ({len(outliers_z)/len(df)*100:.2f}%)")

    # Méthode 3 : Isolation Forest
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination=0.1, random_state=42)
    outliers_iso = iso.fit_predict(df[[col]])
    n_outliers_iso = (outliers_iso == -1).sum()
    print(f"Méthode Isolation Forest : {n_outliers_iso} outliers ({n_outliers_iso/len(df)*100:.2f}%)")

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot
    axes[0].boxplot(df[col].dropna())
    axes[0].set_title(f'Boxplot - {col}')
    axes[0].axhline(lower_bound, color='r', linestyle='--', label='Borne IQR inf')
    axes[0].axhline(upper_bound, color='r', linestyle='--', label='Borne IQR sup')
    axes[0].legend()

    # Distribution
    axes[1].hist(df[col].dropna(), bins=50, edgecolor='black')
    axes[1].axvline(lower_bound, color='r', linestyle='--')
    axes[1].axvline(upper_bound, color='r', linestyle='--')
    axes[1].set_title(f'Distribution - {col}')
    axes[1].set_xlabel(col)

    plt.tight_layout()
    plt.show()

    return {
        'iqr': outliers_iqr.index,
        'z_score': outliers_z.index,
        'isolation_forest': df.index[outliers_iso == -1]
    }

# Stratégies de traitement
def traiter_outliers(df, col, method='cap'):
    """
    Traite les outliers selon la méthode choisie

    Methods:
    - 'remove': Supprimer les outliers
    - 'cap': Capping (remplacer par les bornes)
    - 'log': Transformation logarithmique
    - 'winsorize': Winsorisation
    """
    if method == 'remove':
        # IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    elif method == 'cap':
        # Capping
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)

    elif method == 'log':
        # Log transformation
        df[col] = np.log1p(df[col])

    elif method == 'winsorize':
        # Winsorisation
        from scipy.stats.mstats import winsorize
        df[col] = winsorize(df[col], limits=[0.05, 0.05])

    return df
```

### 3. Feature Engineering

#### Questions sur les Features

```python
# Guide de Feature Engineering

"""
QUESTIONS À SE POSER :

1. Combinaisons de features
   - Peut-on créer des ratios ? (ex: prix/m²)
   - Peut-on créer des différences ? (ex: âge_max - âge_min)
   - Peut-on créer des produits ? (ex: longueur × largeur)

2. Extraction d'information
   - Dates : jour, mois, année, jour de semaine, trimestre, est_weekend
   - Texte : longueur, nombre de mots, sentiment, entités
   - Catégories : fréquence, regroupement

3. Transformations
   - Log, sqrt, carré (pour normaliser distributions)
   - Binning (discrétisation)
   - Polynomiales (pour capturer non-linéarité)

4. Agrégations
   - Groupby + statistiques (mean, sum, count, etc.)
   - Rolling windows (séries temporelles)

5. Encoding
   - One-hot encoding (peu de catégories)
   - Target encoding (beaucoup de catégories)
   - Frequency encoding
   - Embedding (Deep Learning)
"""

# Exemples de Feature Engineering

def feature_engineering_dates(df, date_col):
    """
    Extrait des features d'une colonne date
    """
    df[date_col] = pd.to_datetime(df[date_col])

    df[f'{date_col}_year'] = df[date_col].dt.year
    df[f'{date_col}_month'] = df[date_col].dt.month
    df[f'{date_col}_day'] = df[date_col].dt.day
    df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
    df[f'{date_col}_quarter'] = df[date_col].dt.quarter
    df[f'{date_col}_is_weekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
    df[f'{date_col}_is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df[f'{date_col}_is_month_end'] = df[date_col].dt.is_month_end.astype(int)

    return df

def feature_engineering_agregations(df, group_col, agg_col):
    """
    Crée des features d'agrégation
    """
    agg_features = df.groupby(group_col)[agg_col].agg([
        'mean', 'median', 'std', 'min', 'max', 'sum', 'count'
    ]).reset_index()

    agg_features.columns = [group_col] + [f'{agg_col}_{stat}_by_{group_col}'
                                          for stat in ['mean', 'median', 'std', 'min', 'max', 'sum', 'count']]

    df = df.merge(agg_features, on=group_col, how='left')
    return df

def feature_engineering_interactions(df, cols):
    """
    Crée des interactions entre features
    """
    for i, col1 in enumerate(cols):
        for col2 in cols[i+1:]:
            # Produit
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            # Ratio
            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-5)
            # Différence
            df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]

    return df
```

### 4. Encodage des Variables Catégorielles

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder

def guide_encodage(df, col, target=None):
    """
    Guide pour choisir la méthode d'encodage
    """
    n_unique = df[col].nunique()

    print(f"Colonne : {col}")
    print(f"Nombre de catégories uniques : {n_unique}")

    if n_unique == 2:
        print("→ RECOMMANDATION : Label Encoding (2 catégories)")
        print("  from sklearn.preprocessing import LabelEncoder")
    elif n_unique <= 10:
        print("→ RECOMMANDATION : One-Hot Encoding")
        print("  pd.get_dummies() ou OneHotEncoder")
    elif n_unique <= 50:
        print("→ RECOMMANDATION : Target Encoding ou Frequency Encoding")
        print("  from category_encoders import TargetEncoder")
    else:
        print("→ RECOMMANDATION : Target Encoding, Hashing ou Embedding")
        print("  Attention au overfitting avec Target Encoding")
    print()

# 1. Label Encoding (ordinale ou binaire)
le = LabelEncoder()
df['col_encoded'] = le.fit_transform(df['col'])

# 2. One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['col'], drop_first=True)

# 3. Target Encoding
te = TargetEncoder()
df['col_encoded'] = te.fit_transform(df['col'], df['target'])

# 4. Frequency Encoding
freq = df['col'].value_counts(normalize=True)
df['col_encoded'] = df['col'].map(freq)
```

### 5. Normalisation et Standardisation

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def guide_normalisation(df, col):
    """
    Guide pour choisir la méthode de normalisation
    """
    print(f"Analyse de {col}")
    print("="*60)

    # Statistiques
    mean = df[col].mean()
    median = df[col].median()
    std = df[col].std()
    skew = df[col].skew()

    print(f"Moyenne : {mean:.2f}")
    print(f"Médiane : {median:.2f}")
    print(f"Écart-type : {std:.2f}")
    print(f"Skewness : {skew:.2f}")

    # Recommandation
    if abs(skew) < 0.5:
        print("\n→ Distribution proche de la normale")
        print("  RECOMMANDATION : StandardScaler (Z-score)")
    elif abs(skew) >= 0.5:
        print("\n→ Distribution asymétrique")
        print("  RECOMMANDATION : RobustScaler (résistant aux outliers)")

    print("\n→ Pour borner les valeurs dans [0,1] : MinMaxScaler")
    print("→ Pour réseaux de neurones : Normalisation [0,1] ou [-1,1] recommandée")
    print()

# Méthodes de normalisation

# 1. Standardisation (Z-score) : moyenne=0, écart-type=1
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[num_cols])

# 2. Min-Max : valeurs dans [0, 1]
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[num_cols])

# 3. Robust : résistant aux outliers
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df[num_cols])
```

---

## Phase 4 : Modélisation

### Questions pour Choisir un Modèle

#### Arbre de Décision

```
1. Quel est le type de problème ?
   ├─ Classification
   │  ├─ Linéairement séparable ? → Logistic Regression, SVM linéaire
   │  ├─ Non-linéaire ?
   │  │  ├─ Petites données (<10k) → SVM (kernel RBF), Decision Tree
   │  │  ├─ Moyennes données (10k-100k) → Random Forest, XGBoost
   │  │  └─ Grandes données (>100k) → XGBoost, LightGBM, Deep Learning
   │  └─ Interprétabilité requise ? → Logistic Regression, Decision Tree
   │
   ├─ Régression
   │  ├─ Relation linéaire ? → Linear Regression, Ridge, Lasso
   │  ├─ Non-linéaire ?
   │  │  ├─ Petites données → SVR, Decision Tree
   │  │  ├─ Moyennes données → Random Forest, XGBoost
   │  │  └─ Grandes données → XGBoost, LightGBM, Deep Learning
   │  └─ Interprétabilité ? → Linear Regression, Decision Tree
   │
   ├─ Clustering
   │  ├─ Nombre de clusters connu ? → K-Means
   │  ├─ Clusters de formes arbitraires ? → DBSCAN, HDBSCAN
   │  └─ Hiérarchie importante ? → Hierarchical Clustering
   │
   ├─ Réduction de dimensionnalité
   │  ├─ Linéaire + compression → PCA
   │  ├─ Visualisation → t-SNE, UMAP
   │  └─ Non-linéaire + génération → Autoencoder
   │
   └─ Détection d'anomalies
      ├─ Isolation → Isolation Forest
      ├─ Frontière distribution → One-Class SVM
      └─ Reconstruction → Autoencoder

2. Quelle est la taille des données ?
   - <1k : Simple models (Decision Tree, Logistic Regression)
   - 1k-100k : Ensemble methods (Random Forest, XGBoost)
   - >100k : Gradient Boosting, Deep Learning

3. Quel est le type de données ?
   - Tabulaires : XGBoost, Random Forest
   - Images : CNN (ResNet, EfficientNet)
   - Texte : Transformers (BERT), RNN
   - Séries temporelles : LSTM, GRU, Prophet

4. Contraintes de temps ?
   - Entraînement rapide : Logistic Regression, Decision Tree
   - Inférence rapide : Linear models, petits trees
   - Temps non contraint : Deep Learning, XGBoost avec tuning
```

### Workflow de Modélisation

```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

# 1. Diviser les données
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # stratify pour classification
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

# 2. Définir la baseline
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
baseline_score = baseline.score(X_val, y_val)
print(f"Baseline Accuracy: {baseline_score:.4f}")

# 3. Tester plusieurs modèles
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results = {}

for name, model in models.items():
    # Entraîner
    model.fit(X_train, y_train)

    # Évaluer
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    results[name] = {
        'train_score': train_score,
        'val_score': val_score,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    print(f"\n{name}")
    print(f"  Train: {train_score:.4f}")
    print(f"  Val: {val_score:.4f}")
    print(f"  CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Overfitting: {train_score - val_score:.4f}")

# 4. Sélectionner le meilleur modèle
best_model_name = max(results, key=lambda k: results[k]['val_score'])
print(f"\n🏆 Meilleur modèle : {best_model_name}")

# 5. Optimiser les hyperparamètres
if best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nMeilleurs paramètres : {grid_search.best_params_}")
    print(f"Meilleur score CV : {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

# 6. Évaluation finale sur test set
y_pred_test = best_model.predict(X_test)
test_score = accuracy_score(y_test, y_pred_test)
print(f"\n📊 Score final sur test set : {test_score:.4f}")
```

---

## Phase 5 : Évaluation

### Métriques selon le Type de Problème

#### Classification

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

def evaluation_classification(y_true, y_pred, y_proba=None):
    """
    Évaluation complète pour classification
    """
    print("="*80)
    print("ÉVALUATION - CLASSIFICATION")
    print("="*80)

    # Métriques de base
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\nAccuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    # Matrice de confusion
    print("\nMatrice de Confusion :")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Classification report
    print("\nClassification Report :")
    print(classification_report(y_true, y_pred))

    # ROC-AUC (si probabilités disponibles)
    if y_proba is not None:
        if len(np.unique(y_true)) == 2:  # Binaire
            auc = roc_auc_score(y_true, y_proba[:, 1])
            print(f"\nROC-AUC : {auc:.4f}")

            # Courbe ROC
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Courbe ROC')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

    # Visualisation matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Prédiction')
    plt.ylabel('Réalité')
    plt.title('Matrice de Confusion')
    plt.show()

# Utilisation
# evaluation_classification(y_test, y_pred, model.predict_proba(X_test))
```

**Guide de choix de métriques :**

| Contexte | Métrique Principale | Raison |
|----------|---------------------|--------|
| **Classes équilibrées** | Accuracy | Simple et suffisant |
| **Classes déséquilibrées** | F1-Score, ROC-AUC | Prend en compte le déséquilibre |
| **Coût des faux négatifs élevé** (ex: cancer) | Recall | Minimiser les cas manqués |
| **Coût des faux positifs élevé** (ex: spam) | Precision | Minimiser les fausses alarmes |
| **Trade-off** | F1-Score | Équilibre precision/recall |
| **Ranking/probabilités** | ROC-AUC | Évalue qualité des scores |

#### Régression

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluation_regression(y_true, y_pred):
    """
    Évaluation complète pour régression
    """
    print("="*80)
    print("ÉVALUATION - RÉGRESSION")
    print("="*80)

    # Métriques
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\nMSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")

    # Interprétation R²
    if r2 > 0.9:
        print("  → Excellent modèle")
    elif r2 > 0.7:
        print("  → Bon modèle")
    elif r2 > 0.5:
        print("  → Modèle acceptable")
    else:
        print("  → Modèle faible")

    # Visualisations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot : prédictions vs réalité
    axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='k')
    axes[0].plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 'r--', lw=2, label='Prédiction parfaite')
    axes[0].set_xlabel('Valeur Réelle')
    axes[0].set_ylabel('Prédiction')
    axes[0].set_title(f'Prédictions vs Réalité (R² = {r2:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Distribution des résidus
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Résidus')
    axes[1].set_ylabel('Fréquence')
    axes[1].set_title(f'Distribution des Résidus (MAE = {mae:.4f})')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Utilisation
# evaluation_regression(y_test, y_pred)
```

**Guide de choix de métriques :**

| Métrique | Caractéristiques | Usage |
|----------|------------------|-------|
| **MSE** | Pénalise fortement grandes erreurs | Quand grandes erreurs inacceptables |
| **RMSE** | Même unité que la cible | Interprétation facile |
| **MAE** | Robuste aux outliers | Quand outliers dans les erreurs |
| **R²** | Proportion de variance expliquée | Comparaison de modèles |
| **MAPE** | Erreur en pourcentage | Quand échelles variables |

---

## Phase 6 : Déploiement

### Checklist de Déploiement

- [ ] **Sérialiser le modèle**
  ```python
  import joblib
  joblib.dump(model, 'model.pkl')
  # ou
  import pickle
  with open('model.pkl', 'wb') as f:
      pickle.dump(model, f)
  ```

- [ ] **Créer une API** (Flask/FastAPI)
- [ ] **Dockeriser l'application**
- [ ] **Tests unitaires**
- [ ] **CI/CD pipeline**
- [ ] **Monitoring des performances**
- [ ] **Logging des prédictions**
- [ ] **Gestion des versions**
- [ ] **Documentation**

---

## Questions Critiques à se Poser

### Avant de Commencer

1. **Le ML est-il nécessaire ?**
   - Peut-on résoudre avec des règles simples ?
   - Y a-t-il assez de données ?
   - Le ROI justifie-t-il l'investissement ?

2. **Les données sont-elles de qualité ?**
   - Représentatives de la population cible ?
   - Récentes et à jour ?
   - Suffisamment volumineuses ?
   - Bien labelisées (supervisé) ?

3. **Le problème est-il bien défini ?**
   - Objectifs clairs et mesurables ?
   - Critères de succès définis ?
   - Contraintes identifiées ?

### Pendant le Projet

4. **Le modèle apprend-il correctement ?**
   - Overfitting ? (train >> val)
   - Underfitting ? (train et val faibles)
   - Convergence atteinte ?

5. **Les performances sont-elles suffisantes ?**
   - Meilleures que la baseline ?
   - Atteignent les objectifs ?
   - Généralisent sur nouvelles données ?

6. **Le modèle est-il interprétable ?**
   - Features importantes identifiées ?
   - Prédictions expliquables ?
   - Confiance dans les prédictions ?

### Avant Déploiement

7. **Le modèle est-il robuste ?**
   - Testé sur cas limites ?
   - Gère les données manquantes ?
   - Stable dans le temps ?

8. **Le système est-il prêt ?**
   - Infrastructure scalable ?
   - Monitoring en place ?
   - Plan de maintenance défini ?

---

## Templates de Documentation

### Template de Rapport de Projet

```markdown
# Rapport Projet ML : [Nom du Projet]

## 1. Résumé Exécutif
- Problématique : [...]
- Solution : [...]
- Résultats : [...]
- Impact : [...]

## 2. Contexte et Objectifs
### 2.1 Contexte
[Description du contexte métier]

### 2.2 Problématique
[Problème à résoudre]

### 2.3 Objectifs
- Objectif principal : [...]
- Objectifs secondaires : [...]
- Critères de succès : [...]

## 3. Données
### 3.1 Sources
[Sources de données utilisées]

### 3.2 Description
- Volume : [...]
- Période : [...]
- Features : [...]

### 3.3 Qualité
- Valeurs manquantes : [...]
- Outliers : [...]
- Distribution : [...]

## 4. Méthodologie
### 4.1 Préparation des Données
[Étapes de preprocessing]

### 4.2 Feature Engineering
[Features créées]

### 4.3 Modélisation
- Modèles testés : [...]
- Modèle sélectionné : [...]
- Hyperparamètres : [...]

## 5. Résultats
### 5.1 Performances
- Métrique principale : [...]
- Métriques secondaires : [...]
- Comparaison baseline : [...]

### 5.2 Analyse
[Analyse des résultats, features importantes, etc.]

## 6. Déploiement
### 6.1 Architecture
[Schéma de déploiement]

### 6.2 Monitoring
[Métriques suivies]

## 7. Conclusion et Recommandations
### 7.1 Conclusion
[Synthèse]

### 7.2 Limitations
[Limitations identifiées]

### 7.3 Perspectives
[Améliorations futures]
```

---

## Checklist Finale

### Avant de Valider le Projet

- [ ] Problématique claire et objectifs définis
- [ ] Données collectées et analysées
- [ ] EDA complétée
- [ ] Preprocessing et feature engineering documentés
- [ ] Plusieurs modèles testés
- [ ] Baseline dépassée
- [ ] Hyperparamètres optimisés
- [ ] Cross-validation effectuée
- [ ] Évaluation sur test set
- [ ] Analyse des erreurs
- [ ] Features importantes identifiées
- [ ] Modèle sérialisé
- [ ] Documentation complète
- [ ] Code versionné (git)
- [ ] Tests unitaires
- [ ] Rapport de projet rédigé

---

**🎯 Avec ce guide, vous avez toutes les clés pour mener à bien un projet ML de A à Z !**

---

**Navigation :**
- [➡️ Guide de Décision ML](00_Guide_Decision_ML.md)
- [➡️ Workflows ML](00_Workflows_ML.md)
- [🏠 Retour au Sommaire](README.md)
