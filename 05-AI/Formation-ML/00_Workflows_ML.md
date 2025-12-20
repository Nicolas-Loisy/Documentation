# Workflows ML : Construire, Optimiser, Valider et Tester

## 📋 Table des Matières

1. [Workflow Complet d'un Projet ML](#workflow-complet-dun-projet-ml)
2. [Workflow de Construction d'un Modèle](#workflow-de-construction-dun-modèle)
3. [Workflow d'Optimisation](#workflow-doptimisation)
4. [Workflow de Validation](#workflow-de-validation)
5. [Workflow de Deep Learning](#workflow-de-deep-learning)
6. [Pipeline de Production](#pipeline-de-production)
7. [Diagrammes de Décision](#diagrammes-de-décision)

---

## Workflow Complet d'un Projet ML

### Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────────┐
│                     WORKFLOW PROJET ML COMPLET                       │
└─────────────────────────────────────────────────────────────────────┘

1. COMPRÉHENSION DU PROBLÈME
   │
   ├─ Définir la problématique métier
   ├─ Identifier le type de problème ML
   ├─ Définir les objectifs mesurables
   └─ Identifier les contraintes
   │
   ↓
2. COLLECTE ET EXPLORATION DES DONNÉES (EDA)
   │
   ├─ Identifier les sources de données
   ├─ Collecter les données
   ├─ Analyser statistiques descriptives
   ├─ Visualiser les distributions
   ├─ Identifier valeurs manquantes, outliers
   └─ Analyser les corrélations
   │
   ↓
3. PRÉPARATION DES DONNÉES
   │
   ├─ Traiter valeurs manquantes
   ├─ Gérer les outliers
   ├─ Feature Engineering
   ├─ Encoder variables catégorielles
   ├─ Normaliser/Standardiser
   └─ Split Train/Val/Test
   │
   ↓
4. MODÉLISATION
   │
   ├─ Définir baseline
   ├─ Tester plusieurs modèles
   ├─ Sélectionner le meilleur
   ├─ Optimiser hyperparamètres
   └─ Valider avec cross-validation
   │
   ↓
5. ÉVALUATION
   │
   ├─ Évaluer sur test set
   ├─ Calculer métriques
   ├─ Analyser les erreurs
   └─ Interpréter le modèle
   │
   ↓
6. DÉPLOIEMENT
   │
   ├─ Sérialiser le modèle
   ├─ Créer API
   ├─ Dockeriser
   ├─ Monitoring
   └─ Maintenance
```

---

## Workflow de Construction d'un Modèle

### Étapes Détaillées

```python
# ═══════════════════════════════════════════════════════════════════
# WORKFLOW DE CONSTRUCTION D'UN MODÈLE ML
# ═══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────────────────────────
# ÉTAPE 1 : CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────────
print("="*80)
print("ÉTAPE 1 : CHARGEMENT DES DONNÉES")
print("="*80)

df = pd.read_csv('data.csv')
print(f"✓ Données chargées : {df.shape[0]} lignes × {df.shape[1]} colonnes")

# ─────────────────────────────────────────────────────────────────
# ÉTAPE 2 : EXPLORATION DES DONNÉES (EDA)
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("ÉTAPE 2 : EXPLORATION DES DONNÉES")
print("="*80)

# 2.1 Aperçu général
print("\n📊 Aperçu des données :")
print(df.head())
print("\n📈 Informations générales :")
print(df.info())
print("\n📉 Statistiques descriptives :")
print(df.describe())

# 2.2 Valeurs manquantes
print("\n❓ Valeurs manquantes :")
missing = df.isnull().sum()
missing_pct = 100 * missing / len(df)
missing_table = pd.DataFrame({
    'Manquantes': missing,
    'Pourcentage': missing_pct
})
print(missing_table[missing_table['Manquantes'] > 0])

# 2.3 Distribution de la variable cible
print("\n🎯 Distribution de la variable cible :")
print(df['target'].value_counts())

# 2.4 Corrélations
print("\n🔗 Corrélations avec la cible :")
correlations = df.corr()['target'].sort_values(ascending=False)
print(correlations)

# ─────────────────────────────────────────────────────────────────
# ÉTAPE 3 : PRÉPARATION DES DONNÉES
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("ÉTAPE 3 : PRÉPARATION DES DONNÉES")
print("="*80)

# 3.1 Séparer features et target
X = df.drop('target', axis=1)
y = df['target']
print(f"✓ Features : {X.shape}")
print(f"✓ Target : {y.shape}")

# 3.2 Traiter valeurs manquantes
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)
print(f"✓ Valeurs manquantes traitées")

# 3.3 Encoder variables catégorielles
cat_cols = X_imputed.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    X_encoded = pd.get_dummies(X_imputed, columns=cat_cols, drop_first=True)
    print(f"✓ Variables catégorielles encodées : {len(cat_cols)} colonnes")
else:
    X_encoded = X_imputed

# 3.4 Split Train/Val/Test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\n✓ Données divisées :")
print(f"  Train : {len(X_train)} ({len(X_train)/len(X_encoded)*100:.1f}%)")
print(f"  Val   : {len(X_val)} ({len(X_val)/len(X_encoded)*100:.1f}%)")
print(f"  Test  : {len(X_test)} ({len(X_test)/len(X_encoded)*100:.1f}%)")

# 3.5 Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print(f"✓ Données normalisées")

# ─────────────────────────────────────────────────────────────────
# ÉTAPE 4 : BASELINE
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("ÉTAPE 4 : BASELINE")
print("="*80)

from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train_scaled, y_train)
baseline_score = baseline.score(X_val_scaled, y_val)

print(f"📊 Baseline (most_frequent) : {baseline_score:.4f}")

# ─────────────────────────────────────────────────────────────────
# ÉTAPE 5 : MODÉLISATION - TESTER PLUSIEURS MODÈLES
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("ÉTAPE 5 : MODÉLISATION")
print("="*80)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results = {}

print("\n🤖 Entraînement des modèles...\n")
for name, model in models.items():
    # Entraîner
    model.fit(X_train_scaled, y_train)

    # Prédire
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    # Scores
    train_score = accuracy_score(y_train, y_train_pred)
    val_score = accuracy_score(y_val, y_val_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    results[name] = {
        'model': model,
        'train_score': train_score,
        'val_score': val_score,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    print(f"{name}")
    print(f"  Train    : {train_score:.4f}")
    print(f"  Val      : {val_score:.4f}")
    print(f"  CV       : {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
    print(f"  Overfit  : {train_score - val_score:.4f}")
    print()

# ─────────────────────────────────────────────────────────────────
# ÉTAPE 6 : SÉLECTION DU MEILLEUR MODÈLE
# ─────────────────────────────────────────────────────────────────
print("="*80)
print("ÉTAPE 6 : SÉLECTION DU MEILLEUR MODÈLE")
print("="*80)

best_model_name = max(results, key=lambda k: results[k]['val_score'])
best_model = results[best_model_name]['model']

print(f"\n🏆 Meilleur modèle : {best_model_name}")
print(f"   Val Score : {results[best_model_name]['val_score']:.4f}")

# ─────────────────────────────────────────────────────────────────
# ÉTAPE 7 : OPTIMISATION DES HYPERPARAMÈTRES
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("ÉTAPE 7 : OPTIMISATION DES HYPERPARAMÈTRES")
print("="*80)

from sklearn.model_selection import GridSearchCV

# Définir grille selon le modèle
if best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
    }
elif best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
    }
else:
    param_grid = {}

if param_grid:
    print(f"\n🔧 Optimisation de {best_model_name}...\n")

    grid_search = GridSearchCV(
        best_model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    print(f"\n✓ Meilleurs paramètres : {grid_search.best_params_}")
    print(f"✓ Meilleur score CV : {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

# ─────────────────────────────────────────────────────────────────
# ÉTAPE 8 : ÉVALUATION FINALE SUR TEST SET
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("ÉTAPE 8 : ÉVALUATION FINALE")
print("="*80)

y_test_pred = best_model.predict(X_test_scaled)
test_score = accuracy_score(y_test, y_test_pred)

print(f"\n📊 Score final sur test set : {test_score:.4f}")
print(f"\n📋 Classification Report :\n")
print(classification_report(y_test, y_test_pred))

# Matrice de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédiction')
plt.ylabel('Réalité')
plt.title('Matrice de Confusion')
plt.show()

# ─────────────────────────────────────────────────────────────────
# ÉTAPE 9 : SAUVEGARDE DU MODÈLE
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("ÉTAPE 9 : SAUVEGARDE")
print("="*80)

import joblib

# Sauvegarder le modèle et le scaler
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✓ Modèle sauvegardé : best_model.pkl")
print("✓ Scaler sauvegardé : scaler.pkl")

print("\n" + "="*80)
print("🎉 WORKFLOW TERMINÉ AVEC SUCCÈS !")
print("="*80)
```

---

## Workflow d'Optimisation

### Diagramme d'Optimisation

```
┌──────────────────────────────────────────────────────────────────┐
│                  WORKFLOW D'OPTIMISATION                          │
└──────────────────────────────────────────────────────────────────┘

1. MODÈLE INITIAL
   │
   ↓
2. DIAGNOSTIC
   │
   ├─ Overfitting ? (train >> val)
   │  └─ OUI → [A] Réduire complexité
   │            - Diminuer max_depth (trees)
   │            - Augmenter régularisation (L1/L2)
   │            - Dropout (neural networks)
   │            - Moins de features
   │            - Plus de données
   │
   ├─ Underfitting ? (train et val faibles)
   │  └─ OUI → [B] Augmenter complexité
   │            - Augmenter max_depth
   │            - Plus de features
   │            - Modèle plus complexe
   │            - Diminuer régularisation
   │
   └─ Performance OK mais améliorable ?
      └─ OUI → [C] Optimisation fine
                - Grid Search / Random Search
                - Bayesian Optimization
                - Feature Engineering
                - Ensemble methods
   │
   ↓
3. APPLIQUER MODIFICATIONS
   │
   ↓
4. ÉVALUER
   │
   ├─ Amélioration ? → Continuer
   └─ Pas d'amélioration ? → Retour au meilleur modèle
   │
   ↓
5. VALIDATION FINALE
```

### Code d'Optimisation Complète

```python
# ═══════════════════════════════════════════════════════════════════
# WORKFLOW D'OPTIMISATION COMPLÈTE
# ═══════════════════════════════════════════════════════════════════

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

# ─────────────────────────────────────────────────────────────────
# ÉTAPE 1 : DIAGNOSTIC
# ─────────────────────────────────────────────────────────────────

def diagnostic_modele(model, X_train, y_train, X_val, y_val):
    """
    Diagnostique overfitting/underfitting
    """
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)

    print("="*60)
    print("DIAGNOSTIC DU MODÈLE")
    print("="*60)
    print(f"Score Train : {train_score:.4f}")
    print(f"Score Val   : {val_score:.4f}")
    print(f"Différence  : {train_score - val_score:.4f}")

    if train_score - val_score > 0.1:
        print("\n⚠️  OVERFITTING DÉTECTÉ")
        print("Recommandations :")
        print("  - Augmenter régularisation")
        print("  - Diminuer complexité du modèle")
        print("  - Ajouter plus de données")
        print("  - Feature selection")
        print("  - Early stopping")
        return "overfitting"

    elif train_score < 0.7 and val_score < 0.7:
        print("\n⚠️  UNDERFITTING DÉTECTÉ")
        print("Recommandations :")
        print("  - Augmenter complexité du modèle")
        print("  - Ajouter plus de features")
        print("  - Diminuer régularisation")
        print("  - Feature engineering")
        return "underfitting"

    else:
        print("\n✓ Modèle équilibré")
        print("Recommandations :")
        print("  - Optimiser hyperparamètres")
        print("  - Feature engineering avancé")
        print("  - Ensemble methods")
        return "balanced"

# ─────────────────────────────────────────────────────────────────
# ÉTAPE 2 : OPTIMISATION HYPERPARAMÈTRES - GRID SEARCH
# ─────────────────────────────────────────────────────────────────

def optimisation_grid_search(model, param_grid, X_train, y_train, cv=5):
    """
    Grid Search pour optimiser hyperparamètres
    """
    print("\n" + "="*60)
    print("GRID SEARCH")
    print("="*60)
    print(f"Paramètres testés : {param_grid}")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n✓ Meilleurs paramètres : {grid_search.best_params_}")
    print(f"✓ Meilleur score CV : {grid_search.best_score_:.4f}")

    # Résultats détaillés
    results = pd.DataFrame(grid_search.cv_results_)
    results = results.sort_values('rank_test_score')
    print("\nTop 5 configurations :")
    print(results[['params', 'mean_test_score', 'std_test_score']].head())

    return grid_search.best_estimator_

# ─────────────────────────────────────────────────────────────────
# ÉTAPE 3 : OPTIMISATION HYPERPARAMÈTRES - RANDOM SEARCH
# ─────────────────────────────────────────────────────────────────

def optimisation_random_search(model, param_distributions, X_train, y_train,
                                n_iter=50, cv=5):
    """
    Random Search (plus rapide que Grid Search)
    """
    print("\n" + "="*60)
    print("RANDOM SEARCH")
    print("="*60)
    print(f"Paramètres : {param_distributions}")
    print(f"Itérations : {n_iter}")

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    print(f"\n✓ Meilleurs paramètres : {random_search.best_params_}")
    print(f"✓ Meilleur score CV : {random_search.best_score_:.4f}")

    return random_search.best_estimator_

# ─────────────────────────────────────────────────────────────────
# ÉTAPE 4 : FEATURE IMPORTANCE ET SÉLECTION
# ─────────────────────────────────────────────────────────────────

def feature_importance_analysis(model, X, feature_names):
    """
    Analyse l'importance des features
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\n" + "="*60)
        print("IMPORTANCE DES FEATURES")
        print("="*60)

        print("\nTop 10 features :")
        for i in range(min(10, len(feature_names))):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

        # Visualisation
        plt.figure(figsize=(10, 6))
        plt.bar(range(min(20, len(feature_names))),
                importances[indices[:min(20, len(feature_names))]])
        plt.xticks(range(min(20, len(feature_names))),
                   [feature_names[i] for i in indices[:min(20, len(feature_names))]],
                   rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Top 20 Features les Plus Importantes')
        plt.tight_layout()
        plt.show()

        return importances, indices

# ─────────────────────────────────────────────────────────────────
# ÉTAPE 5 : LEARNING CURVES
# ─────────────────────────────────────────────────────────────────

from sklearn.model_selection import learning_curve

def plot_learning_curves(model, X, y, cv=5):
    """
    Trace les courbes d'apprentissage
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=cv,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Train Score', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.15)

    plt.plot(train_sizes, val_mean, label='Validation Score', marker='o')
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.15)

    plt.xlabel('Taille du Training Set')
    plt.ylabel('Score')
    plt.title('Courbes d\'Apprentissage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Diagnostic
    final_gap = train_mean[-1] - val_mean[-1]
    if final_gap > 0.1:
        print("⚠️  Overfitting : écart important entre train et val")
        print("   → Ajouter plus de données ou régulariser")
    elif val_mean[-1] < 0.7:
        print("⚠️  Underfitting : scores faibles")
        print("   → Augmenter complexité du modèle")
    else:
        print("✓ Courbes saines")

# ─────────────────────────────────────────────────────────────────
# EXEMPLE D'UTILISATION COMPLÈTE
# ─────────────────────────────────────────────────────────────────

# 1. Diagnostic
status = diagnostic_modele(model, X_train_scaled, y_train,
                           X_val_scaled, y_val)

# 2. Optimisation selon diagnostic
if status == "overfitting":
    # Augmenter régularisation
    param_grid = {
        'max_depth': [3, 5, 7],
        'min_samples_split': [10, 20, 30],
        'n_estimators': [100]
    }
elif status == "underfitting":
    # Augmenter complexité
    param_grid = {
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'n_estimators': [200, 300]
    }
else:
    # Optimisation fine
    param_grid = {
        'max_depth': [5, 7, 10],
        'min_samples_split': [5, 10],
        'n_estimators': [100, 200]
    }

# 3. Grid Search
best_model = optimisation_grid_search(
    RandomForestClassifier(random_state=42),
    param_grid,
    X_train_scaled, y_train
)

# 4. Feature Importance
importances, indices = feature_importance_analysis(
    best_model, X_train_scaled, X_train.columns
)

# 5. Learning Curves
plot_learning_curves(best_model, X_train_scaled, y_train)

# 6. Évaluation finale
y_val_pred = best_model.predict(X_val_scaled)
val_score = accuracy_score(y_val, y_val_pred)
print(f"\n📊 Score final après optimisation : {val_score:.4f}")
```

---

## Workflow de Validation

### Stratégies de Validation

```
┌──────────────────────────────────────────────────────────────────┐
│                    STRATÉGIES DE VALIDATION                       │
└──────────────────────────────────────────────────────────────────┘

1. HOLDOUT (Train/Val/Test Split)
   ┌─────────────┬──────┬──────┐
   │    Train    │ Val  │ Test │
   │    70%      │ 15%  │ 15%  │
   └─────────────┴──────┴──────┘
   Usage : Grandes données (>10k)

2. K-FOLD CROSS-VALIDATION
   Fold 1: [Test][Train][Train][Train][Train]
   Fold 2: [Train][Test][Train][Train][Train]
   Fold 3: [Train][Train][Test][Train][Train]
   Fold 4: [Train][Train][Train][Test][Train]
   Fold 5: [Train][Train][Train][Train][Test]
   Usage : Petites/moyennes données

3. STRATIFIED K-FOLD
   - Comme K-Fold mais préserve la distribution des classes
   Usage : Classes déséquilibrées

4. TIME SERIES SPLIT
   Fold 1: [Train][Test]─────────────────────
   Fold 2: [Train──────][Test]───────────────
   Fold 3: [Train─────────────][Test]────────
   Fold 4: [Train────────────────────][Test]─
   Usage : Séries temporelles

5. LEAVE-ONE-OUT (LOO)
   - Chaque sample utilisé une fois comme test
   Usage : Très petites données (<100)
```

### Code de Validation

```python
# ═══════════════════════════════════════════════════════════════════
# STRATÉGIES DE VALIDATION
# ═══════════════════════════════════════════════════════════════════

from sklearn.model_selection import (
    cross_val_score, cross_validate,
    KFold, StratifiedKFold, TimeSeriesSplit, LeaveOneOut
)

# ─────────────────────────────────────────────────────────────────
# 1. K-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────

def k_fold_validation(model, X, y, k=5):
    """
    K-Fold Cross-Validation standard
    """
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

    print(f"K-Fold CV ({k} folds)")
    print(f"  Scores : {scores}")
    print(f"  Moyenne : {scores.mean():.4f} (± {scores.std():.4f})")

    return scores

# ─────────────────────────────────────────────────────────────────
# 2. STRATIFIED K-FOLD (pour classes déséquilibrées)
# ─────────────────────────────────────────────────────────────────

def stratified_k_fold_validation(model, X, y, k=5):
    """
    Stratified K-Fold : préserve distribution des classes
    """
    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')

    print(f"Stratified K-Fold CV ({k} folds)")
    print(f"  Scores : {scores}")
    print(f"  Moyenne : {scores.mean():.4f} (± {scores.std():.4f})")

    # Vérifier équilibre des classes dans chaque fold
    for fold, (train_idx, val_idx) in enumerate(skfold.split(X, y)):
        train_dist = np.bincount(y.iloc[train_idx]) / len(train_idx)
        val_dist = np.bincount(y.iloc[val_idx]) / len(val_idx)
        print(f"  Fold {fold+1} - Train: {train_dist}, Val: {val_dist}")

    return scores

# ─────────────────────────────────────────────────────────────────
# 3. CROSS-VALIDATE (métriques multiples)
# ─────────────────────────────────────────────────────────────────

def cross_validate_multimetrics(model, X, y, k=5):
    """
    Cross-validation avec plusieurs métriques
    """
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted'
    }

    cv_results = cross_validate(
        model, X, y,
        cv=k,
        scoring=scoring,
        return_train_score=True
    )

    print(f"Cross-Validation ({k} folds) - Métriques multiples")
    for metric in scoring.keys():
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']
        print(f"\n{metric.upper()}")
        print(f"  Train : {train_scores.mean():.4f} (± {train_scores.std():.4f})")
        print(f"  Test  : {test_scores.mean():.4f} (± {test_scores.std():.4f})")

    return cv_results

# ─────────────────────────────────────────────────────────────────
# 4. TIME SERIES SPLIT
# ─────────────────────────────────────────────────────────────────

def time_series_validation(model, X, y, n_splits=5):
    """
    Validation pour séries temporelles
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)

        print(f"Fold {fold+1}: Train [{train_idx[0]}:{train_idx[-1]}], "
              f"Val [{val_idx[0]}:{val_idx[-1]}], Score: {score:.4f}")

    print(f"\nMoyenne : {np.mean(scores):.4f} (± {np.std(scores):.4f})")

    return scores
```

---

## Workflow de Deep Learning

### Diagramme DL

```
┌──────────────────────────────────────────────────────────────────┐
│              WORKFLOW DEEP LEARNING SPÉCIFIQUE                    │
└──────────────────────────────────────────────────────────────────┘

1. ARCHITECTURE
   │
   ├─ Définir architecture (layers, neurons, activations)
   ├─ Choisir loss function
   └─ Choisir optimizer
   │
   ↓
2. ENTRAÎNEMENT INITIAL
   │
   ├─ Petits epochs (10-20)
   ├─ Learning rate par défaut
   └─ Monitoring (loss, accuracy)
   │
   ↓
3. DIAGNOSTIC
   │
   ├─ Overfitting ?
   │  └─ Ajouter : Dropout, L2 regularization, Data augmentation
   │
   ├─ Underfitting ?
   │  └─ Augmenter : Capacité du modèle, epochs, compléxité
   │
   └─ Convergence lente ?
      └─ Ajuster : Learning rate, optimizer, batch size
   │
   ↓
4. OPTIMISATION
   │
   ├─ Learning Rate Scheduling
   ├─ Early Stopping
   ├─ Callbacks (ModelCheckpoint, ReduceLROnPlateau)
   └─ Data Augmentation
   │
   ↓
5. FINE-TUNING
   │
   ├─ Transfer Learning (si applicable)
   ├─ Unfreeze layers
   └─ Fine-tune avec LR faible
   │
   ↓
6. ENSEMBLE (optionnel)
   │
   └─ Combiner plusieurs modèles
```

### Code Deep Learning Workflow

```python
# ═══════════════════════════════════════════════════════════════════
# WORKFLOW DEEP LEARNING COMPLET
# ═══════════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ─────────────────────────────────────────────────────────────────
# 1. DÉFINIR L'ARCHITECTURE
# ─────────────────────────────────────────────────────────────────

def creer_modele(input_shape, num_classes):
    """
    Crée un modèle Neural Network
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Hidden layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation='relu'),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# ─────────────────────────────────────────────────────────────────
# 2. COMPILER LE MODÈLE
# ─────────────────────────────────────────────────────────────────

model = creer_modele(input_shape=(X_train.shape[1],), num_classes=3)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# ─────────────────────────────────────────────────────────────────
# 3. CALLBACKS POUR OPTIMISATION
# ─────────────────────────────────────────────────────────────────

# Early Stopping : arrête si pas d'amélioration
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Model Checkpoint : sauvegarde meilleur modèle
checkpoint = callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Reduce LR on Plateau : réduit LR si stagnation
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# TensorBoard : visualisation
tensorboard = callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1
)

# ─────────────────────────────────────────────────────────────────
# 4. ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint, reduce_lr, tensorboard],
    verbose=1
)

# ─────────────────────────────────────────────────────────────────
# 5. VISUALISATION DE L'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────

def plot_training_history(history):
    """
    Visualise les courbes d'entraînement
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_training_history(history)

# ─────────────────────────────────────────────────────────────────
# 6. ÉVALUATION FINALE
# ─────────────────────────────────────────────────────────────────

test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\n📊 Test Accuracy : {test_acc:.4f}")
print(f"📊 Test Loss : {test_loss:.4f}")
```

---

## Pipeline de Production

### Architecture Complète

```
┌──────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE PRODUCTION                         │
└──────────────────────────────────────────────────────────────────┘

┌─────────────┐
│  RAW DATA   │
└──────┬──────┘
       │
       ↓
┌─────────────────────┐
│  DATA PROCESSING    │
│  - Cleaning         │
│  - Feature Eng.     │
│  - Encoding         │
│  - Scaling          │
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│  MODEL TRAINING     │
│  - Train/Val/Test   │
│  - Cross-validation │
│  - Hyperparameter   │
│    Tuning           │
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│  MODEL EVALUATION   │
│  - Metrics          │
│  - Error Analysis   │
│  - A/B Testing      │
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│  MODEL DEPLOYMENT   │
│  - API (Flask/      │
│    FastAPI)         │
│  - Docker           │
│  - CI/CD            │
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│  MONITORING         │
│  - Performance      │
│  - Data Drift       │
│  - Model Drift      │
│  - Retraining       │
└─────────────────────┘
```

### Code Pipeline Scikit-Learn

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ═══════════════════════════════════════════════════════════════════
# PIPELINE SCIKIT-LEARN COMPLET
# ═══════════════════════════════════════════════════════════════════

# Identifier colonnes numériques et catégorielles
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Pipeline pour features numériques
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline pour features catégorielles
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# Combiner les transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline complet : preprocessing + modèle
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Entraîner le pipeline
pipeline.fit(X_train, y_train)

# Prédire
y_pred = pipeline.predict(X_test)

# Score
score = pipeline.score(X_test, y_test)
print(f"Score : {score:.4f}")

# Sauvegarder le pipeline complet
import joblib
joblib.dump(pipeline, 'pipeline_complet.pkl')

# Charger et utiliser
pipeline_loaded = joblib.load('pipeline_complet.pkl')
predictions = pipeline_loaded.predict(new_data)
```

---

## Diagrammes de Décision

### Problème d'Overfitting

```
OVERFITTING DÉTECTÉ (train >> val)
│
├─ Régularisation
│  ├─ L1 (Lasso) → Sélection de features
│  ├─ L2 (Ridge) → Réduire coefficients
│  └─ Dropout (NN) → Désactiver neurones aléatoirement
│
├─ Réduire Complexité
│  ├─ Diminuer max_depth (trees)
│  ├─ Diminuer nombre de layers (NN)
│  └─ Feature selection
│
├─ Augmenter Données
│  ├─ Collecter plus de données
│  └─ Data augmentation (images)
│
└─ Early Stopping
   └─ Arrêter entraînement avant overfitting
```

### Problème d'Underfitting

```
UNDERFITTING DÉTECTÉ (train et val faibles)
│
├─ Augmenter Complexité
│  ├─ Augmenter max_depth
│  ├─ Ajouter layers (NN)
│  └─ Utiliser modèle plus complexe
│
├─ Feature Engineering
│  ├─ Créer nouvelles features
│  ├─ Interactions
│  └─ Transformations (log, sqrt, etc.)
│
├─ Diminuer Régularisation
│  ├─ Diminuer alpha (Lasso/Ridge)
│  └─ Diminuer dropout rate
│
└─ Entraîner Plus Longtemps
   └─ Augmenter epochs/iterations
```

---

**🎯 Ces workflows vous guident étape par étape dans vos projets ML !**

---

**Navigation :**

- [⬅️ Guide de Décision ML](00_Guide_Decision_ML.md)
- [➡️ Notebooks Tutoriels](README_ML.md)
- [🏠 Retour au Sommaire](README_ML.md)
