"""
═══════════════════════════════════════════════════════════════════════════════
TUTORIEL COMPLET : RÉGRESSION LINÉAIRE
═══════════════════════════════════════════════════════════════════════════════

Ce tutoriel couvre :
1. Comprendre la régression linéaire
2. Préparer les données
3. Entraîner le modèle
4. Optimiser les hyperparamètres
5. Valider le modèle
6. Tester et interpréter les résultats
7. Comparaison Ridge et Lasso

Chaque étape est expliquée en détail avec les paramètres et fonctions.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("TUTORIEL : RÉGRESSION LINÉAIRE")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 1 : COMPRENDRE LA RÉGRESSION LINÉAIRE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 1 : THÉORIE")
print("="*80)

print("""
🎯 OBJECTIF
-----------
La régression linéaire vise à modéliser la relation entre :
- Une variable dépendante Y (cible)
- Une ou plusieurs variables indépendantes X (features)

📐 FORMULE MATHÉMATIQUE
-----------------------
Régression linéaire simple (1 feature) :
    Y = β₀ + β₁·X + ε

Régression linéaire multiple (n features) :
    Y = β₀ + β₁·X₁ + β₂·X₂ + ... + βₙ·Xₙ + ε

où :
- Y : variable à prédire
- X : features (variables explicatives)
- β₀ : intercept (ordonnée à l'origine)
- β₁, β₂, ..., βₙ : coefficients (pentes)
- ε : erreur résiduelle

🎯 OBJECTIF D'OPTIMISATION
---------------------------
Minimiser la somme des erreurs au carré (MSE) :

    MSE = (1/n) Σ(yᵢ - ŷᵢ)²

où :
- yᵢ : valeur réelle
- ŷᵢ : valeur prédite
- n : nombre d'observations

📊 HYPOTHÈSES DE LA RÉGRESSION LINÉAIRE
----------------------------------------
1. Linéarité : relation linéaire entre X et Y
2. Indépendance : observations indépendantes
3. Homoscédasticité : variance constante des résidus
4. Normalité : résidus suivent une distribution normale
5. Absence de multicolinéarité : features peu corrélées entre elles
""")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 2 : GÉNÉRATION ET PRÉPARATION DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 2 : PRÉPARATION DES DONNÉES")
print("="*80)

# 2.1 Générer des données synthétiques
print("\n📊 Génération de données synthétiques...\n")

# make_regression : Génère un problème de régression
# - n_samples : nombre d'échantillons
# - n_features : nombre de features
# - n_informative : nombre de features utiles
# - noise : écart-type du bruit gaussien
# - random_state : graine pour reproductibilité
X, y = make_regression(
    n_samples=500,
    n_features=5,
    n_informative=3,
    noise=10.0,
    random_state=42
)

print(f"✓ Données générées : {X.shape[0]} échantillons × {X.shape[1]} features")
print(f"✓ Target : {y.shape[0]} valeurs")

# Créer un DataFrame pour faciliter la visualisation
feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y

print("\n📈 Aperçu des données :")
print(df.head())

print("\n📉 Statistiques descriptives :")
print(df.describe())

# 2.2 Visualisation des données
print("\n📊 Visualisation des distributions...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

# Histogrammes des features
for i, col in enumerate(feature_names):
    axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Distribution de {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Fréquence')

# Histogramme de la target
axes[5].hist(df['Target'], bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[5].set_title('Distribution de Target')
axes[5].set_xlabel('Target')
axes[5].set_ylabel('Fréquence')

plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/regression_distributions.png')
plt.show()

print("✓ Graphique sauvegardé : regression_distributions.png")

# 2.3 Matrice de corrélation
print("\n🔗 Analyse des corrélations...")

correlation_matrix = df.corr()
print("\nCorrélations avec la Target :")
print(correlation_matrix['Target'].sort_values(ascending=False))

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5, center=0)
plt.title('Matrice de Corrélation')
plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/regression_correlation.png')
plt.show()

print("✓ Graphique sauvegardé : regression_correlation.png")

# 2.4 Division des données
print("\n✂️  Division des données...")

# train_test_split : Divise les données en train et test
# - test_size : proportion du test set (0.2 = 20%)
# - random_state : graine pour reproductibilité
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"✓ Train set : {len(X_train)} échantillons ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Val set   : {len(X_val)} échantillons ({len(X_val)/len(X)*100:.1f}%)")
print(f"✓ Test set  : {len(X_test)} échantillons ({len(X_test)/len(X)*100:.1f}%)")

# 2.5 Normalisation
print("\n⚖️  Normalisation des données...")

# StandardScaler : Standardise les features (moyenne=0, écart-type=1)
# Formule : z = (x - μ) / σ
# - fit() : calcule moyenne et écart-type sur train
# - transform() : applique la transformation
# ⚠️ IMPORTANT : fit uniquement sur train, transform sur train/val/test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Moyenne des features (train) : {X_train_scaled.mean(axis=0)}")
print(f"✓ Écart-type des features (train) : {X_train_scaled.std(axis=0)}")
print("✓ Données normalisées")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 3 : MODÈLE DE BASE (BASELINE)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 3 : BASELINE")
print("="*80)

print("""
📌 POURQUOI UNE BASELINE ?
--------------------------
Avant de construire un modèle complexe, établir une baseline permet de :
- Avoir un point de référence
- Valider que le modèle apporte de la valeur
- Identifier rapidement les problèmes

Baseline simple : Prédire la moyenne de Y
""")

# Baseline : prédire toujours la moyenne
y_train_mean = y_train.mean()
y_val_pred_baseline = np.full(len(y_val), y_train_mean)

mse_baseline = mean_squared_error(y_val, y_val_pred_baseline)
rmse_baseline = np.sqrt(mse_baseline)
mae_baseline = mean_absolute_error(y_val, y_val_pred_baseline)
r2_baseline = r2_score(y_val, y_val_pred_baseline)

print(f"\n📊 BASELINE (prédire moyenne = {y_train_mean:.2f})")
print(f"  MSE  : {mse_baseline:.2f}")
print(f"  RMSE : {rmse_baseline:.2f}")
print(f"  MAE  : {mae_baseline:.2f}")
print(f"  R²   : {r2_baseline:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 4 : RÉGRESSION LINÉAIRE SIMPLE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 4 : RÉGRESSION LINÉAIRE")
print("="*80)

print("""
🤖 LinearRegression de Scikit-Learn
------------------------------------
Implémente la régression linéaire par moindres carrés ordinaires (OLS).

PARAMÈTRES PRINCIPAUX :
- fit_intercept (bool, default=True) : Calculer l'intercept ?
- normalize (bool, default=False) : Normaliser avant fit ?
- n_jobs (int, default=None) : Nombre de CPU (-1 = tous)

MÉTHODE :
- fit(X, y) : Entraîne le modèle
- predict(X) : Fait des prédictions
- score(X, y) : Retourne le R² score

ATTRIBUTS APRÈS FIT :
- coef_ : Coefficients de la régression
- intercept_ : Ordonnée à l'origine
""")

# 4.1 Créer et entraîner le modèle
print("\n🚀 Entraînement du modèle...\n")

# Créer l'instance du modèle
# fit_intercept=True : calculer l'intercept (β₀)
model_lr = LinearRegression(fit_intercept=True)

# fit() : Entraîne le modèle sur les données d'entraînement
# Calcule les coefficients qui minimisent MSE
model_lr.fit(X_train_scaled, y_train)

print("✓ Modèle entraîné")

# 4.2 Examiner les paramètres du modèle
print("\n📐 Paramètres du modèle :")
print(f"  Intercept (β₀) : {model_lr.intercept_:.4f}")
print(f"\n  Coefficients (β₁, ..., βₙ) :")
for i, coef in enumerate(model_lr.coef_):
    print(f"    Feature_{i+1} : {coef:.4f}")

# 4.3 Prédictions
print("\n🔮 Prédictions...\n")

# predict() : Génère des prédictions
# ŷ = β₀ + β₁·X₁ + β₂·X₂ + ... + βₙ·Xₙ
y_train_pred = model_lr.predict(X_train_scaled)
y_val_pred = model_lr.predict(X_val_scaled)
y_test_pred = model_lr.predict(X_test_scaled)

print("✓ Prédictions générées")

# 4.4 Évaluation
print("\n📊 ÉVALUATION\n")

# Métriques sur train
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Métriques sur validation
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print("Train Set :")
print(f"  MSE  : {train_mse:.2f}")
print(f"  RMSE : {train_rmse:.2f}")
print(f"  MAE  : {train_mae:.2f}")
print(f"  R²   : {train_r2:.4f}")

print("\nValidation Set :")
print(f"  MSE  : {val_mse:.2f}")
print(f"  RMSE : {val_rmse:.2f}")
print(f"  MAE  : {val_mae:.2f}")
print(f"  R²   : {val_r2:.4f}")

print("\nComparaison avec Baseline :")
print(f"  Amélioration R² : {val_r2 - r2_baseline:.4f}")
print(f"  Amélioration RMSE : {rmse_baseline - val_rmse:.2f}")

# 4.5 Visualisation des prédictions
print("\n📊 Visualisation des prédictions...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot : Prédictions vs Réalité
axes[0].scatter(y_val, y_val_pred, alpha=0.6, edgecolors='k')
axes[0].plot([y_val.min(), y_val.max()],
             [y_val.min(), y_val.max()],
             'r--', lw=2, label='Prédiction parfaite')
axes[0].set_xlabel('Valeurs Réelles')
axes[0].set_ylabel('Prédictions')
axes[0].set_title(f'Prédictions vs Réalité (R² = {val_r2:.4f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Histogramme des résidus
residuals = y_val - y_val_pred
axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Résidus')
axes[1].set_ylabel('Fréquence')
axes[1].set_title(f'Distribution des Résidus (MAE = {val_mae:.2f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/regression_predictions.png')
plt.show()

print("✓ Graphique sauvegardé : regression_predictions.png")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 5 : VALIDATION CROISÉE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 5 : VALIDATION CROISÉE (CROSS-VALIDATION)")
print("="*80)

print("""
🔄 CROSS-VALIDATION
-------------------
Technique pour évaluer la robustesse d'un modèle :
- Divise les données en K folds
- Entraîne K fois (chaque fois avec 1 fold pour validation)
- Moyenne les scores pour avoir une estimation plus fiable

AVANTAGES :
✓ Utilise toutes les données pour entraînement et validation
✓ Réduit la variance de l'estimation
✓ Détecte l'overfitting

K typique : 5 ou 10
""")

# cross_val_score : Effectue la cross-validation
# - estimator : modèle à évaluer
# - X, y : données
# - cv : nombre de folds
# - scoring : métrique ('r2', 'neg_mean_squared_error', etc.)
# - n_jobs : parallélisation
cv_scores_r2 = cross_val_score(
    model_lr, X_train_scaled, y_train,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

cv_scores_mse = cross_val_score(
    model_lr, X_train_scaled, y_train,
    cv=5,
    scoring='neg_mean_squared_error',  # négatif car sklearn minimise
    n_jobs=-1
)

cv_scores_mse = -cv_scores_mse  # Remettre en positif

print(f"\n📊 RÉSULTATS CROSS-VALIDATION (5 folds)\n")
print(f"R² Scores : {cv_scores_r2}")
print(f"  Moyenne : {cv_scores_r2.mean():.4f} (± {cv_scores_r2.std():.4f})")

print(f"\nMSE Scores : {cv_scores_mse}")
print(f"  Moyenne : {cv_scores_mse.mean():.2f} (± {cv_scores_mse.std():.2f})")

# Visualisation
plt.figure(figsize=(10, 5))
plt.bar(range(1, 6), cv_scores_r2, alpha=0.7, edgecolor='black')
plt.axhline(cv_scores_r2.mean(), color='r', linestyle='--',
            label=f'Moyenne = {cv_scores_r2.mean():.4f}')
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.title('Cross-Validation : R² par Fold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/regression_cv.png')
plt.show()

print("✓ Graphique sauvegardé : regression_cv.png")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 6 : RÉGULARISATION (RIDGE ET LASSO)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 6 : RÉGULARISATION - RIDGE ET LASSO")
print("="*80)

print("""
🛡️  POURQUOI LA RÉGULARISATION ?
--------------------------------
Problèmes de la régression linéaire classique :
- Overfitting si beaucoup de features
- Multicollinéarité (features corrélées)
- Coefficients instables

SOLUTION : Ajouter un terme de pénalité

1️⃣  RIDGE REGRESSION (L2)
-------------------------
Minimise : MSE + α·Σ(βᵢ²)

PARAMÈTRES :
- alpha : Force de régularisation
  * alpha = 0 : régression linéaire classique
  * alpha petit : peu de régularisation
  * alpha grand : forte régularisation

AVANTAGES :
✓ Réduit les coefficients
✓ Gère la multicollinéarité
✓ Garde toutes les features

INCONVÉNIENTS :
✗ Ne fait pas de sélection de features

2️⃣  LASSO REGRESSION (L1)
--------------------------
Minimise : MSE + α·Σ|βᵢ|

AVANTAGES :
✓ Met certains coefficients à 0 → Sélection de features
✓ Modèle plus simple et interprétable

INCONVÉNIENTS :
✗ Si features corrélées, sélectionne arbitrairement

3️⃣  ELASTICNET (L1 + L2)
-------------------------
Combine Ridge et Lasso
Minimise : MSE + α·(λ·Σ|βᵢ| + (1-λ)·Σ(βᵢ²))

PARAMÈTRES :
- alpha : Force de régularisation
- l1_ratio : Mix L1/L2 (0=Ridge, 1=Lasso, 0.5=équilibre)
""")

# 6.1 RIDGE REGRESSION
print("\n" + "-"*80)
print("6.1 RIDGE REGRESSION (L2)")
print("-"*80)

# Tester différents alpha
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
ridge_results = []

for alpha in alphas:
    # Ridge : Régression linéaire avec pénalité L2
    # - alpha : Force de régularisation
    # - fit_intercept : Calculer l'intercept
    # - solver : Algorithme ('auto', 'svd', 'cholesky', 'lsqr', 'sag')
    model_ridge = Ridge(alpha=alpha, fit_intercept=True)
    model_ridge.fit(X_train_scaled, y_train)

    y_val_pred_ridge = model_ridge.predict(X_val_scaled)
    r2_ridge = r2_score(y_val, y_val_pred_ridge)
    mse_ridge = mean_squared_error(y_val, y_val_pred_ridge)

    ridge_results.append({
        'alpha': alpha,
        'r2': r2_ridge,
        'mse': mse_ridge,
        'model': model_ridge
    })

    print(f"Alpha = {alpha:6.3f} | R² = {r2_ridge:.4f} | MSE = {mse_ridge:.2f}")

# Meilleur alpha
best_ridge = max(ridge_results, key=lambda x: x['r2'])
print(f"\n✓ Meilleur alpha Ridge : {best_ridge['alpha']}")
print(f"  R² : {best_ridge['r2']:.4f}")

# 6.2 LASSO REGRESSION
print("\n" + "-"*80)
print("6.2 LASSO REGRESSION (L1)")
print("-"*80)

lasso_results = []

for alpha in alphas:
    # Lasso : Régression linéaire avec pénalité L1
    # - alpha : Force de régularisation
    # - max_iter : Nombre max d'itérations
    # - tol : Tolérance pour l'arrêt
    model_lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    model_lasso.fit(X_train_scaled, y_train)

    y_val_pred_lasso = model_lasso.predict(X_val_scaled)
    r2_lasso = r2_score(y_val, y_val_pred_lasso)
    mse_lasso = mean_squared_error(y_val, y_val_pred_lasso)

    # Compter features sélectionnées (coef != 0)
    n_features_selected = np.sum(model_lasso.coef_ != 0)

    lasso_results.append({
        'alpha': alpha,
        'r2': r2_lasso,
        'mse': mse_lasso,
        'n_features': n_features_selected,
        'model': model_lasso
    })

    print(f"Alpha = {alpha:6.3f} | R² = {r2_lasso:.4f} | Features = {n_features_selected}/{X.shape[1]}")

# Meilleur alpha
best_lasso = max(lasso_results, key=lambda x: x['r2'])
print(f"\n✓ Meilleur alpha Lasso : {best_lasso['alpha']}")
print(f"  R² : {best_lasso['r2']:.4f}")
print(f"  Features sélectionnées : {best_lasso['n_features']}/{X.shape[1]}")

# 6.3 Comparaison des coefficients
print("\n📊 Comparaison des coefficients...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Linear Regression
axes[0].bar(range(len(model_lr.coef_)), model_lr.coef_, alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Feature Index')
axes[0].set_ylabel('Coefficient')
axes[0].set_title('Linear Regression')
axes[0].grid(True, alpha=0.3)

# Ridge
axes[1].bar(range(len(best_ridge['model'].coef_)), best_ridge['model'].coef_,
            alpha=0.7, edgecolor='black', color='orange')
axes[1].set_xlabel('Feature Index')
axes[1].set_ylabel('Coefficient')
axes[1].set_title(f'Ridge (α={best_ridge["alpha"]})')
axes[1].grid(True, alpha=0.3)

# Lasso
axes[2].bar(range(len(best_lasso['model'].coef_)), best_lasso['model'].coef_,
            alpha=0.7, edgecolor='black', color='green')
axes[2].set_xlabel('Feature Index')
axes[2].set_ylabel('Coefficient')
axes[2].set_title(f'Lasso (α={best_lasso["alpha"]})')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/regression_coefficients.png')
plt.show()

print("✓ Graphique sauvegardé : regression_coefficients.png")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 7 : ÉVALUATION FINALE SUR TEST SET
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 7 : ÉVALUATION FINALE SUR TEST SET")
print("="*80)

print("""
🎯 ÉVALUATION FINALE
--------------------
Le test set n'a JAMAIS été vu par le modèle pendant l'entraînement.
Il sert à évaluer la performance réelle en production.

⚠️ IMPORTANT : Évaluer une seule fois sur le test set !
Si on l'utilise plusieurs fois, il devient un validation set.
""")

# Évaluer tous les modèles sur le test set
models = {
    'Linear Regression': model_lr,
    'Ridge': best_ridge['model'],
    'Lasso': best_lasso['model']
}

print("\n📊 RÉSULTATS SUR TEST SET\n")
print("-" * 60)
print(f"{'Modèle':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
print("-" * 60)

test_results = {}

for name, model in models.items():
    y_test_pred = model.predict(X_test_scaled)

    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    test_results[name] = {
        'r2': test_r2,
        'rmse': test_rmse,
        'mae': test_mae
    }

    print(f"{name:<20} {test_r2:<10.4f} {test_rmse:<10.2f} {test_mae:<10.2f}")

print("-" * 60)

# Meilleur modèle
best_model_name = max(test_results, key=lambda k: test_results[k]['r2'])
print(f"\n🏆 Meilleur modèle : {best_model_name}")
print(f"   R² = {test_results[best_model_name]['r2']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 8 : INTERPRÉTATION ET DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 8 : INTERPRÉTATION")
print("="*80)

best_model = models[best_model_name]
y_test_pred = best_model.predict(X_test_scaled)

# 8.1 Analyse des résidus
print("\n📊 Analyse des résidus...\n")

residuals_test = y_test - y_test_pred

print(f"Statistiques des résidus :")
print(f"  Moyenne : {residuals_test.mean():.4f} (devrait être proche de 0)")
print(f"  Écart-type : {residuals_test.std():.2f}")
print(f"  Min : {residuals_test.min():.2f}")
print(f"  Max : {residuals_test.max():.2f}")

# Visualisation complète
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Prédictions vs Réalité
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k')
axes[0, 0].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                'r--', lw=2)
axes[0, 0].set_xlabel('Valeurs Réelles')
axes[0, 0].set_ylabel('Prédictions')
axes[0, 0].set_title('Prédictions vs Réalité')
axes[0, 0].grid(True, alpha=0.3)

# 2. Résidus vs Prédictions
axes[0, 1].scatter(y_test_pred, residuals_test, alpha=0.6, edgecolors='k')
axes[0, 1].axhline(0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Prédictions')
axes[0, 1].set_ylabel('Résidus')
axes[0, 1].set_title('Résidus vs Prédictions')
axes[0, 1].grid(True, alpha=0.3)

# 3. Distribution des résidus
axes[1, 0].hist(residuals_test, bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Résidus')
axes[1, 0].set_ylabel('Fréquence')
axes[1, 0].set_title('Distribution des Résidus')
axes[1, 0].grid(True, alpha=0.3)

# 4. Q-Q plot
from scipy import stats
stats.probplot(residuals_test, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normalité des résidus)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/regression_diagnostics.png')
plt.show()

print("✓ Graphique sauvegardé : regression_diagnostics.png")

# 8.2 Importance des features
print("\n📊 Importance des features (coefficients)...\n")

if hasattr(best_model, 'coef_'):
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': best_model.coef_,
        'Abs_Coefficient': np.abs(best_model.coef_)
    })
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

    print(feature_importance)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
    plt.xlabel('Coefficient')
    plt.title(f'Importance des Features - {best_model_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/regression_feature_importance.png')
    plt.show()

    print("✓ Graphique sauvegardé : regression_feature_importance.png")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 9 : SAUVEGARDE DU MODÈLE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 9 : SAUVEGARDE")
print("="*80)

import joblib

# Sauvegarder le modèle et le scaler
joblib.dump(best_model, 'e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/best_regression_model.pkl')
joblib.dump(scaler, 'e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/scaler.pkl')

print("✓ Modèle sauvegardé : best_regression_model.pkl")
print("✓ Scaler sauvegardé : scaler.pkl")

print("""
📦 UTILISATION DU MODÈLE SAUVEGARDÉ
-----------------------------------
Pour charger et utiliser le modèle :

```python
import joblib
import numpy as np

# Charger
model = joblib.load('best_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prédire sur nouvelles données
X_new = np.array([[...]])  # Vos nouvelles données
X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
```
""")

# ═══════════════════════════════════════════════════════════════════════════
# RÉSUMÉ ET CONCLUSIONS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🎉 RÉSUMÉ ET CONCLUSIONS")
print("="*80)

print(f"""
📊 RÉSULTATS FINAUX
-------------------
Meilleur modèle : {best_model_name}
  R² Score : {test_results[best_model_name]['r2']:.4f}
  RMSE : {test_results[best_model_name]['rmse']:.2f}
  MAE : {test_results[best_model_name]['mae']:.2f}

📈 PERFORMANCE
--------------
- R² proche de 1 : Excellent modèle
- R² > 0.7 : Bon modèle
- R² > 0.5 : Modèle acceptable
- R² < 0.5 : Modèle faible

🎯 INTERPRÉTATION DU R²
-----------------------
Le modèle explique {test_results[best_model_name]['r2']*100:.1f}% de la variance de la variable cible.

✅ CHECKLIST RÉGRESSION LINÉAIRE
---------------------------------
✓ Données préparées et normalisées
✓ Baseline établie
✓ Modèle entraîné
✓ Cross-validation effectuée
✓ Régularisation testée (Ridge/Lasso)
✓ Évaluation sur test set
✓ Résidus analysés
✓ Modèle sauvegardé

🚀 PROCHAINES ÉTAPES
--------------------
1. Tester sur de nouvelles données réelles
2. Feature engineering avancé
3. Essayer d'autres modèles (Random Forest, XGBoost)
4. Optimiser les hyperparamètres davantage
5. Déployer en production

💡 CONSEILS
-----------
- Toujours vérifier les hypothèses de la régression linéaire
- Analyser les résidus pour diagnostiquer les problèmes
- La normalisation est cruciale pour Ridge/Lasso
- Cross-validation donne une meilleure estimation de la performance
- Ne jamais sur-optimiser sur le test set !
""")

print("="*80)
print("✨ TUTORIEL TERMINÉ AVEC SUCCÈS ! ✨")
print("="*80)
