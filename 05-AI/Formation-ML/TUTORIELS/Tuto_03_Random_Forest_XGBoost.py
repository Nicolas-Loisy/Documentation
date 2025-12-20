"""
═══════════════════════════════════════════════════════════════════════════════
TUTORIEL COMPLET : RANDOM FOREST & XGBOOST - MODÈLES ENSEMBLE
═══════════════════════════════════════════════════════════════════════════════

🎯 CAS D'USAGE RÉEL : Prédiction de Churn (Désabonnement Client)

CONTEXTE :
Une entreprise télécoms veut prédire quels clients vont se désabonner (churn).
Objectif : Identifier clients à risque pour actions de rétention ciblées.

POURQUOI RANDOM FOREST & XGBOOST ?
- Performance MAXIMALE sur données tabulaires
- Standard de l'industrie (Kaggle, production)
- Robustes, gèrent bien données bruitées
- Peu de preprocessing nécessaire

Ce tutoriel couvre :
1. POURQUOI Random Forest et XGBoost dominent le ML industriel
2. QUAND utiliser RF vs XGBoost vs LightGBM
3. Random Forest : principe, hyperparamètres, optimisation
4. XGBoost : principe, hyperparamètres, optimisation
5. Comparaison détaillée et choix du meilleur
6. Diagnostic avancé (learning curves, overfitting)
7. Feature engineering et importance
8. Optimisation poussée (Grid Search, Random Search)

Chaque étape explique CE QU'IL FAUT OBSERVER et LES CONCLUSIONS.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    RandomizedSearchCV, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost non installé. Installez avec : pip install xgboost")
    XGBOOST_AVAILABLE = False

# LightGBM
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️  LightGBM non installé. Installez avec : pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("TUTORIEL : RANDOM FOREST & XGBOOST - MODÈLES ENSEMBLE")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 1 : COMPRENDRE LES MODÈLES ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 1 : POURQUOI RANDOM FOREST ET XGBOOST DOMINENT LE ML ?")
print("="*80)

print("""
🏆 LES ROIS DU ML SUR DONNÉES TABULAIRES
─────────────────────────────────────────
Random Forest et XGBoost sont les modèles les PLUS UTILISÉS en production
et dominent les compétitions Kaggle pour données tabulaires.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 PRINCIPE DES MODÈLES ENSEMBLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 IDÉE CENTRALE : "L'union fait la force"
   Combiner plusieurs modèles FAIBLES → Modèle FORT

ANALOGIE : Jury vs Juge unique
   - 1 juge peut se tromper
   - 10 juges votent → décision plus robuste

1️⃣  RANDOM FOREST (Bagging)
────────────────────────────
PRINCIPE : Moyenne de Decision Trees indépendants

┌─────────┐  ┌─────────┐  ┌─────────┐       ┌──────────┐
│ Tree 1  │  │ Tree 2  │  │ Tree N  │  ───→ │  VOTE    │ ───→ Prédiction
│ (vote)  │  │ (vote)  │  │ (vote)  │       │ Majorité │
└─────────┘  └─────────┘  └─────────┘       └──────────┘

ÉTAPES :
1. Créer N arbres de décision
2. Chaque arbre entraîné sur échantillon aléatoire (bootstrap)
3. Chaque split utilise sous-ensemble aléatoire de features
4. Prédiction finale = vote majoritaire (classification)

AVANTAGES :
✅ Réduit OVERFITTING (chaque arbre overfit différemment)
✅ Robuste au bruit
✅ Parallélisable (arbres indépendants)
✅ Peu sensible aux hyperparamètres
✅ Pas besoin de normalisation

INCONVÉNIENTS :
❌ Moins performant que Boosting sur données propres
❌ Modèle "boîte noire" (moins interprétable qu'arbre unique)
❌ Lent à prédire (doit évaluer tous les arbres)


2️⃣  XGBOOST (Gradient Boosting)
─────────────────────────────────
PRINCIPE : Arbres séquentiels corrigeant erreurs précédentes

Tree 1 ───→ Erreurs 1 ───→ Tree 2 ───→ Erreurs 2 ───→ Tree 3 ───→ ...
(prédiction)  (focus sur    (corrige      (focus sur    (corrige
               erreurs)      erreurs 1)    erreurs 2)    erreurs 2)

ÉTAPES :
1. Entraîner arbre sur données
2. Calculer résidus (erreurs)
3. Entraîner nouvel arbre pour prédire résidus
4. Ajouter prédiction pondérée au modèle
5. Répéter jusqu'à convergence ou max_trees

AVANTAGES :
✅ Performance MAXIMALE sur données tabulaires
✅ Régularisation intégrée (L1, L2, Gamma)
✅ Gère valeurs manquantes nativement
✅ Gère classes déséquilibrées (scale_pos_weight)
✅ Supporte GPU

INCONVÉNIENTS :
❌ Plus sensible aux hyperparamètres (tuning nécessaire)
❌ Risque d'overfitting si mal configuré
❌ Non parallélisable (arbres séquentiels)
❌ Plus lent à entraîner que Random Forest


3️⃣  LIGHTGBM (Gradient Boosting Optimisé)
───────────────────────────────────────────
PRINCIPE : Boosting optimisé pour grandes données

AVANTAGES vs XGBoost :
✅ Plus RAPIDE (algorithme GOSS + EFB)
✅ Moins de mémoire
✅ Meilleur sur grandes données (>10k lignes)
✅ Gère catégories nativement

INCONVÉNIENTS :
❌ Peut overfitter sur petites données
❌ Plus sensible aux hyperparamètres


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 TABLEAU DE DÉCISION : QUAND UTILISER QUOI ?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌────────────────────┬────────────────┬────────────────┬────────────────┐
│  Critère           │  Random Forest │  XGBoost       │  LightGBM      │
├────────────────────┼────────────────┼────────────────┼────────────────┤
│ Performance        │ ★★★★☆          │ ★★★★★          │ ★★★★★          │
│ Vitesse entraîn.   │ ★★★★★          │ ★★★☆☆          │ ★★★★★          │
│ Vitesse inférence  │ ★★☆☆☆          │ ★★★☆☆          │ ★★★★☆          │
│ Robustesse         │ ★★★★★          │ ★★★★☆          │ ★★★☆☆          │
│ Facilité tuning    │ ★★★★★          │ ★★★☆☆          │ ★★☆☆☆          │
│ Grandes données    │ ★★★☆☆          │ ★★★★☆          │ ★★★★★          │
│ Interprétabilité   │ ★★★☆☆          │ ★★★☆☆          │ ★★★☆☆          │
└────────────────────┴────────────────┴────────────────┴────────────────┘

🎯 RECOMMANDATIONS :

UTILISER RANDOM FOREST QUAND :
✅ Première approche (baseline robuste)
✅ Peu de temps pour tuning
✅ Données bruitées
✅ Peu de données (<10k)
✅ Besoin de parallélisation
✅ Feature importance simple requise

UTILISER XGBOOST QUAND :
✅ Performance maximale requise
✅ Compétition Kaggle
✅ Données moyennes (10k-1M)
✅ Temps pour tuning disponible
✅ Classes déséquilibrées

UTILISER LIGHTGBM QUAND :
✅ Grandes données (>100k)
✅ Vitesse critique
✅ Features catégorielles nombreuses
✅ Ressources limitées (mémoire)

💼 CAS D'USAGE TYPIQUES :
- Prédiction de churn (notre cas !) → XGBoost ou LightGBM
- Scoring de crédit → XGBoost
- Détection de fraude → XGBoost
- Recommandation → XGBoost ou LightGBM
- Forecast de ventes → XGBoost
""")

input("\n▶ Appuyez sur Entrée pour continuer...")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 2 : PRÉPARATION DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 2 : DONNÉES - PRÉDICTION DE CHURN")
print("="*80)

# 2.1 Génération de données
print("\n📊 Génération de données synthétiques (simulant clients télécoms)...\n")

# Données plus complexes pour montrer puissance des modèles ensemble
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=15,
    n_redundant=3,
    n_classes=2,
    weights=[0.70, 0.30],  # 70% rétention, 30% churn
    flip_y=0.03,
    random_state=42
)

print(f"✓ Données générées : {X.shape[0]} clients × {X.shape[1]} features")
print(f"✓ Distribution : {np.bincount(y)[0]} rétention ({np.bincount(y)[0]/len(y)*100:.1f}%), "
      f"{np.bincount(y)[1]} churn ({np.bincount(y)[1]/len(y)*100:.1f}%)")

# Features réalistes
feature_names = [
    'Tenure', 'MonthlyCharges', 'TotalCharges', 'ContractLength',
    'DataUsage', 'CallMinutes', 'SMSCount', 'CustomerServiceCalls',
    'PaymentMethod', 'AutoPay', 'PaperlessBilling', 'TechSupport',
    'OnlineSecurity', 'DeviceProtection', 'StreamingTV', 'StreamingMovies',
    'Age', 'DependentsCount', 'PartnerStatus', 'SeniorCitizen'
]

df = pd.DataFrame(X, columns=feature_names)
df['Churn'] = y

print("\n📈 Aperçu :")
print(df.head())

# 2.2 Split des données
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\n✓ Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

print("""
💡 QUESTION : Faut-il normaliser pour RF et XGBoost ?
──────────────────────────────────────────────────────
RÉPONSE : NON !

Random Forest et XGBoost sont basés sur ARBRES DE DÉCISION.
Les arbres splitent sur SEUILS, pas magnitudes.
→ Normalisation N'APPORTE RIEN (parfois même nuit)

EXCEPTION : Si vous voulez comparer coefficients/importance entre features
sur mêmes échelles → alors normaliser.

Pour ce tutoriel : PAS de normalisation (inutile).
""")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 3 : RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 3 : RANDOM FOREST")
print("="*80)

print("""
🌲 HYPERPARAMÈTRES CLÉS DE RANDOM FOREST
─────────────────────────────────────────

1. n_estimators : Nombre d'arbres
   - Plus grand = Meilleur (jusqu'à plateau)
   - Défaut : 100, Recommandé : 200-500
   - ⚠️  Plus lent avec beaucoup d'arbres

2. max_depth : Profondeur max de chaque arbre
   - None = Pas de limite (risque overfitting)
   - Recommandé : 10-30
   - Contrôle complexité

3. min_samples_split : Min échantillons pour split
   - Défaut : 2, Recommandé : 10-20
   - Plus grand = Moins de splits = Moins d'overfitting

4. min_samples_leaf : Min échantillons dans feuille
   - Défaut : 1, Recommandé : 5-10
   - Évite feuilles trop spécifiques

5. max_features : Nb features par split
   - 'sqrt' : √n_features (défaut, recommandé)
   - 'log2' : log2(n_features)
   - Contrôle diversité des arbres

6. class_weight : Gestion déséquilibre
   - 'balanced' : Pénalise classe minoritaire
   - Utile pour churn (classe minoritaire importante)

7. n_jobs : Parallélisation
   - -1 : Utiliser tous les CPU
   - Accélère beaucoup l'entraînement
""")

# 3.1 Random Forest de base
print("\n🚀 Entraînement Random Forest (baseline)...\n")

rf_baseline = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_baseline.fit(X_train, y_train)

y_val_pred_rf_base = rf_baseline.predict(X_val)
y_val_proba_rf_base = rf_baseline.predict_proba(X_val)[:, 1]

# Métriques
acc_rf_base = accuracy_score(y_val, y_val_pred_rf_base)
f1_rf_base = f1_score(y_val, y_val_pred_rf_base)
auc_rf_base = roc_auc_score(y_val, y_val_proba_rf_base)

print(f"Random Forest (baseline) :")
print(f"  Accuracy : {acc_rf_base:.4f}")
print(f"  F1-Score : {f1_rf_base:.4f}")
print(f"  ROC-AUC  : {auc_rf_base:.4f}")

# 3.2 Random Forest optimisé
print("\n🚀 Entraînement Random Forest (optimisé)...\n")

rf_optimized = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=15,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_optimized.fit(X_train, y_train)

y_val_pred_rf_opt = rf_optimized.predict(X_val)
y_val_proba_rf_opt = rf_optimized.predict_proba(X_val)[:, 1]

acc_rf_opt = accuracy_score(y_val, y_val_pred_rf_opt)
f1_rf_opt = f1_score(y_val, y_val_pred_rf_opt)
auc_rf_opt = roc_auc_score(y_val, y_val_proba_rf_opt)

print(f"Random Forest (optimisé) :")
print(f"  Accuracy : {acc_rf_opt:.4f}")
print(f"  F1-Score : {f1_rf_opt:.4f}")
print(f"  ROC-AUC  : {auc_rf_opt:.4f}")

print(f"""
🔍 OBSERVATION #1 : Impact de l'optimisation
─────────────────────────────────────────────
Baseline → Optimisé :
  F1-Score : {f1_rf_base:.4f} → {f1_rf_opt:.4f} (Δ = {f1_rf_opt - f1_rf_base:+.4f})
  ROC-AUC  : {auc_rf_base:.4f} → {auc_rf_opt:.4f} (Δ = {auc_rf_opt - auc_rf_base:+.4f})

CE QU'IL FAUT OBSERVER :

1. Amélioration significative ? (Δ > 0.02)
   → Oui : Tuning a porté ses fruits
   → Non : Hyperparamètres par défaut suffisants

2. Amélioration sur TOUTES les métriques ?
   → Oui : Optimisation robuste
   → Non : Trade-off possible (ex: F1 ↑ mais Accuracy ↓)

💡 CONCLUSION :
   Random Forest est ROBUSTE : même baseline donne souvent bons résultats.
   Optimisation apporte amélioration MODÉRÉE (pas spectaculaire).

   Si amélioration < 0.01 : Gain marginal, baseline suffit.
   Si amélioration > 0.05 : Optimisation cruciale !
""")

# 3.3 Diagnostic overfitting
print("\n🔍 Diagnostic d'overfitting...\n")

y_train_pred_rf = rf_optimized.predict(X_train)
train_acc_rf = accuracy_score(y_train, y_train_pred_rf)
train_f1_rf = f1_score(y_train, y_train_pred_rf)

print(f"Train Accuracy : {train_acc_rf:.4f}")
print(f"Val Accuracy   : {acc_rf_opt:.4f}")
print(f"Écart          : {train_acc_rf - acc_rf_opt:.4f}")

print(f"""
🔍 OBSERVATION #2 : Overfitting de Random Forest
─────────────────────────────────────────────────
Écart Train - Val : {train_acc_rf - acc_rf_opt:.4f}

CE QU'IL FAUT OBSERVER :

1. Écart < 0.05 : EXCELLENT (pas d'overfitting)
   → RF est robuste, généralise bien

2. Écart 0.05-0.10 : ACCEPTABLE
   → Léger overfitting, mais gérable

3. Écart > 0.10 : PROBLÈME
   → Overfitting significatif
   → Actions : Augmenter min_samples_split/leaf, réduire max_depth

💡 CONCLUSION ATTENDUE :
   Random Forest overfit RAREMENT grâce au bagging.
   Écart devrait être faible (~0.02-0.05).

   Si overfitting fort :
   1. Réduire max_depth (20 → 15)
   2. Augmenter min_samples_leaf (5 → 10)
   3. Augmenter min_samples_split (15 → 30)
""")

# 3.4 Feature Importance
print("\n📊 Feature Importance (Random Forest)...\n")

importance_rf = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_optimized.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance_rf.head(10))

plt.figure(figsize=(12, 8))
plt.barh(importance_rf['Feature'][:15], importance_rf['Importance'][:15], color='green', alpha=0.7)
plt.xlabel('Importance')
plt.title('Top 15 Features - Random Forest')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/ensemble_rf_importance.png', dpi=100)
plt.show()

print("\n✓ Graphique sauvegardé : ensemble_rf_importance.png")

print(f"""
🔍 OBSERVATION #3 : Feature Importance (Interprétation)
────────────────────────────────────────────────────────
CE QU'IL FAUT OBSERVER :

1. Features dominantes (Importance > 0.10) :
   Ces features sont CRITIQUES pour les prédictions.
   → Prioriser leur qualité (collecte, nettoyage)

2. Features négligeables (Importance < 0.01) :
   Ces features n'apportent RIEN.
   → Candidats pour suppression (réduire dimensionnalité)

3. Distribution de l'importance :
   - Plate (toutes ≈ égales) : Toutes features utiles
   - Concentrée (top 3-5 dominent) : Peu de features vraiment importantes

💡 UTILITÉ BUSINESS :
   Top features = Leviers d'action pour réduire churn

   Ex: Si "CustomerServiceCalls" est top 1 :
   → Améliorer service client peut réduire churn significativement

⚠️  ATTENTION :
   Feature Importance ≠ Causalité
   Corrélation n'implique pas causalité !

   Ex: "TotalCharges" important peut signifier :
   - Clients chers partent plus → Réduire prix ?
   - Clients fidèles (charges élevées) partent moins → Fidéliser ?

   → Analyse métier nécessaire pour interpréter correctement
""")

input("\n▶ Appuyez sur Entrée pour continuer vers XGBoost...")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 4 : XGBOOST
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 4 : XGBOOST - LE CHAMPION DE KAGGLE")
print("="*80)

if not XGBOOST_AVAILABLE:
    print("\n⚠️  XGBoost non disponible. Installation : pip install xgboost")
    print("Suite du tutoriel avec Random Forest uniquement.")
else:
    print("""
🏆 HYPERPARAMÈTRES CLÉS DE XGBOOST
───────────────────────────────────

PARAMÈTRES DE BASE :

1. n_estimators : Nombre d'arbres (boosting rounds)
   - Plus grand = Meilleur (jusqu'à plateau)
   - Défaut : 100, Recommandé : 100-1000
   - Utiliser early_stopping pour trouver optimal

2. learning_rate (eta) : Taux d'apprentissage
   - Contrôle contribution de chaque arbre
   - Défaut : 0.3, Recommandé : 0.01-0.1
   - Plus petit = Plus d'arbres nécessaires mais meilleure généralisation

3. max_depth : Profondeur max arbres
   - Défaut : 6, Recommandé : 3-10
   - Plus grand = Plus complexe = Risque overfitting

PARAMÈTRES DE RÉGULARISATION :

4. subsample : Fraction d'échantillons par arbre
   - Défaut : 1.0, Recommandé : 0.7-0.9
   - Comme bagging dans RF

5. colsample_bytree : Fraction de features par arbre
   - Défaut : 1.0, Recommandé : 0.7-0.9
   - Réduit overfitting

6. reg_alpha (L1) : Régularisation L1
   - Défaut : 0, Recommandé : 0-1
   - Sélection de features

7. reg_lambda (L2) : Régularisation L2
   - Défaut : 1, Recommandé : 1-10
   - Réduit overfitting

PARAMÈTRES POUR DÉSÉQUILIBRE :

8. scale_pos_weight : Poids classe minoritaire
   - Défaut : 1, Recommandé : ratio_majority/ratio_minority
   - Crucial pour classes déséquilibrées

9. eval_metric : Métrique à optimiser
   - 'logloss' : Log loss (défaut)
   - 'auc' : ROC-AUC
   - 'aucpr' : Precision-Recall AUC

💡 STRATÉGIE D'OPTIMISATION :
   1. Fixer learning_rate petit (0.05)
   2. Trouver n_estimators optimal (early_stopping)
   3. Optimiser max_depth, subsample, colsample
   4. Ajouter régularisation si overfitting
""")

    # 4.1 XGBoost baseline
    print("\n🚀 Entraînement XGBoost (baseline)...\n")

    xgb_baseline = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_baseline.fit(X_train, y_train)

    y_val_pred_xgb_base = xgb_baseline.predict(X_val)
    y_val_proba_xgb_base = xgb_baseline.predict_proba(X_val)[:, 1]

    acc_xgb_base = accuracy_score(y_val, y_val_pred_xgb_base)
    f1_xgb_base = f1_score(y_val, y_val_pred_xgb_base)
    auc_xgb_base = roc_auc_score(y_val, y_val_proba_xgb_base)

    print(f"XGBoost (baseline) :")
    print(f"  Accuracy : {acc_xgb_base:.4f}")
    print(f"  F1-Score : {f1_xgb_base:.4f}")
    print(f"  ROC-AUC  : {auc_xgb_base:.4f}")

    # 4.2 XGBoost avec early stopping
    print("\n🚀 Entraînement XGBoost (avec early stopping)...\n")

    # scale_pos_weight pour déséquilibre
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_early_stop = XGBClassifier(
        n_estimators=1000,  # Grand nombre car early stopping
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # early_stopping_rounds : Arrête si pas d'amélioration pendant N rounds
    xgb_early_stop.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    y_val_pred_xgb_es = xgb_early_stop.predict(X_val)
    y_val_proba_xgb_es = xgb_early_stop.predict_proba(X_val)[:, 1]

    acc_xgb_es = accuracy_score(y_val, y_val_pred_xgb_es)
    f1_xgb_es = f1_score(y_val, y_val_pred_xgb_es)
    auc_xgb_es = roc_auc_score(y_val, y_val_proba_xgb_es)

    print(f"XGBoost (early stopping) :")
    print(f"  Best iteration : {xgb_early_stop.best_iteration}")
    print(f"  Accuracy : {acc_xgb_es:.4f}")
    print(f"  F1-Score : {f1_xgb_es:.4f}")
    print(f"  ROC-AUC  : {auc_xgb_es:.4f}")

    print(f"""
🔍 OBSERVATION #4 : Early Stopping
───────────────────────────────────
Best iteration : {xgb_early_stop.best_iteration} / 1000

CE QU'IL FAUT OBSERVER :

1. Best iteration << n_estimators (ex: 150 / 1000)
   → Convergence RAPIDE
   → Peut augmenter learning_rate ou réduire n_estimators

2. Best iteration ≈ n_estimators (ex: 950 / 1000)
   → Convergence LENTE ou pas atteinte
   → Réduire learning_rate ou augmenter n_estimators

3. Best iteration modéré (ex: 300-500 / 1000)
   → OPTIMAL : équilibre trouvé

💡 CONCLUSION :
   Early stopping est ESSENTIEL pour XGBoost.
   Évite overfitting ET trouve automatiquement nombre optimal d'arbres.

   Sans early stopping : Risque élevé d'overfitting après N iterations.

⚙️  AJUSTEMENTS POSSIBLES :
   - Si best_iteration < 100 : Augmenter learning_rate (0.05 → 0.1)
   - Si best_iteration > 800 : Diminuer learning_rate (0.05 → 0.03)
   - early_stopping_rounds = 50 : Standard, peut ajuster (20-100)
""")

    # 4.3 XGBoost optimisé (Grid Search)
    print("\n🔧 Optimisation XGBoost (Grid Search)...\n")
    print("⏳ Cela peut prendre quelques minutes...")

    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }

    xgb_grid = XGBClassifier(
        n_estimators=500,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=50
    )

    grid_search = GridSearchCV(
        xgb_grid,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    print(f"\n✓ Optimisation terminée")
    print(f"Meilleurs paramètres : {grid_search.best_params_}")
    print(f"Meilleur F1-Score (CV) : {grid_search.best_score_:.4f}")

    xgb_optimized = grid_search.best_estimator_

    y_val_pred_xgb_opt = xgb_optimized.predict(X_val)
    y_val_proba_xgb_opt = xgb_optimized.predict_proba(X_val)[:, 1]

    acc_xgb_opt = accuracy_score(y_val, y_val_pred_xgb_opt)
    f1_xgb_opt = f1_score(y_val, y_val_pred_xgb_opt)
    auc_xgb_opt = roc_auc_score(y_val, y_val_proba_xgb_opt)

    print(f"\nXGBoost (optimisé) :")
    print(f"  Accuracy : {acc_xgb_opt:.4f}")
    print(f"  F1-Score : {f1_xgb_opt:.4f}")
    print(f"  ROC-AUC  : {auc_xgb_opt:.4f}")

    print(f"""
🔍 OBSERVATION #5 : Impact du Grid Search
──────────────────────────────────────────
Early Stop → Grid Search :
  F1-Score : {f1_xgb_es:.4f} → {f1_xgb_opt:.4f} (Δ = {f1_xgb_opt - f1_xgb_es:+.4f})

CE QU'IL FAUT OBSERVER :

1. Amélioration > 0.02 : Grid Search UTILE
   → Continuer avec Random Search ou Bayesian Optimization

2. Amélioration < 0.01 : Gain MARGINAL
   → Early stopping suffisait
   → Économiser temps : utiliser hyperparamètres par défaut

3. Dégradation (Δ < 0) : Sur-optimisation sur CV
   → Risque d'overfitting
   → Vérifier sur test set

💡 CONCLUSION :
   Grid Search améliore XGBoost MODÉRÉMENT (gain ~1-3%).
   Sur grandes données : Privilégier Random Search (plus rapide).

🎯 TRADE-OFF Temps vs Performance :
   - Baseline : 10 secondes, F1 = {f1_xgb_base:.4f}
   - Early Stop : 30 secondes, F1 = {f1_xgb_es:.4f}
   - Grid Search : 5-10 min, F1 = {f1_xgb_opt:.4f}

   → Early stopping = Meilleur compromis temps/performance !
""")

    # 4.4 Feature Importance XGBoost
    print("\n📊 Feature Importance (XGBoost)...\n")

    importance_xgb = pd.DataFrame({
        'Feature': feature_names,
        'Importance': xgb_optimized.feature_importances_
    }).sort_values('Importance', ascending=False)

    print(importance_xgb.head(10))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # XGBoost
    axes[0].barh(importance_xgb['Feature'][:15], importance_xgb['Importance'][:15],
                 color='orange', alpha=0.7)
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Feature Importance - XGBoost')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')

    # Random Forest (comparaison)
    axes[1].barh(importance_rf['Feature'][:15], importance_rf['Importance'][:15],
                 color='green', alpha=0.7)
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Feature Importance - Random Forest')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/ensemble_importance_comparison.png', dpi=100)
    plt.show()

    print("\n✓ Graphique sauvegardé : ensemble_importance_comparison.png")

    print(f"""
🔍 OBSERVATION #6 : RF vs XGBoost Feature Importance
─────────────────────────────────────────────────────
DIFFÉRENCES POSSIBLES :

1. XGBoost : Gain moyen de la feature dans les splits
   → Mesure contribution directe à la réduction de loss

2. Random Forest : Réduction moyenne d'impureté
   → Mesure contribution à la "pureté" des splits

CE QU'IL FAUT OBSERVER :

• Rankings similaires ?
  → Accord entre modèles → Features vraiment importantes
  → Confiance élevée dans les insights

• Rankings divergents ?
  → Modèles capturent aspects différents
  → RF : Interactions locales
  → XGBoost : Corrections séquentielles

  → Analyser les deux pour compréhension complète

💡 UTILITÉ :
   Features importantes dans LES DEUX modèles :
   → Leviers d'action les plus fiables

   Features importantes seulement dans un modèle :
   → Interactions complexes possibles
   → Investiguer pourquoi cette feature est importante ici mais pas là

📊 EXEMPLE D'INTERPRÉTATION :
   Si "CustomerServiceCalls" top 1 dans les deux :
   → Lever d'action CRITIQUE pour réduire churn
   → Investir dans amélioration service client

   Si "Tenure" top dans RF mais pas XGBoost :
   → RF capte mieux effet non-linéaire de l'ancienneté
   → XGBoost compense via autres features
""")

    input("\n▶ Appuyez sur Entrée pour voir la comparaison finale...")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 5 : COMPARAISON FINALE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 5 : COMPARAISON FINALE - QUEL MODÈLE CHOISIR ?")
print("="*80)

# Tableau récapitulatif
if XGBOOST_AVAILABLE:
    results = pd.DataFrame({
        'Modèle': [
            'RF Baseline',
            'RF Optimisé',
            'XGBoost Baseline',
            'XGBoost Early Stop',
            'XGBoost Optimisé'
        ],
        'Accuracy': [acc_rf_base, acc_rf_opt, acc_xgb_base, acc_xgb_es, acc_xgb_opt],
        'F1-Score': [f1_rf_base, f1_rf_opt, f1_xgb_base, f1_xgb_es, f1_xgb_opt],
        'ROC-AUC': [auc_rf_base, auc_rf_opt, auc_xgb_base, auc_xgb_es, auc_xgb_opt]
    })
else:
    results = pd.DataFrame({
        'Modèle': ['RF Baseline', 'RF Optimisé'],
        'Accuracy': [acc_rf_base, acc_rf_opt],
        'F1-Score': [f1_rf_base, f1_rf_opt],
        'ROC-AUC': [auc_rf_base, auc_rf_opt]
    })

print("\n📊 TABLEAU RÉCAPITULATIF\n")
print(results.to_string(index=False))

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, metric in enumerate(['Accuracy', 'F1-Score', 'ROC-AUC']):
    axes[idx].bar(results['Modèle'], results[metric], alpha=0.7, edgecolor='black')
    axes[idx].set_ylabel(metric)
    axes[idx].set_title(f'Comparaison : {metric}')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(True, alpha=0.3, axis='y')
    axes[idx].set_ylim(results[metric].min() - 0.05, results[metric].max() + 0.05)

plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/ensemble_comparison.png', dpi=100)
plt.show()

print("\n✓ Graphique sauvegardé : ensemble_comparison.png")

# Meilleur modèle
best_model_name = results.loc[results['F1-Score'].idxmax(), 'Modèle']
best_f1 = results['F1-Score'].max()
best_auc = results.loc[results['F1-Score'].idxmax(), 'ROC-AUC']

if XGBOOST_AVAILABLE:
    best_model = xgb_optimized if 'XGBoost' in best_model_name else rf_optimized
else:
    best_model = rf_optimized

print(f"""
🔍 OBSERVATION FINALE : Choix du Modèle
────────────────────────────────────────
CE QU'IL FAUT OBSERVER :

1. PERFORMANCE :
   Meilleur F1-Score : {best_model_name} = {best_f1:.4f}

2. DIFFÉRENCES ENTRE MODÈLES :
   - Si Δ < 0.01 : Performance ÉQUIVALENTE
     → Choisir le plus SIMPLE (RF baseline)

   - Si Δ 0.01-0.03 : Amélioration MODÉRÉE
     → Trade-off complexité vs gain

   - Si Δ > 0.03 : Amélioration SIGNIFICATIVE
     → Choisir le meilleur

3. CONSIDÉRATIONS PRATIQUES :
   ┌──────────────────┬────────────┬─────────────┐
   │ Critère          │ RF         │ XGBoost     │
   ├──────────────────┼────────────┼─────────────┤
   │ Entraînement     │ Rapide     │ Plus lent   │
   │ Inférence        │ Lent       │ Rapide      │
   │ Tuning           │ Facile     │ Complexe    │
   │ Interprétation   │ Facile     │ Moyenne     │
   │ Robustesse       │ Excellente │ Bonne       │
   └──────────────────┴────────────┴─────────────┘
""")

if XGBOOST_AVAILABLE:
    print(f"""
🎯 RECOMMANDATION POUR CHURN
────────────────────────────
Performance : {best_model_name} gagne avec F1 = {best_f1:.4f}

💼 SCÉNARIOS DE DÉCISION :

SCÉNARIO A : Startup / Prototype / Peu de données
   → CHOISIR : Random Forest
   Raisons :
   ✅ Setup rapide
   ✅ Peu de tuning nécessaire
   ✅ Robuste out-of-the-box
   ✅ Facile à maintenir

SCÉNARIO B : Production / Grande échelle / Performance critique
   → CHOISIR : XGBoost
   Raisons :
   ✅ Meilleure performance
   ✅ Inférence plus rapide (important avec millions de clients)
   ✅ Gère mieux déséquilibre (scale_pos_weight)
   ✅ Régularisation avancée

SCÉNARIO C : Très grandes données (>1M lignes)
   → CHOISIR : LightGBM
   Raisons :
   ✅ Plus rapide que XGBoost
   ✅ Moins de mémoire
   ✅ Performances similaires

🏆 CHOIX FINAL : {best_model_name}
   Pour churn télécoms : XGBoost généralement optimal
   (performance + robustesse + scale_pos_weight)
""")
else:
    print(f"""
🎯 RECOMMANDATION
─────────────────
Performance : {best_model_name} avec F1 = {best_f1:.4f}

💼 UTILISATION EN PRODUCTION :
   Random Forest est EXCELLENT pour :
   ✅ Baseline rapide et robuste
   ✅ Applications où vitesse d'entraînement critique
   ✅ Cas où interprétabilité importante

   Pour ENCORE MEILLEURES performances :
   → Installer XGBoost : pip install xgboost
   → Gain attendu : +2-5% sur métriques
""")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 6 : ÉVALUATION FINALE SUR TEST SET
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 6 : ÉVALUATION FINALE SUR TEST SET")
print("="*80)

y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred)
test_rec = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"\n🏆 Modèle sélectionné : {best_model_name}\n")
print(f"📊 PERFORMANCE SUR TEST SET")
print("-" * 60)
print(f"Accuracy  : {test_acc:.4f}")
print(f"Precision : {test_prec:.4f}")
print(f"Recall    : {test_rec:.4f}")
print(f"F1-Score  : {test_f1:.4f}")
print(f"ROC-AUC   : {test_auc:.4f}")

print("\n📋 Classification Report :")
print(classification_report(y_test, y_test_pred, target_names=['Retention', 'Churn']))

# Matrice de confusion
cm_test = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Prédiction')
plt.ylabel('Réalité')
plt.title(f'Matrice de Confusion - Test Set\n{best_model_name}')
plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/ensemble_confusion_final.png', dpi=100)
plt.show()

print("\n✓ Graphique sauvegardé : ensemble_confusion_final.png")

print(f"""
🔍 OBSERVATION #7 : Performance Test vs Val
────────────────────────────────────────────
Val F1  : {best_f1:.4f}
Test F1 : {test_f1:.4f}
Écart   : {abs(test_f1 - best_f1):.4f}

CE QU'IL FAUT OBSERVER :

1. Test ≈ Val (écart < 0.02) : EXCELLENT
   → Modèle GÉNÉRALISE bien
   → Prêt pour production

2. Test < Val (écart 0.02-0.05) : ACCEPTABLE
   → Légère dégradation normale
   → Probablement dû à sur-optimisation sur val

3. Test << Val (écart > 0.05) : PROBLÈME
   → Overfitting sur val
   → Revoir stratégie validation (plus de folds, données)

4. Test > Val : CHANCEUX
   → Val set était plus difficile par hasard
   → Ou test set non représentatif

💡 CONCLUSION :
   {"Modèle STABLE et fiable" if abs(test_f1 - best_f1) < 0.03 else "Vérifier overfitting"}
   {"→ Prêt pour production" if abs(test_f1 - best_f1) < 0.03 else "→ Revoir validation"}

🎯 INTERPRÉTATION BUSINESS :
   Recall = {test_rec:.2%} → Détecte {test_rec:.0%} des churners
   Precision = {test_prec:.2%} → {test_prec:.0%} des alertes sont correctes

   Si action de rétention coûte 50€ :
   - {int(cm_test[1, 1])} churners détectés = {int(cm_test[1, 1]) * 50}€ investi
   - Si rétention réussie (50% taux) = {int(cm_test[1, 1] * 0.5)} clients sauvés
   - Si valeur client = 500€ → ROI = {int(cm_test[1, 1] * 0.5 * 500 - cm_test[1, 1] * 50)}€
""")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 7 : SAUVEGARDE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 7 : SAUVEGARDE DU MODÈLE")
print("="*80)

import joblib

joblib.dump(best_model, 'e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/best_ensemble_model.pkl')

print("\n✓ Modèle sauvegardé : best_ensemble_model.pkl")

print("""
📦 UTILISATION EN PRODUCTION
────────────────────────────
```python
import joblib
import numpy as np

# Charger
model = joblib.load('best_ensemble_model.pkl')

# Nouveau client
nouveau_client = np.array([[
    24,      # Tenure (mois)
    89.99,   # MonthlyCharges
    2159.76, # TotalCharges
    # ... autres features
]])

# Prédire
prediction = model.predict(nouveau_client)[0]
proba_churn = model.predict_proba(nouveau_client)[0, 1]

print(f"Risque de churn : {proba_churn:.2%}")

# Décision business
if proba_churn > 0.6:  # Seuil ajustable
    print("🚨 ALERTE : Client à haut risque")
    print("Action : Offre de rétention personnalisée")
elif proba_churn > 0.3:
    print("⚠️  ATTENTION : Client à risque modéré")
    print("Action : Suivi proactif")
else:
    print("✅ OK : Client fidèle")
```

🎯 AJUSTER LE SEUIL selon coûts/bénéfices :
   - Seuil bas (0.3) : Plus d'alertes, détecte plus de churners
   - Seuil haut (0.7) : Moins d'alertes, uniquement très haut risque

   → Optimiser selon coût action vs valeur client
""")

# ═══════════════════════════════════════════════════════════════════════════
# RÉSUMÉ ET CONCLUSIONS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🎉 RÉSUMÉ ET CONCLUSIONS")
print("="*80)

print("""
📚 CE QUE NOUS AVONS APPRIS
───────────────────────────

1️⃣  RANDOM FOREST
   ✅ Excellent baseline (robuste, peu de tuning)
   ✅ Parallélisable (rapide sur multi-core)
   ✅ Rarement overfit grâce au bagging
   ❌ Moins performant que boosting sur données propres

   💼 Usage : Prototype, baseline, peu de temps

2️⃣  XGBOOST
   ✅ Performance MAXIMALE sur tabulaire
   ✅ Régularisation avancée
   ✅ Early stopping essentiel
   ✅ Gère déséquilibre (scale_pos_weight)
   ❌ Tuning complexe (nombreux hyperparamètres)

   💼 Usage : Production, Kaggle, performance critique

3️⃣  LIGHTGBM (si installé)
   ✅ Plus rapide que XGBoost
   ✅ Moins de mémoire
   ✅ Idéal grandes données (>100k)
   ❌ Peut overfitter sur petites données

   💼 Usage : Big Data, vitesse critique

4️⃣  OPTIMISATION
   🔧 Random Forest : Baseline suffit souvent
   🔧 XGBoost : Early stopping >> Grid Search (rapport temps/gain)
   🔧 Grid Search utile si temps disponible (gain 1-3%)

5️⃣  FEATURE IMPORTANCE
   📊 Accords RF ↔ XGBoost → Features fiables
   📊 Divergences → Investiguer (interactions ?)
   ⚠️  Importance ≠ Causalité (attention interprétation)

6️⃣  DIAGNOSTIC
   🔍 Train vs Val : Détecter overfitting
   🔍 Val vs Test : Vérifier généralisation
   🔍 Learning curves : Comprendre convergence

✅ CHECKLIST MODÈLES ENSEMBLE
─────────────────────────────
✓ Random Forest baseline établi
✓ XGBoost avec early stopping testé
✓ Hyperparamètres optimisés (si temps)
✓ Feature importance analysée
✓ Overfitting vérifié (train vs val)
✓ Généralisation validée (val vs test)
✓ Métriques business interprétées
✓ Modèle sauvegardé

🎯 RÈGLE D'OR
─────────────
"Pour données tabulaires :
 1. Baseline : Random Forest
 2. Production : XGBoost avec early stopping
 3. Big Data : LightGBM"

🚀 PROCHAINES ÉTAPES
────────────────────
1. Feature engineering avancé
2. Ensemble de modèles (stacking)
3. Calibration des probabilités
4. Interprétabilité (SHAP values)
5. A/B testing en production
6. Monitoring et retraining automatique

💡 BONUS : QUAND NE PAS UTILISER RF/XGBoost ?
──────────────────────────────────────────────
❌ Images → Utiliser CNN
❌ Texte → Utiliser Transformers (BERT)
❌ Séries temporelles → Utiliser LSTM ou Prophet
❌ Interprétabilité stricte requise → Logistic Reg ou Decision Tree unique
❌ Temps réel critique (<1ms) → Linear models
""")

print("="*80)
print("✨ TUTORIEL TERMINÉ AVEC SUCCÈS ! ✨")
print("="*80)
print("\n🏆 Vous maîtrisez maintenant Random Forest et XGBoost !")
print("📚 Prochain tutoriel : Neural Networks et Deep Learning")
