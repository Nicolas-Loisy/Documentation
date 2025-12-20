"""
═══════════════════════════════════════════════════════════════════════════════
TUTORIEL COMPLET : CLASSIFICATION - LOGISTIC REGRESSION & DECISION TREE
═══════════════════════════════════════════════════════════════════════════════

🎯 CAS D'USAGE RÉEL : Prédiction de Risque de Crédit Bancaire

CONTEXTE :
Une banque veut automatiser l'évaluation du risque de défaut de paiement.
Données : historique de clients avec leurs caractéristiques et statut de remboursement.
Objectif : Prédire si un nouveau client va rembourser (0) ou faire défaut (1).

Ce tutoriel couvre :
1. POURQUOI et QUAND utiliser chaque modèle
2. Préparation des données avec explications des observations
3. Logistic Regression (quand/pourquoi ?)
4. Decision Tree (quand/pourquoi ?)
5. Comparaison et choix du meilleur modèle
6. Diagnostic : que signifient les métriques ?
7. Analyse des erreurs et conclusions

Chaque étape explique CE QU'IL FAUT OBSERVER et LES CONCLUSIONS à en tirer.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("TUTORIEL : CLASSIFICATION - PRÉDICTION DE RISQUE DE CRÉDIT")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 1 : COMPRENDRE LE PROBLÈME ET CHOISIR LE BON MODÈLE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 1 : QUAND UTILISER QUEL MODÈLE DE CLASSIFICATION ?")
print("="*80)

print("""
🎯 CONTEXTE DU PROBLÈME
-----------------------
Type : CLASSIFICATION BINAIRE
- Classe 0 : Client va rembourser (BON client)
- Classe 1 : Client va faire défaut (MAUVAIS client)

⚠️  IMPORTANCE DES ERREURS :
- Faux Positif (prédire défaut alors que bon) → Client refusé à tort → Perte de business
- Faux Négatif (prédire bon alors que défaut) → Prêt non remboursé → Perte financière

💡 Dans ce cas : Faux Négatif plus grave que Faux Positif
   → Nous allons privilégier le RECALL (minimiser faux négatifs)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 TABLEAU DE DÉCISION : QUEL MODÈLE CHOISIR ?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣  LOGISTIC REGRESSION
───────────────────────
✅ UTILISER QUAND :
   - Relation linéaire entre features et log-odds de la classe
   - Besoin d'INTERPRÉTABILITÉ (expliquer pourquoi client refusé)
   - Besoin de PROBABILITÉS calibrées
   - Baseline rapide
   - Réglementation stricte (banque, santé) → besoin d'expliquer décisions

❌ NE PAS UTILISER QUAND :
   - Relations fortement non-linéaires
   - Interactions complexes entre features
   - Features hautement corrélées sans régularisation

📐 FORMULE :
   P(y=1|X) = 1 / (1 + e^-(β₀ + β₁X₁ + ... + βₙXₙ))

💼 CAS D'USAGE TYPIQUES :
   - Scoring de crédit (notre cas !)
   - Diagnostic médical
   - Email spam/non-spam
   - Churn prediction


2️⃣  DECISION TREE
──────────────────
✅ UTILISER QUAND :
   - Relations non-linéaires
   - Interactions complexes entre features
   - Besoin de VISUALISATION des règles de décision
   - Pas besoin de normalisation
   - Features catégorielles et numériques mélangées

❌ NE PAS UTILISER QUAND :
   - Données bruitées (OVERFITTING facile !)
   - Besoin de stabilité (petit changement → arbre différent)
   - Seul (préférer Random Forest pour production)

📐 PRINCIPE :
   Partitionne l'espace des features par seuils successifs
   Ex: "Si revenu > 50k ET âge > 30 ALORS bon client"

💼 CAS D'USAGE TYPIQUES :
   - Systèmes experts médicaux
   - Aide à la décision (règles explicites)
   - Prototypage rapide


3️⃣  COMPARAISON RAPIDE
───────────────────────
┌──────────────────┬─────────────────────┬──────────────────────┐
│  Critère         │  Logistic Reg       │  Decision Tree       │
├──────────────────┼─────────────────────┼──────────────────────┤
│ Interprétabilité │ ★★★★★ (coeff.)      │ ★★★★☆ (règles)      │
│ Performance      │ ★★★☆☆               │ ★★★★☆                │
│ Vitesse          │ ★★★★★               │ ★★★★☆                │
│ Overfitting      │ ★★★★☆ (rare)       │ ★★☆☆☆ (fréquent)    │
│ Non-linéarité    │ ★☆☆☆☆               │ ★★★★★                │
│ Robustesse       │ ★★★★☆               │ ★★☆☆☆                │
└──────────────────┴─────────────────────┴──────────────────────┘

🎯 RECOMMANDATION POUR NOTRE CAS (CRÉDIT BANCAIRE) :
   1. Commencer par Logistic Regression (interprétabilité + réglementation)
   2. Comparer avec Decision Tree
   3. En production : utiliser Random Forest ou XGBoost (meilleure performance)
""")

input("\n▶ Appuyez sur Entrée pour continuer...")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 2 : GÉNÉRATION ET EXPLORATION DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 2 : DONNÉES ET EXPLORATION")
print("="*80)

# 2.1 Générer des données réalistes
print("\n📊 Génération de données synthétiques (simulant clients bancaires)...\n")

# make_classification : Crée un problème de classification
# - n_samples : nombre de clients
# - n_features : nombre total de features
# - n_informative : features réellement prédictives
# - n_redundant : features corrélées aux informatives (bruit réaliste)
# - n_classes : nombre de classes (2 = binaire)
# - weights : proportion des classes [classe 0, classe 1]
#   weights=[0.8, 0.2] → 80% classe 0, 20% classe 1 (déséquilibré, réaliste pour crédit)
# - flip_y : pourcentage de labels aléatoirement inversés (bruit)
# - random_state : reproductibilité
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=7,
    n_redundant=2,
    n_classes=2,
    weights=[0.8, 0.2],  # 80% bons clients, 20% défaut (réaliste)
    flip_y=0.05,  # 5% d'erreur dans les labels (bruit réaliste)
    random_state=42
)

print(f"✓ Données générées : {X.shape[0]} clients × {X.shape[1]} features")
print(f"✓ Classes : {np.bincount(y)}")
print(f"  - Classe 0 (BON client) : {np.bincount(y)[0]} ({np.bincount(y)[0]/len(y)*100:.1f}%)")
print(f"  - Classe 1 (DÉFAUT) : {np.bincount(y)[1]} ({np.bincount(y)[1]/len(y)*100:.1f}%)")

print(f"""
🔍 OBSERVATION #1 : Distribution des classes
────────────────────────────────────────────
Classes DÉSÉQUILIBRÉES : {np.bincount(y)[0]/len(y)*100:.1f}% vs {np.bincount(y)[1]/len(y)*100:.1f}%

💡 CONCLUSION :
   C'est RÉALISTE pour un problème de crédit (majorité de bons clients).

⚠️  CONSÉQUENCE :
   - Un modèle naïf qui prédit toujours "0" aurait {np.bincount(y)[0]/len(y)*100:.1f}% d'accuracy !
   - Accuracy seule est TROMPEUSE sur classes déséquilibrées
   - Privilégier : F1-Score, ROC-AUC, Precision/Recall

✅ ACTION :
   - Utiliser stratify=y dans train_test_split
   - Considérer class_weight='balanced' dans les modèles
   - Évaluer avec plusieurs métriques
""")

# Créer DataFrame avec noms de features réalistes
feature_names = [
    'Revenu', 'Age', 'Anciennete_Emploi', 'Montant_Credit',
    'Taux_Endettement', 'Nb_Credits_Actifs', 'Historique_Paiement',
    'Epargne', 'Valeur_Patrimoine', 'Nb_Dependants'
]
df = pd.DataFrame(X, columns=feature_names)
df['Defaut'] = y

print("\n📈 Aperçu des données :")
print(df.head(10))

print("\n📉 Statistiques descriptives :")
print(df.describe())

# 2.2 Visualisation des distributions
print("\n📊 Analyse des distributions par classe...")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()

for i, col in enumerate(feature_names):
    axes[i].hist(df[df['Defaut'] == 0][col], bins=30, alpha=0.6,
                 label='BON (0)', edgecolor='black', color='green')
    axes[i].hist(df[df['Defaut'] == 1][col], bins=30, alpha=0.6,
                 label='DÉFAUT (1)', edgecolor='black', color='red')
    axes[i].set_title(col)
    axes[i].set_xlabel('Valeur')
    axes[i].set_ylabel('Fréquence')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/classification_distributions.png', dpi=100)
plt.show()

print("\n✓ Graphique sauvegardé : classification_distributions.png")

print(f"""
🔍 OBSERVATION #2 : Séparabilité des classes
────────────────────────────────────────────
Regardez les histogrammes ci-dessus.

CE QU'IL FAUT OBSERVER :
1. Y a-t-il des features où les distributions des 2 classes sont bien séparées ?
   → Ces features sont PRÉDICTIVES

2. Y a-t-il des features où les distributions se chevauchent totalement ?
   → Ces features sont PEU INFORMATIVES

3. Les distributions sont-elles linéairement séparables ou complexes ?
   → Linéaire → Logistic Regression suffira
   → Complexe → Decision Tree ou modèles non-linéaires

💡 CONCLUSION ATTENDUE :
   Si make_classification a fait son travail, certaines features montrent
   une bonne séparation → le problème est RÉSOLUBLE par ML.
""")

# 2.3 Matrice de corrélation
print("\n🔗 Analyse des corrélations...")

correlation_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5, center=0, cbar_kws={'label': 'Corrélation'})
plt.title('Matrice de Corrélation')
plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/classification_correlation.png', dpi=100)
plt.show()

print("\n✓ Graphique sauvegardé : classification_correlation.png")

print("\nCorrélations avec la cible (Defaut) :")
target_corr = correlation_matrix['Defaut'].sort_values(ascending=False)
print(target_corr)

print(f"""
🔍 OBSERVATION #3 : Corrélations
─────────────────────────────────
CE QU'IL FAUT OBSERVER :
1. Corrélation avec la cible (Defaut) :
   - |corr| > 0.3 : Feature FORTEMENT prédictive
   - |corr| 0.1-0.3 : Feature MODÉRÉMENT prédictive
   - |corr| < 0.1 : Feature PEU prédictive

2. Corrélations entre features (multicolinéarité) :
   - |corr| > 0.8 entre 2 features : MULTICOLINÉARITÉ

💡 CONCLUSIONS :
   - Features les plus corrélées avec Defaut = features les plus importantes
   - Multicolinéarité → peut affecter Logistic Regression (coefficients instables)
   - Multicolinéarité → PAS de problème pour Decision Tree

✅ ACTIONS POSSIBLES :
   - Garder seulement les features les plus corrélées (feature selection)
   - Utiliser régularisation L1/L2 pour Logistic Regression
   - Ou utiliser PCA pour réduire multicolinéarité
""")

input("\n▶ Appuyez sur Entrée pour continuer...")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 3 : PRÉPARATION DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 3 : PRÉPARATION DES DONNÉES")
print("="*80)

# 3.1 Division des données
print("\n✂️  Division des données...\n")

# stratify=y : CRUCIAL pour classes déséquilibrées
# Assure que train, val, test ont la même distribution des classes
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"✓ Train set : {len(X_train)} clients")
print(f"  - Classe 0 : {np.bincount(y_train)[0]} ({np.bincount(y_train)[0]/len(y_train)*100:.1f}%)")
print(f"  - Classe 1 : {np.bincount(y_train)[1]} ({np.bincount(y_train)[1]/len(y_train)*100:.1f}%)")

print(f"\n✓ Val set : {len(X_val)} clients")
print(f"  - Classe 0 : {np.bincount(y_val)[0]} ({np.bincount(y_val)[0]/len(y_val)*100:.1f}%)")
print(f"  - Classe 1 : {np.bincount(y_val)[1]} ({np.bincount(y_val)[1]/len(y_val)*100:.1f}%)")

print(f"\n✓ Test set : {len(X_test)} clients")
print(f"  - Classe 0 : {np.bincount(y_test)[0]} ({np.bincount(y_test)[0]/len(y_test)*100:.1f}%)")
print(f"  - Classe 1 : {np.bincount(y_test)[1]} ({np.bincount(y_test)[1]/len(y_test)*100:.1f}%)")

print(f"""
🔍 OBSERVATION #4 : Stratification
───────────────────────────────────
Les proportions sont identiques dans train/val/test grâce à stratify=y.

💡 CONCLUSION :
   Sans stratify, on pourrait avoir par malchance un test set avec
   très peu de classe 1 → évaluation non représentative.
""")

# 3.2 Normalisation
print("\n⚖️  Normalisation (IMPORTANTE pour Logistic Regression)...\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✓ Données normalisées")
print(f"  Moyenne (train) : {X_train_scaled.mean():.4f}")
print(f"  Std (train) : {X_train_scaled.std():.4f}")

print(f"""
❓ POURQUOI NORMALISER ?
────────────────────────
1. Logistic Regression : features sur échelles différentes → coefficients biaisés
   Ex: Revenu (0-100k) vs Age (18-80) → Revenu domine artificiellement

2. Decision Tree : PAS NÉCESSAIRE (splits basés sur seuils, pas magnitudes)
   Mais on normalise quand même pour comparaison équitable.

💡 CONCLUSION :
   Toujours normaliser pour modèles basés sur distances/gradients.
   (Logistic Reg, SVM, Neural Networks, KNN)
""")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 4 : BASELINE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 4 : BASELINE (Modèle Naïf)")
print("="*80)

from sklearn.dummy import DummyClassifier

print("""
❓ POURQUOI UNE BASELINE ?
──────────────────────────
Avant de construire un modèle complexe, établir un point de référence.
Si notre modèle ne bat pas la baseline → PROBLÈME !
""")

# Stratégie 1 : Toujours prédire la classe majoritaire
baseline_majority = DummyClassifier(strategy='most_frequent')
baseline_majority.fit(X_train_scaled, y_train)
y_val_pred_baseline = baseline_majority.predict(X_val_scaled)

acc_baseline = accuracy_score(y_val, y_val_pred_baseline)
f1_baseline = f1_score(y_val, y_val_pred_baseline)

print(f"\n📊 BASELINE (prédire toujours classe {baseline_majority.classes_[np.argmax(baseline_majority.class_prior_)]})")
print(f"  Accuracy : {acc_baseline:.4f}")
print(f"  F1-Score : {f1_baseline:.4f}")

print(f"""
🔍 OBSERVATION #5 : Performance de la baseline
───────────────────────────────────────────────
Accuracy = {acc_baseline:.1%}

CE QUE CELA SIGNIFIE :
Un modèle stupide qui prédit toujours "BON client" a déjà {acc_baseline:.1%} de bonnes réponses !

💡 CONCLUSIONS :
   1. Accuracy seule est TROMPEUSE sur classes déséquilibrées
   2. F1-Score = {f1_baseline:.4f} → très faible car Recall = 0 pour classe minoritaire
   3. Notre modèle doit avoir F1 >> {f1_baseline:.4f} pour être utile
   4. ROC-AUC doit être >> 0.5 (hasard)

🎯 OBJECTIF :
   Battre la baseline sur toutes les métriques, surtout F1 et ROC-AUC.
""")

input("\n▶ Appuyez sur Entrée pour continuer...")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 5 : LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 5 : LOGISTIC REGRESSION")
print("="*80)

print("""
🎯 RAPPEL : POURQUOI LOGISTIC REGRESSION POUR CE CAS ?
───────────────────────────────────────────────────────
✅ Banques doivent JUSTIFIER leurs décisions (réglementation)
✅ Coefficients permettent d'expliquer : "Refusé car revenu trop faible"
✅ Probabilités calibrées utiles pour ajuster seuil de décision
✅ Rapide, stable, bien compris

📐 PRINCIPE :
   P(Défaut=1 | features) = sigmoid(β₀ + β₁·X₁ + ... + βₙ·Xₙ)

   Si P > 0.5 → Prédire Défaut (1)
   Si P < 0.5 → Prédire Bon (0)
""")

# 5.1 Entraînement
print("\n🚀 Entraînement du modèle...\n")

# LogisticRegression : Classification linéaire
# PARAMÈTRES IMPORTANTS :
# - penalty : Type de régularisation ('l1', 'l2', 'elasticnet', None)
#   * 'l2' (Ridge) : réduit coefficients, gère multicolinéarité
#   * 'l1' (Lasso) : sélection de features (coefficients → 0)
# - C : Inverse de la force de régularisation (default=1.0)
#   * C petit → forte régularisation (coefficients petits)
#   * C grand → faible régularisation (risque overfitting)
# - class_weight : Gestion du déséquilibre des classes
#   * 'balanced' : pénalise plus les erreurs sur classe minoritaire
#   * None : traite toutes les erreurs pareil
# - solver : Algorithme d'optimisation
#   * 'lbfgs' : Bon pour la plupart des cas
#   * 'liblinear' : Bon pour petites données
#   * 'saga' : Supporte L1, rapide sur grandes données
# - max_iter : Nombre max d'itérations pour convergence

# Modèle sans class_weight (pour comparaison)
model_lr = LogisticRegression(
    penalty='l2',
    C=1.0,
    max_iter=1000,
    random_state=42
)
model_lr.fit(X_train_scaled, y_train)

# Modèle avec class_weight='balanced' (recommandé pour classes déséquilibrées)
model_lr_balanced = LogisticRegression(
    penalty='l2',
    C=1.0,
    class_weight='balanced',  # IMPORTANT pour déséquilibre
    max_iter=1000,
    random_state=42
)
model_lr_balanced.fit(X_train_scaled, y_train)

print("✓ Modèles entraînés")
print("  - Logistic Regression (standard)")
print("  - Logistic Regression (class_weight='balanced')")

# 5.2 Prédictions
y_val_pred_lr = model_lr.predict(X_val_scaled)
y_val_pred_lr_balanced = model_lr_balanced.predict(X_val_scaled)

# Probabilités (utile pour ROC-AUC et ajuster seuil)
y_val_proba_lr = model_lr.predict_proba(X_val_scaled)[:, 1]
y_val_proba_lr_balanced = model_lr_balanced.predict_proba(X_val_scaled)[:, 1]

# 5.3 Évaluation
print("\n📊 ÉVALUATION\n")

def evaluate_classification(y_true, y_pred, y_proba, model_name):
    """Évaluation complète avec explications"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    print(f"{model_name}")
    print("-" * 60)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print()

    return acc, prec, rec, f1, auc

acc_lr, prec_lr, rec_lr, f1_lr, auc_lr = evaluate_classification(
    y_val, y_val_pred_lr, y_val_proba_lr, "Logistic Regression (standard)"
)

acc_lr_bal, prec_lr_bal, rec_lr_bal, f1_lr_bal, auc_lr_bal = evaluate_classification(
    y_val, y_val_pred_lr_balanced, y_val_proba_lr_balanced,
    "Logistic Regression (class_weight='balanced')"
)

print(f"""
🔍 OBSERVATION #6 : Métriques de Classification
────────────────────────────────────────────────
COMPRENDRE LES MÉTRIQUES :

1. ACCURACY = (TP + TN) / Total
   → % de prédictions correctes
   ⚠️  TROMPEUR sur classes déséquilibrées !

2. PRECISION = TP / (TP + FP)
   → Parmi ceux prédits "Défaut", combien le sont vraiment ?
   💼 Important si coût d'enquête est élevé

3. RECALL (Sensibilité) = TP / (TP + FN)
   → Parmi les vrais "Défaut", combien sont détectés ?
   💼 CRUCIAL pour notre cas (ne pas louper les mauvais clients !)

4. F1-SCORE = 2 * (Precision × Recall) / (Precision + Recall)
   → Moyenne harmonique de Precision et Recall
   💼 Bon compromis pour classes déséquilibrées

5. ROC-AUC = Aire sous courbe ROC
   → Capacité à séparer les classes (0.5=hasard, 1.0=parfait)
   💼 Indépendant du seuil de décision

OBSERVATIONS ATTENDUES :

• Modèle standard :
  Recall = {rec_lr:.4f} → Peut-être FAIBLE (rate des défauts)
  Precision = {prec_lr:.4f} → Peut-être ÉLEVÉE

• Modèle balanced :
  Recall = {rec_lr_bal:.4f} → Devrait être PLUS ÉLEVÉ
  Precision = {prec_lr_bal:.4f} → Peut-être PLUS FAIBLE

💡 CONCLUSION :
   class_weight='balanced' améliore Recall au détriment de Precision.
   Pour risque crédit : Recall > Precision (ne pas louper défauts)
   → Choisir modèle balanced si Recall nettement meilleur.
""")

# 5.4 Matrice de confusion
print("\n📊 Matrices de Confusion...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_lr = confusion_matrix(y_val, y_val_pred_lr)
cm_lr_bal = confusion_matrix(y_val, y_val_pred_lr_balanced)

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Logistic Regression (standard)')
axes[0].set_xlabel('Prédiction')
axes[0].set_ylabel('Réalité')

sns.heatmap(cm_lr_bal, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
axes[1].set_title('Logistic Regression (balanced)')
axes[1].set_xlabel('Prédiction')
axes[1].set_ylabel('Réalité')

plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/classification_confusion_lr.png', dpi=100)
plt.show()

print("\n✓ Graphique sauvegardé : classification_confusion_lr.png")

print(f"""
🔍 OBSERVATION #7 : Matrice de Confusion
─────────────────────────────────────────
LECTURE DE LA MATRICE :

          Prédit 0  Prédit 1
Réel 0  │   TN    │   FP    │  ← Faux Positifs (client bon refusé)
Réel 1  │   FN    │   TP    │  ← Faux Négatifs (défaut non détecté)

CE QU'IL FAUT OBSERVER :

1. Modèle standard :
   FN (bas-gauche) = {cm_lr[1, 0]} → Défauts NON DÉTECTÉS
   FP (haut-droit) = {cm_lr[0, 1]} → Bons clients REFUSÉS

2. Modèle balanced :
   FN (bas-gauche) = {cm_lr_bal[1, 0]} → Devrait être PLUS PETIT
   FP (haut-droit) = {cm_lr_bal[0, 1]} → Peut être PLUS GRAND

💡 CONCLUSION :
   Le modèle balanced détecte plus de défauts (FN ↓) mais refuse plus
   de bons clients (FP ↑). C'est un TRADE-OFF.

🎯 DÉCISION BUSINESS :
   Quel coût est le plus grave ?
   - 1 défaut non détecté = perte de X€ (montant du prêt)
   - 1 bon client refusé = perte de Y€ (intérêts potentiels)

   Si X >> Y → Choisir modèle balanced (minimiser FN)
   Si Y >> X → Choisir modèle standard (minimiser FP)
""")

# 5.5 Courbe ROC
print("\n📊 Courbes ROC...")

fpr_lr, tpr_lr, _ = roc_curve(y_val, y_val_proba_lr)
fpr_lr_bal, tpr_lr_bal, _ = roc_curve(y_val, y_val_proba_lr_balanced)

plt.figure(figsize=(10, 6))
plt.plot(fpr_lr, tpr_lr, label=f'LR standard (AUC={auc_lr:.4f})', linewidth=2)
plt.plot(fpr_lr_bal, tpr_lr_bal, label=f'LR balanced (AUC={auc_lr_bal:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Hasard (AUC=0.5)', linewidth=1)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR = Recall)')
plt.title('Courbes ROC - Logistic Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/classification_roc_lr.png', dpi=100)
plt.show()

print("\n✓ Graphique sauvegardé : classification_roc_lr.png")

print(f"""
🔍 OBSERVATION #8 : Courbe ROC
──────────────────────────────
INTERPRÉTATION :

- Axe X (FPR) : Taux de Faux Positifs (bons clients refusés)
- Axe Y (TPR) : Taux de Vrais Positifs = Recall (défauts détectés)

CE QU'IL FAUT OBSERVER :

1. Courbe proche du coin supérieur gauche = BON modèle
   (TPR élevé avec FPR faible)

2. AUC (Aire sous courbe) :
   - AUC = 0.5 : Modèle au HASARD (ligne diagonale)
   - AUC = 0.7-0.8 : Modèle ACCEPTABLE
   - AUC = 0.8-0.9 : Modèle BON
   - AUC > 0.9 : Modèle EXCELLENT

3. Nos modèles :
   - LR standard : AUC = {auc_lr:.4f}
   - LR balanced : AUC = {auc_lr_bal:.4f}

💡 CONCLUSION :
   Si AUC ≈ même pour les 2 modèles :
   → class_weight n'affecte PAS la capacité de discrimination
   → Il affecte juste le SEUIL de décision par défaut

   On peut donc :
   1. Entraîner avec class_weight='balanced'
   2. Puis ajuster le seuil manuellement selon coûts business
""")

# 5.6 Importance des features (coefficients)
print("\n📊 Importance des Features (Coefficients)...")

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model_lr_balanced.coef_[0],
    'Abs_Coefficient': np.abs(model_lr_balanced.coef_[0])
})
coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

print(coef_df)

plt.figure(figsize=(10, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.xlabel('Coefficient')
plt.title('Importance des Features - Logistic Regression')
plt.axvline(0, color='black', linestyle='-', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/classification_feature_importance_lr.png', dpi=100)
plt.show()

print("\n✓ Graphique sauvegardé : classification_feature_importance_lr.png")

print(f"""
🔍 OBSERVATION #9 : Coefficients (Importance)
──────────────────────────────────────────────
INTERPRÉTATION DES COEFFICIENTS :

1. Coefficient POSITIF → Augmente probabilité de Défaut
   Ex: "Taux_Endettement" positif → Plus d'endettement → Plus de risque

2. Coefficient NÉGATIF → Diminue probabilité de Défaut
   Ex: "Revenu" négatif → Plus de revenu → Moins de risque

3. |Coefficient| grand → Feature IMPORTANTE
   |Coefficient| petit → Feature PEU IMPORTANTE

💼 UTILITÉ BUSINESS :
   "M. Dupont refusé car :
   - Taux endettement élevé (coef = +{coef_df.iloc[0]['Coefficient']:.2f})
   - Revenu faible (coef = {[c for f, c in zip(feature_names, model_lr_balanced.coef_[0]) if 'Revenu' in f][0] if any('Revenu' in f for f in feature_names) else 'N/A'})"

   → EXPLICATION RÉGLEMENTAIRE possible !

⚠️  ATTENTION :
   Les coefficients supposent INDÉPENDANCE des features.
   Si multicolinéarité forte → coefficients instables.
   Solution : Régularisation L1 ou L2.
""")

input("\n▶ Appuyez sur Entrée pour continuer vers Decision Tree...")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 6 : DECISION TREE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 6 : DECISION TREE")
print("="*80)

print("""
🎯 RAPPEL : POURQUOI DECISION TREE ?
────────────────────────────────────
✅ Capture RELATIONS NON-LINÉAIRES
✅ Capture INTERACTIONS automatiquement (ex: "Si revenu < 30k ET âge < 25")
✅ VISUALISATION des règles de décision
✅ Pas besoin de normalisation
✅ Gère features catégorielles nativement

❌ MAIS : Risque élevé d'OVERFITTING

📐 PRINCIPE :
   Partitionne récursivement l'espace des features.
   Ex:
   - Racine : "Revenu < 40k ?"
     - OUI : "Endettement > 50% ?" → Défaut
     - NON : "Age < 25 ?" → ...
""")

# 6.1 Entraînement
print("\n🚀 Entraînement des modèles...\n")

# DecisionTreeClassifier : Arbre de décision
# PARAMÈTRES CRITIQUES POUR ÉVITER OVERFITTING :
# - max_depth : Profondeur maximale de l'arbre
#   * None : Pas de limite (DANGER overfitting !)
#   * 3-5 : Arbre simple (peut underfitter)
#   * 10-20 : Compromis
# - min_samples_split : Nb min d'échantillons pour split
#   * 2 (default) : Split agressif (overfitting)
#   * 20-50 : Plus conservateur
# - min_samples_leaf : Nb min d'échantillons dans feuille
#   * 1 (default) : Feuilles très spécifiques (overfitting)
#   * 10-20 : Feuilles plus générales
# - criterion : Mesure de qualité du split
#   * 'gini' : Impureté de Gini (default, rapide)
#   * 'entropy' : Gain d'information (plus lent)
# - class_weight : Gestion déséquilibre
#   * 'balanced' : Pénalise erreurs classe minoritaire

# Arbre sans contraintes (pour voir overfitting)
model_tree_overfit = DecisionTreeClassifier(
    random_state=42
)
model_tree_overfit.fit(X_train_scaled, y_train)

# Arbre avec contraintes (recommandé)
model_tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)
model_tree.fit(X_train_scaled, y_train)

print("✓ Modèles entraînés")
print("  - Decision Tree (overfitting)")
print("  - Decision Tree (avec contraintes)")

# 6.2 Prédictions
y_val_pred_tree_over = model_tree_overfit.predict(X_val_scaled)
y_val_pred_tree = model_tree.predict(X_val_scaled)

y_val_proba_tree_over = model_tree_overfit.predict_proba(X_val_scaled)[:, 1]
y_val_proba_tree = model_tree.predict_proba(X_val_scaled)[:, 1]

# 6.3 Évaluation
print("\n📊 ÉVALUATION\n")

# Train scores (pour détecter overfitting)
y_train_pred_tree_over = model_tree_overfit.predict(X_train_scaled)
y_train_pred_tree = model_tree.predict(X_train_scaled)

train_acc_over = accuracy_score(y_train, y_train_pred_tree_over)
train_acc = accuracy_score(y_train, y_train_pred_tree)

acc_tree_over, prec_tree_over, rec_tree_over, f1_tree_over, auc_tree_over = evaluate_classification(
    y_val, y_val_pred_tree_over, y_val_proba_tree_over,
    "Decision Tree (overfitting)"
)

acc_tree, prec_tree, rec_tree, f1_tree, auc_tree = evaluate_classification(
    y_val, y_val_pred_tree, y_val_proba_tree,
    "Decision Tree (avec contraintes)"
)

print(f"Train Accuracy (overfitting) : {train_acc_over:.4f}")
print(f"Val Accuracy (overfitting)   : {acc_tree_over:.4f}")
print(f"Écart (overfitting)          : {train_acc_over - acc_tree_over:.4f}\n")

print(f"Train Accuracy (contraintes) : {train_acc:.4f}")
print(f"Val Accuracy (contraintes)   : {acc_tree:.4f}")
print(f"Écart (contraintes)          : {train_acc - acc_tree:.4f}")

print(f"""
🔍 OBSERVATION #10 : Overfitting du Decision Tree
──────────────────────────────────────────────────
CE QU'IL FAUT OBSERVER :

Écart Train - Val :
- Overfitting : {train_acc_over - acc_tree_over:.4f}
- Contraintes : {train_acc - acc_tree:.4f}

💡 RÈGLE :
   Si écart > 0.05 → OVERFITTING probable
   Si écart > 0.10 → OVERFITTING FORT

OBSERVATIONS ATTENDUES :

1. Arbre sans contraintes :
   - Train Acc ≈ 1.00 (apprend par cœur !)
   - Val Acc < Train (généralise mal)
   - Écart IMPORTANT → OVERFITTING

2. Arbre avec contraintes :
   - Train Acc < 1.00 (n'apprend pas par cœur)
   - Val Acc ≈ Train (généralise mieux)
   - Écart FAIBLE → Bon équilibre

💡 CONCLUSION :
   Les contraintes (max_depth, min_samples_split, etc.) sont
   ESSENTIELLES pour éviter overfitting avec Decision Tree.

   En production : Utiliser Random Forest (moyenne de trees)
   → Réduit overfitting naturellement.
""")

# 6.4 Visualisation de l'arbre
print("\n📊 Visualisation de l'arbre (limité à profondeur 3 pour lisibilité)...")

plt.figure(figsize=(20, 10))
plot_tree(
    model_tree,
    max_depth=3,
    feature_names=feature_names,
    class_names=['BON', 'DÉFAUT'],
    filled=True,
    fontsize=10
)
plt.title('Decision Tree (profondeur limitée à 3 pour visualisation)')
plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/classification_tree.png', dpi=100)
plt.show()

print("\n✓ Graphique sauvegardé : classification_tree.png")

print(f"""
🔍 OBSERVATION #11 : Structure de l'Arbre
──────────────────────────────────────────
LECTURE DE L'ARBRE :

1. Nœud racine (en haut) : Premier split le plus important
   → Feature la plus discriminante

2. Chaque nœud montre :
   - Condition de split
   - gini : Impureté (0 = pur, 0.5 = 50/50)
   - samples : Nombre d'échantillons dans ce nœud
   - value : [nb classe 0, nb classe 1]
   - class : Classe majoritaire

3. Couleur :
   - Bleu foncé : Majorité classe 0 (BON)
   - Orange foncé : Majorité classe 1 (DÉFAUT)
   - Couleur claire : Incertain (50/50)

💡 INTERPRÉTATION BUSINESS :
   L'arbre crée des RÈGLES DE DÉCISION explicites :

   "Si Revenu < 40k ET Endettement > 50% → DÉFAUT"

   → Peut être intégré dans système expert
   → Facile à expliquer aux experts métier

⚠️  LIMITES :
   - Arbre instable (petit changement données → arbre différent)
   - Seuils discontinus (ex: 39.9k → BON, 40.1k → DÉFAUT)
   - Peut manquer relations linéaires simples
""")

# 6.5 Feature Importance
print("\n📊 Importance des Features (Decision Tree)...")

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model_tree.feature_importances_
})
importance_df = importance_df.sort_values('Importance', ascending=False)

print(importance_df)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Decision Tree importance
axes[0].barh(importance_df['Feature'], importance_df['Importance'], color='green', alpha=0.7)
axes[0].set_xlabel('Importance')
axes[0].set_title('Feature Importance - Decision Tree')
axes[0].grid(True, alpha=0.3, axis='x')

# Logistic Regression importance (pour comparaison)
axes[1].barh(coef_df['Feature'], coef_df['Abs_Coefficient'], color='blue', alpha=0.7)
axes[1].set_xlabel('|Coefficient|')
axes[1].set_title('Feature Importance - Logistic Regression')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/classification_feature_importance_comparison.png', dpi=100)
plt.show()

print("\n✓ Graphique sauvegardé : classification_feature_importance_comparison.png")

print(f"""
🔍 OBSERVATION #12 : Importance des Features (Tree vs LR)
──────────────────────────────────────────────────────────
DIFFÉRENCES POSSIBLES :

1. Decision Tree : Importance = réduction d'impureté cumulée
   → Features utilisées tôt dans l'arbre = importantes

2. Logistic Regression : |Coefficient| = impact linéaire
   → Features avec corrélation forte = importantes

CE QU'IL FAUT OBSERVER :

• Si rankings similaires → Features vraiment importantes
  Accord entre modèles → Confiance élevée

• Si rankings très différents → Modèles capturent aspects différents
  Ex: Tree trouve interactions, LR trouve effets linéaires

💡 CONCLUSION :
   Features importantes dans LES DEUX modèles :
   → Ce sont les features RÉELLEMENT prédictives
   → Prioriser leur qualité (collecte, nettoyage)

   Features importantes seulement dans Tree :
   → Interactions non-linéaires possibles
   → Peut créer nouvelles features (ex: Revenu × Endettement)
""")

input("\n▶ Appuyez sur Entrée pour voir la comparaison finale...")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 7 : COMPARAISON FINALE ET CHOIX DU MODÈLE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 7 : COMPARAISON FINALE")
print("="*80)

# 7.1 Tableau récapitulatif
print("\n📊 TABLEAU RÉCAPITULATIF DES PERFORMANCES\n")

results = pd.DataFrame({
    'Modèle': [
        'Baseline',
        'LR Standard',
        'LR Balanced',
        'Tree Overfit',
        'Tree Constrained'
    ],
    'Accuracy': [
        acc_baseline, acc_lr, acc_lr_bal, acc_tree_over, acc_tree
    ],
    'Precision': [
        0, prec_lr, prec_lr_bal, prec_tree_over, prec_tree
    ],
    'Recall': [
        0, rec_lr, rec_lr_bal, rec_tree_over, rec_tree
    ],
    'F1-Score': [
        f1_baseline, f1_lr, f1_lr_bal, f1_tree_over, f1_tree
    ],
    'ROC-AUC': [
        0.5, auc_lr, auc_lr_bal, auc_tree_over, auc_tree
    ]
})

print(results.to_string(index=False))

# Visualisation comparative
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    ax.bar(results['Modèle'], results[metric], alpha=0.7, edgecolor='black')
    ax.set_ylabel(metric)
    ax.set_title(f'Comparaison : {metric}')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/classification_comparison.png', dpi=100)
plt.show()

print("\n✓ Graphique sauvegardé : classification_comparison.png")

print(f"""
🔍 OBSERVATION FINALE : Quel modèle choisir ?
──────────────────────────────────────────────
ANALYSE MÉTRIQUE PAR MÉTRIQUE :

1. ACCURACY :
   {results.loc[results['Accuracy'].idxmax(), 'Modèle']} gagne avec {results['Accuracy'].max():.4f}
   ⚠️  Mais rappel : peu fiable sur classes déséquilibrées

2. F1-SCORE (métrique clé) :
   {results.loc[results['F1-Score'].idxmax(), 'Modèle']} gagne avec {results['F1-Score'].max():.4f}

3. RECALL (crucial pour détecter défauts) :
   {results.loc[results['Recall'].idxmax(), 'Modèle']} gagne avec {results['Recall'].max():.4f}

4. ROC-AUC (capacité de discrimination) :
   {results.loc[results['ROC-AUC'].idxmax(), 'Modèle']} gagne avec {results['ROC-AUC'].max():.4f}

""")

# Trouver le meilleur modèle
best_f1_model = results.loc[results['F1-Score'].idxmax(), 'Modèle']
best_auc_model = results.loc[results['ROC-AUC'].idxmax(), 'Modèle']

print(f"""
🎯 RECOMMANDATION FINALE
────────────────────────
Pour CRÉDIT BANCAIRE :

1. PRIORITÉ : Minimiser Faux Négatifs (défauts non détectés)
   → Privilégier RECALL

2. CONTRAINTE : Expliquer les décisions (réglementation)
   → Logistic Regression préférable

3. TRADE-OFF : Recall vs Precision
   → LR Balanced meilleur équilibre

🏆 CHOIX : {best_f1_model}
   F1-Score : {results.loc[results['Modèle'] == best_f1_model, 'F1-Score'].values[0]:.4f}
   Recall : {results.loc[results['Modèle'] == best_f1_model, 'Recall'].values[0]:.4f}
   ROC-AUC : {results.loc[results['Modèle'] == best_f1_model, 'ROC-AUC'].values[0]:.4f}

💼 EN PRODUCTION :
   Option A : Logistic Regression (interprétabilité, réglementation)
   Option B : Random Forest ou XGBoost (performance max)
   → Compromis selon besoins métier

📋 NEXT STEPS :
   1. Optimiser seuil de décision selon coûts métier
   2. Feature engineering (créer interactions)
   3. Tester Random Forest / XGBoost
   4. Cross-validation plus poussée
   5. Tester sur test set final
""")

# 7.2 Évaluation sur test set
print("\n" + "="*80)
print("ÉVALUATION FINALE SUR TEST SET")
print("="*80)

# Choisir le meilleur modèle
if best_f1_model == 'LR Balanced':
    best_model = model_lr_balanced
    model_name = 'Logistic Regression (Balanced)'
elif best_f1_model == 'Tree Constrained':
    best_model = model_tree
    model_name = 'Decision Tree (Constrained)'
else:
    best_model = model_lr_balanced
    model_name = 'Logistic Regression (Balanced)'

y_test_pred = best_model.predict(X_test_scaled)
y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print(f"\n🏆 Modèle sélectionné : {model_name}\n")

test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred)
test_rec = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print("📊 PERFORMANCE SUR TEST SET")
print("-" * 60)
print(f"Accuracy  : {test_acc:.4f}")
print(f"Precision : {test_prec:.4f}")
print(f"Recall    : {test_rec:.4f}")
print(f"F1-Score  : {test_f1:.4f}")
print(f"ROC-AUC   : {test_auc:.4f}")

print("\n📋 Classification Report :")
print(classification_report(y_test, y_test_pred, target_names=['BON', 'DÉFAUT']))

# Matrice de confusion finale
cm_test = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Prédiction')
plt.ylabel('Réalité')
plt.title(f'Matrice de Confusion - Test Set\n{model_name}')
plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/classification_confusion_final.png', dpi=100)
plt.show()

print("\n✓ Graphique sauvegardé : classification_confusion_final.png")

print(f"""
🔍 OBSERVATION FINALE : Performance sur données jamais vues
────────────────────────────────────────────────────────────
Le test set représente la VRAIE performance en production.

CE QU'IL FAUT OBSERVER :

1. Test ≈ Val ? → Modèle GÉNÉRALISE bien
2. Test << Val ? → Peut-être sur-optimisé sur val (overfitting)
3. Test > Val ? → Chanceux ou val set non représentatif

NOTRE CAS :
- F1 Test : {test_f1:.4f}
- F1 Val : {results.loc[results['Modèle'] == best_f1_model, 'F1-Score'].values[0]:.4f}
- Écart : {abs(test_f1 - results.loc[results['Modèle'] == best_f1_model, 'F1-Score'].values[0]):.4f}

💡 CONCLUSION :
   Si écart < 0.05 → Performance STABLE → Prêt pour production
   Si écart > 0.05 → Revoir validation ou plus de données
""")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 8 : SAUVEGARDE ET UTILISATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 8 : SAUVEGARDE DU MODÈLE")
print("="*80)

import joblib

joblib.dump(best_model, 'e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/best_classification_model.pkl')
joblib.dump(scaler, 'e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/classification_scaler.pkl')

print("✓ Modèle sauvegardé : best_classification_model.pkl")
print("✓ Scaler sauvegardé : classification_scaler.pkl")

print(f"""
📦 UTILISATION EN PRODUCTION
────────────────────────────
```python
import joblib
import numpy as np

# Charger
model = joblib.load('best_classification_model.pkl')
scaler = joblib.load('classification_scaler.pkl')

# Nouveau client
nouveau_client = np.array([[
    50000,  # Revenu
    35,     # Age
    5,      # Ancienneté emploi
    # ... autres features
]])

# Normaliser
nouveau_client_scaled = scaler.transform(nouveau_client)

# Prédire
prediction = model.predict(nouveau_client_scaled)[0]
probabilite = model.predict_proba(nouveau_client_scaled)[0, 1]

print(f"Prédiction : {{'BON', 'DÉFAUT'}}[prediction]")
print(f"Probabilité de défaut : {{probabilite:.2%}}")

# Décision avec seuil personnalisé
seuil = 0.3  # Ajuster selon coûts métier
if probabilite > seuil:
    print("REFUSER le crédit")
else:
    print("ACCEPTER le crédit")
```
""")

# ═══════════════════════════════════════════════════════════════════════════
# RÉSUMÉ ET LEÇONS APPRISES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🎉 RÉSUMÉ ET LEÇONS APPRISES")
print("="*80)

print(f"""
📚 CE QUE NOUS AVONS APPRIS
───────────────────────────

1️⃣  CHOIX DU MODÈLE dépend du CONTEXTE :
   ✅ Logistic Regression : Interprétabilité, réglementation, baseline
   ✅ Decision Tree : Relations non-linéaires, règles explicites
   ❌ Attention overfitting avec Decision Tree !

2️⃣  CLASSES DÉSÉQUILIBRÉES :
   ⚠️  Accuracy seule est TROMPEUSE
   ✅ Utiliser : F1-Score, ROC-AUC, Precision/Recall
   ✅ Stratifier train/val/test
   ✅ Utiliser class_weight='balanced'

3️⃣  MÉTRIQUES SELON CONTEXTE :
   💼 Crédit : Privilégier RECALL (détecter défauts)
   📧 Spam : Privilégier PRECISION (pas de faux positifs)
   ⚖️  Équilibre : F1-Score

4️⃣  OBSERVATIONS CLÉS :
   📊 Distributions : Séparabilité des classes
   🔗 Corrélations : Features prédictives
   📉 Matrice confusion : Types d'erreurs
   📈 Courbe ROC : Capacité de discrimination
   🌳 Arbre : Règles de décision
   📊 Feature importance : Features critiques

5️⃣  DIAGNOSTIC OVERFITTING :
   🔍 Train >> Val → Overfitting
   ✅ Contraintes (max_depth, min_samples)
   ✅ Régularisation (L1, L2)
   ✅ Cross-validation

6️⃣  BONNES PRATIQUES :
   ✅ Baseline d'abord
   ✅ Commencer simple (Logistic Reg)
   ✅ Comparer plusieurs modèles
   ✅ Interpréter les résultats (pas juste optimiser)
   ✅ Penser MÉTIER (coûts, contraintes)
   ✅ Test set UNE SEULE FOIS

🎯 PROCHAINES ÉTAPES
────────────────────
1. Tester Random Forest et XGBoost (meilleure performance)
2. Feature engineering avancé (interactions, transformations)
3. Optimiser seuil de décision selon coûts métier
4. Cross-validation plus poussée (Stratified K-Fold)
5. Analyse d'erreurs approfondie (cas difficiles)
6. A/B testing en production

💡 RÈGLE D'OR
─────────────
"Le meilleur modèle n'est PAS celui avec la meilleure métrique,
mais celui qui résout le problème MÉTIER de la façon la plus FIABLE."

✅ CHECKLIST CLASSIFICATION
───────────────────────────
✓ Données explorées (EDA)
✓ Classes équilibrées ou stratifiées
✓ Baseline établie
✓ Plusieurs modèles testés
✓ Métriques appropriées choisies
✓ Overfitting vérifié
✓ Feature importance analysée
✓ Test set évalué UNE FOIS
✓ Modèle sauvegardé
✓ Décision business prise
""")

print("="*80)
print("✨ TUTORIEL TERMINÉ AVEC SUCCÈS ! ✨")
print("="*80)
print("\n🚀 Vous maîtrisez maintenant la classification binaire !")
print("📚 Prochain tutoriel : Random Forest et XGBoost")
