"""
═══════════════════════════════════════════════════════════════════════════════
TUTORIEL COMPLET : NEURAL NETWORKS (RÉSEAUX DE NEURONES)
═══════════════════════════════════════════════════════════════════════════════

🎯 CAS D'USAGE RÉEL : Prédiction de Prix Immobilier (Régression)

CONTEXTE :
Agence immobilière veut prédire prix de vente basé sur caractéristiques.
Données complexes avec interactions non-linéaires.

POURQUOI NEURAL NETWORKS ?
- Capture relations NON-LINÉAIRES complexes
- Interactions automatiques entre features
- Performance excellente si assez de données
- Base du Deep Learning

Ce tutoriel couvre :
1. QUAND utiliser Neural Networks vs modèles classiques
2. Architecture : layers, neurons, activations
3. Forward propagation et Backpropagation (expliqué simplement)
4. Optimiseurs (SGD, Adam, RMSprop) - Quand utiliser quoi ?
5. Learning rate et scheduling
6. Régularisation (Dropout, L1/L2, Batch Normalization)
7. Diagnostic overfitting/underfitting
8. Early stopping et callbacks
9. Comparaison avec modèles classiques

Chaque étape explique CE QU'IL FAUT OBSERVER et LES CONCLUSIONS.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers, optimizers

import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
tf.random.set_seed(42)

print("="*80)
print("TUTORIEL : NEURAL NETWORKS (RÉSEAUX DE NEURONES)")
print("="*80)
print(f"TensorFlow version : {tf.__version__}")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 1 : COMPRENDRE LES NEURAL NETWORKS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 1 : QUAND UTILISER NEURAL NETWORKS ?")
print("="*80)

print("""
🧠 QU'EST-CE QU'UN NEURAL NETWORK ?
────────────────────────────────────
Modèle inspiré du cerveau humain : réseau de neurones artificiels.

STRUCTURE :
   Input Layer → Hidden Layers → Output Layer

   [Feature 1]─┐
   [Feature 2]─┼─→ [Neuron 1]─┐
   [Feature 3]─┼─→ [Neuron 2]─┼─→ [Neuron]─→ [Prédiction]
   [Feature N]─┘  [Neuron M]─┘

Chaque connexion a un POIDS (weight) ajusté pendant entraînement.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 QUAND UTILISER NEURAL NETWORKS vs MODÈLES CLASSIQUES ?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ UTILISER NEURAL NETWORKS QUAND :

1. BEAUCOUP DE DONNÉES (>10k échantillons)
   → NN apprennent mieux avec plus de données
   → Modèles classiques plafonnent plus vite

2. RELATIONS COMPLEXES NON-LINÉAIRES
   → Interactions multiples entre features
   → Patterns difficiles à capturer par arbres

3. DONNÉES HAUTE DIMENSIONNALITÉ
   → Images (milliers de pixels)
   → Texte (milliers de mots)
   → Séquences (séries temporelles)

4. BESOIN DE FLEXIBILITÉ
   → Architecture personnalisable
   → Transfer learning possible

5. PERFORMANCE MAXIMALE REQUISE
   → Prêt à sacrifier interprétabilité
   → Temps/ressources disponibles


❌ NE PAS UTILISER NEURAL NETWORKS QUAND :

1. PEU DE DONNÉES (<1k échantillons)
   → NN overfittent facilement
   → Random Forest / XGBoost meilleurs

2. INTERPRÉTABILITÉ CRITIQUE
   → NN = "boîte noire"
   → Préférer Logistic Reg, Decision Tree

3. DONNÉES TABULAIRES SIMPLES
   → Relations linéaires ou simples
   → XGBoost souvent équivalent et plus rapide

4. RESSOURCES LIMITÉES
   → NN gourmands en calcul/mémoire
   → Modèles classiques plus légers

5. DÉPLOIEMENT TEMPS RÉEL STRICT
   → Latence critique (<1ms)
   → Linear models plus rapides


┌────────────────────┬────────────────────┬────────────────────┐
│  Critère           │  NN                │  XGBoost/RF        │
├────────────────────┼────────────────────┼────────────────────┤
│ Données requises   │ >10k (idéal >100k) │ 1k-10k suffit      │
│ Tabulaires         │ ★★★☆☆              │ ★★★★★              │
│ Images/Texte       │ ★★★★★              │ ★☆☆☆☆              │
│ Interprétabilité   │ ★☆☆☆☆              │ ★★★☆☆              │
│ Vitesse entraîn.   │ ★★☆☆☆              │ ★★★★☆              │
│ Vitesse inférence  │ ★★★☆☆              │ ★★★★☆              │
│ Setup complexité   │ ★★☆☆☆ (difficile)  │ ★★★★☆ (facile)     │
│ Flexibilité        │ ★★★★★              │ ★★★☆☆              │
└────────────────────┴────────────────────┴────────────────────┘

🎯 RECOMMANDATION POUR NOTRE CAS (PRIX IMMOBILIER) :

Données : ~2000 échantillons, 20 features → LIMITE
Objectif : Régression, relations potentiellement complexes

STRATÉGIE :
1. Baseline : Linear Regression
2. Amélioration : XGBoost
3. Exploration : Neural Network (pour comparaison)

💡 Dans la réalité : XGBoost probablement meilleur sur tabulaire.
   Mais NN excellent pour APPRENDRE les concepts !
""")

input("\n▶ Appuyez sur Entrée pour continuer...")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 2 : PRÉPARATION DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 2 : DONNÉES - PRÉDICTION DE PRIX IMMOBILIER")
print("="*80)

# 2.1 Génération
print("\n📊 Génération de données synthétiques...\n")

X, y = make_regression(
    n_samples=2000,
    n_features=20,
    n_informative=15,
    noise=20.0,
    random_state=42
)

# Ajouter interactions non-linéaires
X[:, 0] = X[:, 0] ** 2  # Feature 0 au carré
X[:, 1] = np.log1p(np.abs(X[:, 1]))  # Log de feature 1
X[:, 2] = X[:, 2] * X[:, 3]  # Interaction entre 2 et 3

# Normaliser target (prix) pour faciliter entraînement
y = (y - y.mean()) / y.std()

print(f"✓ Données générées : {X.shape[0]} propriétés × {X.shape[1]} features")

# Features réalistes
feature_names = [
    'Surface', 'Chambres', 'Surface_x_Chambres', 'Annee_Construction',
    'Distance_Centre', 'Etage', 'Balcon', 'Parking',
    'Etat', 'Chauffage', 'Isolation', 'Quartier_Score',
    'Commerces_Proximite', 'Transports', 'Ecoles', 'Verdure',
    'Bruit', 'Securite', 'Taxe_Fonciere', 'Charges'
]

df = pd.DataFrame(X, columns=feature_names)
df['Prix'] = y

print("\n📈 Aperçu :")
print(df.head())

# 2.2 Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"\n✓ Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

# 2.3 Normalisation (CRUCIALE pour NN)
print("\n⚖️  Normalisation...\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✓ Données normalisées")

print("""
💡 POURQUOI NORMALISER EST CRUCIAL POUR NEURAL NETWORKS ?
──────────────────────────────────────────────────────────

1. GRADIENT DESCENT CONVERGE MIEUX
   Si features sur échelles différentes (ex: Surface=100 vs Etage=3)
   → Gradients de magnitudes différentes
   → Oscillations, convergence lente ou échec

2. POIDS INITIALISÉS DE FAÇON UNIFORME
   Initialisation suppose inputs normalisés
   → Sinon certains neurons dominés dès le départ

3. ACTIVATIONS STABLES
   Inputs normalisés → Outputs de layers intermédiaires stables
   → Évite "gradient vanishing" ou "exploding"

4. LEARNING RATE UNIQUE FONCTIONNE
   Même learning rate pour toutes les features
   → Sinon faut ajuster par feature (complexe)

⚠️  SANS NORMALISATION :
   - Convergence 10-100× plus lente
   - Souvent ne converge jamais
   - Performance dégradée

✅ AVEC NORMALISATION :
   - Convergence rapide et stable
   - Performance optimale
   - Hyperparamètres plus faciles à tuner

🎯 RÈGLE : TOUJOURS normaliser pour Neural Networks !
""")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 3 : ARCHITECTURE DES NEURAL NETWORKS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 3 : ARCHITECTURE - CONSTRUIRE UN RÉSEAU")
print("="*80)

print("""
🏗️  COMPOSANTS D'UN NEURAL NETWORK
───────────────────────────────────

1️⃣  LAYERS (COUCHES)
────────────────────
Types principaux :

• Dense (Fully Connected) : Tous neurons connectés
  → Usage : Tabulaire, features générales

• Convolutional : Détecte patterns locaux
  → Usage : Images, signaux

• Recurrent (LSTM, GRU) : Mémoire séquentielle
  → Usage : Séries temporelles, texte

• Dropout : Désactive neurons aléatoirement
  → Usage : Régularisation


2️⃣  ACTIVATION FUNCTIONS
─────────────────────────
Introduisent NON-LINÉARITÉ (sinon réseau = régression linéaire !)

┌────────────┬─────────────────┬─────────────────────┐
│ Activation │ Usage           │ Caractéristiques    │
├────────────┼─────────────────┼─────────────────────┤
│ ReLU       │ Hidden layers   │ Rapide, efficace    │
│            │ (par défaut)    │ Peut "mourir"       │
├────────────┼─────────────────┼─────────────────────┤
│ LeakyReLU  │ Hidden layers   │ Évite "dying ReLU"  │
│            │ (alternative)   │                     │
├────────────┼─────────────────┼─────────────────────┤
│ Sigmoid    │ Output binaire  │ Entre 0 et 1        │
│            │                 │ Vanishing gradient  │
├────────────┼─────────────────┼─────────────────────┤
│ Tanh       │ Hidden layers   │ Entre -1 et 1       │
│            │                 │ Meilleur que sigmoid│
├────────────┼─────────────────┼─────────────────────┤
│ Linear     │ Output régression│ Pas d'activation   │
│ (None)     │                 │                     │
└────────────┴─────────────────┴─────────────────────┘

🎯 RECOMMANDATIONS :
   Hidden layers : ReLU (ou LeakyReLU si problème)
   Output régression : Linear (None)
   Output classification binaire : Sigmoid
   Output multi-classe : Softmax


3️⃣  NOMBRE DE LAYERS ET NEURONS
─────────────────────────────────

RÈGLES EMPIRIQUES :

• Nombre de layers (profondeur) :
  - 1-2 hidden layers : Problèmes simples
  - 3-5 layers : Problèmes modérés (notre cas)
  - >5 layers : Deep Learning (images, texte)

• Nombre de neurons par layer :
  - Layer 1 : Plus grand (ex: 128, 256)
  - Layer 2 : Moyen (ex: 64, 128)
  - Layer 3+ : Plus petit (ex: 32, 64)

  Structure "entonnoir" : Réduction progressive

• Taille totale du réseau :
  - Petit : <10k paramètres → Petites données
  - Moyen : 10k-100k → Données moyennes
  - Grand : >100k → Grandes données

⚠️  TROP DE NEURONS : Overfitting
⚠️  TROP PEU : Underfitting

💡 START SIMPLE, SCALE UP :
   Commencer petit, augmenter si underfitting.
""")

# 3.1 Modèle simple (baseline)
print("\n🏗️  Construction du modèle SIMPLE (baseline)...\n")

def create_simple_model(input_dim):
    """
    Modèle simple : 1 hidden layer
    """
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim, name='hidden'),
        layers.Dense(1, activation='linear', name='output')
    ])

    return model

model_simple = create_simple_model(X_train_scaled.shape[1])

# Compiler : spécifie loss, optimizer, metrics
# - loss='mse' : Mean Squared Error (régression)
# - optimizer='adam' : Algorithme d'optimisation
# - metrics=['mae'] : Métriques à suivre (pas optimisées, juste monitorées)
model_simple.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae']
)

# Summary : affiche architecture
print(model_simple.summary())

print(f"""
🔍 OBSERVATION #1 : Architecture du Modèle
───────────────────────────────────────────
CE QU'IL FAUT OBSERVER DANS LE SUMMARY :

1. NOMBRE DE PARAMÈTRES (Total params) :
   Formula : (input_neurons + 1) × output_neurons par layer

   Notre modèle : {model_simple.count_params()} paramètres

   Hidden : ({X_train_scaled.shape[1]} inputs + 1 bias) × 64 neurons = {(X_train_scaled.shape[1] + 1) * 64}
   Output : (64 + 1) × 1 = {65}

2. INTERPRÉTATION :
   - <10k params : Modèle SIMPLE → Bon pour petites données
   - 10k-100k : Modèle MOYEN → Données moyennes
   - >100k : Modèle COMPLEXE → Beaucoup de données requises

   Notre cas : {model_simple.count_params()} params → APPROPRIÉ pour {len(X_train)} échantillons

3. RÈGLE EMPIRIQUE :
   Nombre params ≈ Nombre échantillons / 10

   Notre ratio : {model_simple.count_params()} / {len(X_train)} = {model_simple.count_params() / len(X_train):.2f}
   → {"Bon" if model_simple.count_params() / len(X_train) < 0.2 else "Risque overfitting !"}

💡 CONCLUSION :
   Modèle simple avec 1 layer suffit souvent.
   Augmenter complexité seulement si underfitting.
""")

# 3.2 Modèle profond
print("\n🏗️  Construction du modèle PROFOND...\n")

def create_deep_model(input_dim):
    """
    Modèle profond : 3 hidden layers + Dropout
    """
    model = models.Sequential([
        # Layer 1 : Large
        layers.Dense(128, activation='relu', input_dim=input_dim, name='hidden1'),
        layers.Dropout(0.3, name='dropout1'),

        # Layer 2 : Moyen
        layers.Dense(64, activation='relu', name='hidden2'),
        layers.Dropout(0.2, name='dropout2'),

        # Layer 3 : Small
        layers.Dense(32, activation='relu', name='hidden3'),

        # Output
        layers.Dense(1, activation='linear', name='output')
    ])

    return model

model_deep = create_deep_model(X_train_scaled.shape[1])

model_deep.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae']
)

print(model_deep.summary())

print(f"""
🔍 OBSERVATION #2 : Modèle Simple vs Profond
─────────────────────────────────────────────
Simple : {model_simple.count_params()} paramètres
Profond : {model_deep.count_params()} paramètres

CE QU'IL FAUT OBSERVER :

1. CAPACITÉ DU MODÈLE :
   Plus de paramètres = Plus de capacité d'apprentissage
   → Peut capturer patterns plus complexes
   → Mais risque overfitting ++

2. DROPOUT LAYERS :
   Désactive aléatoirement neurons pendant entraînement
   → Force réseau à apprendre features redondantes
   → RÉGULARISATION puissante contre overfitting

   Dropout rate :
   - 0.2-0.3 : Léger (nos hidden layers)
   - 0.5 : Fort (moins utilisé maintenant)

3. STRUCTURE "ENTONNOIR" :
   128 → 64 → 32 → 1

   POURQUOI ?
   - Début : Features brutes, haute dimensionnalité
   - Milieu : Représentations abstraites
   - Fin : Décision compressée

   Comme cerveau : compression progressive d'information

💡 CONCLUSION :
   Modèle profond plus puissant MAIS :
   - Risque overfitting si pas assez de données
   - Plus lent à entraîner
   - Plus difficile à optimiser

   → Start simple, go deep si underfitting !
""")

input("\n▶ Appuyez sur Entrée pour continuer...")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 4 : ENTRAÎNEMENT ET OPTIMISEURS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 4 : ENTRAÎNEMENT - OPTIMISEURS ET LEARNING RATE")
print("="*80)

print("""
⚙️  OPTIMISEURS : COMMENT LE RÉSEAU APPREND
────────────────────────────────────────────

PRINCIPE : Ajuster poids pour minimiser loss via GRADIENT DESCENT

1️⃣  SGD (Stochastic Gradient Descent)
──────────────────────────────────────
PRINCIPE : θ_{t+1} = θ_t - η × ∇Loss

✅ AVANTAGES :
   - Simple, bien compris
   - Avec momentum : évite minima locaux

❌ INCONVÉNIENTS :
   - Learning rate fixe → Problématique
   - Convergence lente
   - Sensible au scaling des features

💼 USAGE : Rarement utilisé seul maintenant


2️⃣  ADAM (Adaptive Moment Estimation)
───────────────────────────────────────
PRINCIPE : Learning rate ADAPTATIF par paramètre

✅ AVANTAGES :
   - Learning rate s'ajuste automatiquement
   - Convergence RAPIDE
   - Robuste, fonctionne out-of-the-box
   - PEU SENSIBLE au tuning

❌ INCONVÉNIENTS :
   - Parfois converge vers solution sous-optimale
   - Mémoire supplémentaire

💼 USAGE : DÉFAUT RECOMMANDÉ (90% des cas)


3️⃣  RMSprop
─────────────
PRINCIPE : Adapte learning rate basé sur moyenne des gradients récents

✅ AVANTAGES :
   - Bon pour RNN
   - Gère bien gradients bruyants

❌ INCONVÉNIENTS :
   - Moins performant qu'Adam généralement

💼 USAGE : RNN, séries temporelles


4️⃣  ADAGRAD, ADADELTA, etc.
────────────────────────────
Variantes avec adaptations spécifiques.
Rarement utilisées maintenant (Adam les surpasse).


┌────────────┬────────────┬────────────┬────────────────────┐
│ Optimizer  │ Vitesse    │ Robustesse │ Usage              │
├────────────┼────────────┼────────────┼────────────────────┤
│ SGD        │ ★★☆☆☆      │ ★★☆☆☆      │ Rare               │
│ SGD+moment │ ★★★☆☆      │ ★★★☆☆      │ Fine-tuning        │
│ Adam       │ ★★★★★      │ ★★★★★      │ DÉFAUT (90%)       │
│ RMSprop    │ ★★★★☆      │ ★★★☆☆      │ RNN                │
└────────────┴────────────┴────────────┴────────────────────┘

🎯 RECOMMANDATION :
   TOUJOURS commencer avec ADAM.
   Changer seulement si problème spécifique.


📊 LEARNING RATE (η)
────────────────────
Paramètre le PLUS IMPORTANT !

η trop petit → Convergence TRÈS lente
η trop grand → Divergence ou oscillations
η optimal → Convergence rapide et stable

VALEURS TYPIQUES :
- SGD : 0.01 - 0.1
- Adam : 0.0001 - 0.01 (défaut : 0.001)

💡 STRATÉGIE :
   1. Adam avec défaut (0.001) → Marche 80% du temps
   2. Si convergence lente : Augmenter (0.003, 0.01)
   3. Si divergence : Diminuer (0.0003, 0.0001)
""")

# 4.1 Entraîner modèle simple
print("\n🚀 Entraînement du modèle SIMPLE...\n")

# fit() : Entraîne le modèle
# - X_train, y_train : Données d'entraînement
# - validation_data : Données de validation (évalué mais pas entraîné dessus)
# - epochs : Nombre de passages complets sur les données
# - batch_size : Nombre d'échantillons par mise à jour des poids
# - verbose : Affichage (0=silent, 1=progress bar, 2=one line per epoch)

history_simple = model_simple.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    verbose=0  # Silent pour ne pas polluer output
)

print("✓ Entraînement terminé")

# 4.2 Visualiser courbes d'apprentissage
print("\n📊 Courbes d'apprentissage...\n")

def plot_training_history(history, title):
    """
    Visualise loss et métriques pendant entraînement
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title(f'{title} - MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

fig = plot_training_history(history_simple, "Modèle Simple")
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/nn_training_simple.png', dpi=100)
plt.show()

print("✓ Graphique sauvegardé : nn_training_simple.png")

print(f"""
🔍 OBSERVATION #3 : Courbes d'Apprentissage
────────────────────────────────────────────
CE QU'IL FAUT OBSERVER :

1. CONVERGENCE :
   Loss diminue-t-elle régulièrement ?
   ✅ Oui : Apprentissage normal
   ❌ Non (plateau précoce) : Underfitting ou learning rate trop petit
   ❌ Non (oscillations) : Learning rate trop grand ou batch size trop petit

2. OVERFITTING :
   Train Loss << Val Loss ?

   Écart actuel : {history_simple.history['loss'][-1]:.4f} vs {history_simple.history['val_loss'][-1]:.4f}

   ✅ Écart < 10% : Pas d'overfitting
   ⚠️  Écart 10-30% : Léger overfitting
   ❌ Écart > 30% : Overfitting FORT

3. UNDERFITTING :
   Train Loss ET Val Loss élevées ?
   → Modèle pas assez puissant

   Actions :
   - Augmenter nombre de neurons
   - Ajouter layers
   - Entraîner plus longtemps

4. MOMENT D'ARRÊT :
   Val Loss ne diminue plus depuis plusieurs epochs ?
   → EARLY STOPPING recommandé

   Epoch optimal ≈ {np.argmin(history_simple.history['val_loss']) + 1} / 100

💡 INTERPRÉTATION DE NOS COURBES :
   - Convergence : {"✅ Stable" if history_simple.history['loss'][-1] < history_simple.history['loss'][10] else "❌ Problème"}
   - Overfitting : {"✅ Pas d'overfitting" if (history_simple.history['val_loss'][-1] / history_simple.history['loss'][-1] - 1) < 0.3 else "⚠️ Overfitting détecté"}
   - Early stop : Aurait pu arrêter epoch {np.argmin(history_simple.history['val_loss']) + 1}

🎯 CONCLUSION :
   Courbes d'apprentissage = DIAGNOSTIC le plus important !
   À vérifier SYSTÉMATIQUEMENT.
""")

# 4.3 Modèle avec Early Stopping
print("\n🚀 Entraînement avec EARLY STOPPING...\n")

# Recréer modèle (réinitialiser poids)
model_early_stop = create_simple_model(X_train_scaled.shape[1])
model_early_stop.compile(loss='mse', optimizer='adam', metrics=['mae'])

# EarlyStopping : Arrête si val_loss ne s'améliore plus
early_stop_callback = callbacks.EarlyStopping(
    monitor='val_loss',  # Métrique à surveiller
    patience=15,  # Attendre N epochs sans amélioration
    restore_best_weights=True,  # Restaurer meilleurs poids
    verbose=1
)

# ReduceLROnPlateau : Réduit learning rate si plateau
reduce_lr_callback = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Diviser LR par 2
    patience=10,
    min_lr=1e-7,
    verbose=1
)

history_early_stop = model_early_stop.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,  # Beaucoup car early stopping arrêtera
    batch_size=32,
    callbacks=[early_stop_callback, reduce_lr_callback],
    verbose=0
)

print(f"\n✓ Entraînement arrêté epoch {len(history_early_stop.history['loss'])}")

fig = plot_training_history(history_early_stop, "Avec Early Stopping")
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/nn_training_early_stop.png', dpi=100)
plt.show()

print("✓ Graphique sauvegardé : nn_training_early_stop.png")

print(f"""
🔍 OBSERVATION #4 : Early Stopping
───────────────────────────────────
Arrêté epoch {len(history_early_stop.history['loss'])} / 200

CE QU'IL FAUT OBSERVER :

1. EPOCH D'ARRÊT :
   - Arrêt précoce (<50) : Convergence rapide OU underfitting
   - Arrêt moyen (50-150) : OPTIMAL
   - Arrêt tardif (>150) : Modèle complexe ou learning rate trop petit

2. VAL LOSS FINALE :
   Sans early stop : {history_simple.history['val_loss'][-1]:.4f} (epoch 100)
   Avec early stop : {min(history_early_stop.history['val_loss']):.4f} (epoch {np.argmin(history_early_stop.history['val_loss']) + 1})

   Amélioration : {((history_simple.history['val_loss'][-1] - min(history_early_stop.history['val_loss'])) / history_simple.history['val_loss'][-1] * 100):.1f}%

3. REDUCE LR ON PLATEAU :
   Learning rate réduit si val_loss stagne
   → Permet convergence fine
   → Visible sur courbes (accélération après réduction)

💡 CONCLUSION :
   Early Stopping est ESSENTIEL :
   ✅ Évite overfitting automatiquement
   ✅ Économise temps (arrête quand inutile continuer)
   ✅ Trouve epoch optimal automatiquement

   TOUJOURS utiliser en production !

⚙️  TUNING :
   - patience=15 : Standard pour petites données
   - patience=3-5 : Grandes données, entraînement long
   - patience=20-50 : Très petites données, convergence lente
""")

input("\n▶ Appuyez sur Entrée pour continuer...")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 5 : RÉGULARISATION AVANCÉE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 5 : RÉGULARISATION - COMBATTRE L'OVERFITTING")
print("="*80)

print("""
🛡️  TECHNIQUES DE RÉGULARISATION
─────────────────────────────────

1️⃣  DROPOUT
────────────
Désactive aléatoirement neurons pendant entraînement.

PRINCIPE :
   Layer avec 100 neurons, Dropout(0.3)
   → 30 neurons désactivés aléatoirement à chaque batch
   → Force réseau à ne pas dépendre d'un neuron spécifique

✅ AVANTAGES :
   - Très efficace contre overfitting
   - Agit comme ensemble de réseaux

TAUX RECOMMANDÉS :
   - 0.2-0.3 : Léger (hidden layers)
   - 0.5 : Fort (rarement utilisé maintenant)
   - 0.1 : Très léger

💼 USAGE : Entre chaque dense layer (sauf output)


2️⃣  L1 / L2 REGULARIZATION
────────────────────────────
Pénalise poids trop grands.

L1 (Lasso) : loss + λ·Σ|w|
   → Met certains poids à 0
   → Sélection de features

L2 (Ridge) : loss + λ·Σw²
   → Réduit tous les poids
   → Évite poids extrêmes

λ (paramètre) :
   - 0.0001 : Léger
   - 0.001 : Moyen
   - 0.01 : Fort

💼 USAGE : Dans les Dense layers


3️⃣  BATCH NORMALIZATION
────────────────────────
Normalise inputs de chaque layer.

PRINCIPE :
   Output du layer normalisé (mean=0, std=1)
   → Accélère entraînement
   → Régularisation (effet similaire Dropout léger)

✅ AVANTAGES :
   - Entraînement 2-3× plus rapide
   - Permet learning rate plus élevés
   - Réduit sensibilité à l'initialisation

⚠️  ATTENTION :
   Complexifie le modèle
   Pas toujours nécessaire sur petites données

💼 USAGE : Après Dense layer, avant activation


4️⃣  EARLY STOPPING
────────────────────
Arrête entraînement quand val_loss ne s'améliore plus.

✅ AVANTAGES :
   - Simple et très efficace
   - Pas de calcul supplémentaire

💼 USAGE : TOUJOURS !


5️⃣  DATA AUGMENTATION
───────────────────────
Crée variations artificielles des données.

Pour images : rotation, flip, zoom, etc.
Pour tabulaire : bruit, perturbations légères

💼 USAGE : Images surtout


┌─────────────────────┬────────────┬────────────┬──────────────┐
│ Technique           │ Efficacité │ Coût       │ Usage        │
├─────────────────────┼────────────┼────────────┼──────────────┤
│ Dropout             │ ★★★★★      │ Faible     │ TOUJOURS     │
│ Early Stopping      │ ★★★★★      │ Nul        │ TOUJOURS     │
│ L2                  │ ★★★☆☆      │ Nul        │ Si overfitting│
│ Batch Norm          │ ★★★★☆      │ Moyen      │ Grands réseaux│
│ Data Augmentation   │ ★★★★★      │ Moyen      │ Images       │
└─────────────────────┴────────────┴────────────┴──────────────┘

🎯 STRATÉGIE RECOMMANDÉE :
   1. TOUJOURS : Early Stopping
   2. Si overfitting : Ajouter Dropout (0.2-0.3)
   3. Si encore overfit : Ajouter L2 (0.001)
   4. Si grands réseaux : Batch Normalization
""")

# 5.1 Modèle avec régularisation complète
print("\n🏗️  Construction modèle avec RÉGULARISATION COMPLÈTE...\n")

def create_regularized_model(input_dim):
    """
    Modèle avec toutes les régularisations
    """
    model = models.Sequential([
        # Layer 1
        layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),  # L2
            input_dim=input_dim,
            name='hidden1'
        ),
        layers.BatchNormalization(name='bn1'),
        layers.Dropout(0.3, name='dropout1'),

        # Layer 2
        layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            name='hidden2'
        ),
        layers.BatchNormalization(name='bn2'),
        layers.Dropout(0.2, name='dropout2'),

        # Layer 3
        layers.Dense(
            32,
            activation='relu',
            name='hidden3'
        ),

        # Output
        layers.Dense(1, activation='linear', name='output')
    ])

    return model

model_regularized = create_regularized_model(X_train_scaled.shape[1])

model_regularized.compile(
    loss='mse',
    optimizer=optimizers.Adam(learning_rate=0.001),
    metrics=['mae']
)

print(model_regularized.summary())

# Entraîner
history_regularized = model_regularized.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop_callback, reduce_lr_callback],
    verbose=0
)

print(f"✓ Entraînement terminé (epoch {len(history_regularized.history['loss'])})")

# Comparer toutes les versions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history_simple.history['val_loss'], label='Simple', linewidth=2, alpha=0.7)
axes[0].plot(history_early_stop.history['val_loss'], label='Early Stop', linewidth=2, alpha=0.7)
axes[0].plot(history_regularized.history['val_loss'], label='Régularisé', linewidth=2, alpha=0.7)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Loss')
axes[0].set_title('Comparaison : Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE
axes[1].plot(history_simple.history['val_mae'], label='Simple', linewidth=2, alpha=0.7)
axes[1].plot(history_early_stop.history['val_mae'], label='Early Stop', linewidth=2, alpha=0.7)
axes[1].plot(history_regularized.history['val_mae'], label='Régularisé', linewidth=2, alpha=0.7)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation MAE')
axes[1].set_title('Comparaison : Validation MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/nn_comparison.png', dpi=100)
plt.show()

print("\n✓ Graphique sauvegardé : nn_comparison.png")

# Métriques finales
models_comparison = {
    'Simple': (model_simple, history_simple),
    'Early Stop': (model_early_stop, history_early_stop),
    'Régularisé': (model_regularized, history_regularized)
}

print(f"""
🔍 OBSERVATION #5 : Impact de la Régularisation
────────────────────────────────────────────────
CE QU'IL FAUT OBSERVER :

1. VAL LOSS FINALE :
   Simple : {history_simple.history['val_loss'][-1]:.4f}
   Early Stop : {min(history_early_stop.history['val_loss']):.4f}
   Régularisé : {min(history_regularized.history['val_loss']):.4f}

2. STABILITÉ :
   Courbes régularisées généralement plus LISSES
   → Moins d'oscillations
   → Convergence plus stable

3. OVERFITTING :
   Écart Train vs Val :
   - Simple peut avoir grand écart
   - Régularisé devrait avoir écart RÉDUIT

💡 INTERPRÉTATION :
   Si Régularisé >> Simple :
   → Régularisation EFFICACE, nécessaire

   Si Régularisé ≈ Simple :
   → Régularisation inutile, modèle simple suffisait
   → Ou données suffisantes (pas besoin régularisation)

   Si Régularisé < Simple :
   → Sur-régularisation ! Modèle bloqué (underfitting)
   → Réduire dropout ou L2

🎯 CONCLUSION :
   Régularisation nécessaire si et seulement si overfitting.
   Sur petites données : Crucial
   Sur grandes données : Souvent moins nécessaire

⚙️  TUNING RÉGULARISATION :
   Trop d'overfitting :
   1. Augmenter Dropout (0.2 → 0.4)
   2. Augmenter L2 (0.001 → 0.01)
   3. Réduire taille du modèle

   Underfitting :
   1. Réduire Dropout (0.4 → 0.2)
   2. Réduire L2 (0.01 → 0.001)
   3. Augmenter taille du modèle
""")

input("\n▶ Appuyez sur Entrée pour l'évaluation finale...")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 6 : ÉVALUATION FINALE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 6 : ÉVALUATION FINALE ET COMPARAISON")
print("="*80)

# Évaluer tous les modèles sur test set
results = []

for name, (model, history) in models_comparison.items():
    y_pred = model.predict(X_test_scaled, verbose=0).flatten()

    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    results.append({
        'Modèle': name,
        'MSE': test_mse,
        'RMSE': test_rmse,
        'MAE': test_mae,
        'R²': test_r2,
        'Params': model.count_params()
    })

results_df = pd.DataFrame(results)

print("\n📊 RÉSULTATS SUR TEST SET\n")
print(results_df.to_string(index=False))

# Meilleur modèle
best_model_name = results_df.loc[results_df['R²'].idxmax(), 'Modèle']
best_model = models_comparison[best_model_name][0]

print(f"\n🏆 Meilleur modèle : {best_model_name}")

# Visualisation prédictions
y_test_pred = best_model.predict(X_test_scaled, verbose=0).flatten()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter pred vs real
axes[0].scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k')
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2, label='Prédiction parfaite')
axes[0].set_xlabel('Prix Réel (normalisé)')
axes[0].set_ylabel('Prix Prédit')
axes[0].set_title(f'Prédictions vs Réalité\n{best_model_name} (R² = {results_df.loc[results_df["Modèle"] == best_model_name, "R²"].values[0]:.4f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Résidus
residuals = y_test - y_test_pred
axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Résidus')
axes[1].set_ylabel('Fréquence')
axes[1].set_title(f'Distribution des Résidus\n(MAE = {results_df.loc[results_df["Modèle"] == best_model_name, "MAE"].values[0]:.4f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/nn_final_results.png', dpi=100)
plt.show()

print("\n✓ Graphique sauvegardé : nn_final_results.png")

print(f"""
🔍 OBSERVATION #6 : Performance Finale
───────────────────────────────────────
CE QU'IL FAUT OBSERVER :

1. R² SCORE :
   R² = {results_df.loc[results_df['Modèle'] == best_model_name, 'R²'].values[0]:.4f}

   Interprétation :
   - R² > 0.9 : EXCELLENT
   - R² 0.7-0.9 : BON
   - R² 0.5-0.7 : ACCEPTABLE
   - R² < 0.5 : FAIBLE

2. SCATTER PLOT :
   Points proches de la ligne rouge = Bonnes prédictions
   Dispersion = Erreur de prédiction

   Biais visible (systématiquement au-dessus/en-dessous) ?
   → Non : Modèle non biaisé ✅
   → Oui : Problème dans les données ou modèle

3. RÉSIDUS :
   Distribution centrée sur 0 ? ✅ Pas de biais
   Forme gaussienne ? ✅ Erreurs aléatoires
   Asymétrique ou multi-modale ? ❌ Problème

4. COMPARAISON DES MODÈLES :
   Amélioration Simple → Régularisé :
   R² : {results_df.loc[results_df['Modèle'] == 'Simple', 'R²'].values[0]:.4f} → {results_df.loc[results_df['Modèle'] == best_model_name, 'R²'].values[0]:.4f}

   {"✅ Régularisation utile" if results_df.loc[results_df['Modèle'] == best_model_name, 'R²'].values[0] > results_df.loc[results_df['Modèle'] == 'Simple', 'R²'].values[0] + 0.02 else "⚠️ Régularisation peu d'impact"}

💡 CONCLUSION :
   Neural Network performant sur ce problème de régression.

   Comparaison avec modèles classiques recommandée :
   - XGBoost probablement similaire ou meilleur sur tabulaire
   - Linear Regression si relations linéaires

   NN justifié si :
   ✅ Relations très non-linéaires
   ✅ Beaucoup de données (>10k)
   ✅ Besoin de flexibilité architecturale

🎯 EN PRODUCTION :
   Avantages NN :
   ✅ Flexible, personnalisable
   ✅ Transfer learning possible
   ✅ Intégration facile (TensorFlow Serving)

   Inconvénients NN :
   ❌ "Boîte noire"
   ❌ Plus lent que modèles classiques
   ❌ Plus de maintenance
""")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 7 : SAUVEGARDE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 7 : SAUVEGARDE DU MODÈLE")
print("="*80)

# Sauvegarder modèle Keras
best_model.save('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/best_nn_model.h5')

# Sauvegarder scaler
import joblib
joblib.dump(scaler, 'e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/nn_scaler.pkl')

print("\n✓ Modèle sauvegardé : best_nn_model.h5")
print("✓ Scaler sauvegardé : nn_scaler.pkl")

print("""
📦 UTILISATION EN PRODUCTION
────────────────────────────
```python
import tensorflow as tf
import joblib
import numpy as np

# Charger
model = tf.keras.models.load_model('best_nn_model.h5')
scaler = joblib.load('nn_scaler.pkl')

# Nouvelle propriété
nouvelle_propriete = np.array([[
    120,   # Surface
    3,     # Chambres
    360,   # Surface × Chambres
    2015,  # Année construction
    # ... autres features
]])

# Normaliser
nouvelle_propriete_scaled = scaler.transform(nouvelle_propriete)

# Prédire
prix_predit_normalized = model.predict(nouvelle_propriete_scaled)[0, 0]

# Dénormaliser (si target normalisée)
# prix_predit = prix_predit_normalized * target_std + target_mean

print(f"Prix prédit : {prix_predit_normalized:.2f}")
```

⚙️  OPTIMISATION INFÉRENCE :
   Pour production haute performance :
   - Convertir en TensorFlow Lite (mobile)
   - Utiliser TensorFlow Serving (serveur)
   - Quantization pour réduire taille
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

1️⃣  QUAND UTILISER NEURAL NETWORKS
   ✅ Beaucoup de données (>10k)
   ✅ Relations complexes non-linéaires
   ✅ Images, texte, séquences
   ✅ Besoin de flexibilité
   ❌ Petites données tabulaires → XGBoost meilleur

2️⃣  ARCHITECTURE
   🏗️  Start simple : 1-2 hidden layers
   🏗️  Structure entonnoir : 128 → 64 → 32
   🏗️  Activation : ReLU (hidden), Linear (output régression)
   🏗️  Nombre params ≈ Nb échantillons / 10

3️⃣  OPTIMISATION
   ⚙️  Optimizer : ADAM par défaut (0.001)
   ⚙️  Batch size : 32 ou 64 (standard)
   ⚙️  Epochs : Beaucoup + Early Stopping
   ⚙️  Learning rate : 0.001 (défaut), ajuster si besoin

4️⃣  RÉGULARISATION
   🛡️  TOUJOURS : Early Stopping
   🛡️  Si overfitting : Dropout (0.2-0.3)
   🛡️  Si encore overfit : L2 (0.001)
   🛡️  Grands réseaux : Batch Normalization

5️⃣  DIAGNOSTIC
   📊 Courbes d'apprentissage : ESSENTIEL
   📊 Train << Val : Overfitting
   📊 Train et Val élevés : Underfitting
   📊 Résidus centrés sur 0 : Pas de biais

6️⃣  COMPARAISON AVEC MODÈLES CLASSIQUES
   ┌────────────────┬──────────┬──────────┐
   │ Critère        │ NN       │ XGBoost  │
   ├────────────────┼──────────┼──────────┤
   │ Tabulaire      │ ★★★☆☆    │ ★★★★★    │
   │ Images/Texte   │ ★★★★★    │ ★☆☆☆☆    │
   │ Setup          │ ★★☆☆☆    │ ★★★★☆    │
   │ Interprétation │ ★☆☆☆☆    │ ★★★☆☆    │
   └────────────────┴──────────┴──────────┘

✅ CHECKLIST NEURAL NETWORKS
────────────────────────────
✓ Données normalisées (CRUCIAL)
✓ Architecture simple au départ
✓ Optimizer Adam (défaut)
✓ Early Stopping configuré
✓ Dropout si overfitting
✓ Courbes d'apprentissage vérifiées
✓ Test set évalué UNE FOIS
✓ Modèle sauvegardé (.h5)

🎯 RÈGLES D'OR
──────────────
1. "TOUJOURS normaliser les données"
2. "Start simple, scale up si underfitting"
3. "Early Stopping est non-négociable"
4. "Courbes d'apprentissage = Diagnostic #1"
5. "Régulariser seulement si overfitting"

🚀 PROCHAINES ÉTAPES
────────────────────
1. CNN pour images (Tutoriel 5)
2. RNN pour séries temporelles
3. Transfer Learning
4. Hyperparameter tuning avancé (Keras Tuner)
5. Interprétabilité (SHAP, LIME)
6. Déploiement (TensorFlow Serving)

💡 BONUS : HYPERPARAMÈTRES À TUNER (PAR PRIORITÉ)
──────────────────────────────────────────────────
1. Learning rate (0.0001 - 0.01)
2. Architecture (nombre layers/neurons)
3. Dropout rate (0.1 - 0.5)
4. Batch size (16, 32, 64, 128)
5. Optimizer (Adam, SGD+momentum, RMSprop)
6. L2 regularization (0.0001 - 0.01)
7. Batch Normalization (oui/non)
8. Activation function (ReLU, LeakyReLU, tanh)

🔧 DEBUGGING TIPS
─────────────────
❌ Loss ne diminue pas :
   → Learning rate trop petit (augmenter)
   → Architecture trop simple (ajouter layers)
   → Données mal normalisées (vérifier scaler)

❌ Loss explose (NaN) :
   → Learning rate trop grand (diminuer)
   → Gradient exploding (ajouter gradient clipping)
   → Données mal normalisées

❌ Overfitting fort :
   → Dropout (0.3-0.5)
   → L2 regularization
   → Réduire taille modèle
   → Plus de données / Data augmentation

❌ Underfitting :
   → Augmenter taille modèle
   → Entraîner plus longtemps
   → Réduire régularisation
   → Vérifier que données informatives
""")

print("="*80)
print("✨ TUTORIEL TERMINÉ AVEC SUCCÈS ! ✨")
print("="*80)
print("\n🧠 Vous maîtrisez maintenant les Neural Networks !")
print("📚 Prochain tutoriel : CNN pour traitement d'images")
