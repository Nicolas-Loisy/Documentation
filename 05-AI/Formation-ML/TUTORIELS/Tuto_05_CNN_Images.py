"""
═══════════════════════════════════════════════════════════════════════════════
TUTORIEL COMPLET : CNN (CONVOLUTIONAL NEURAL NETWORKS)
═══════════════════════════════════════════════════════════════════════════════

🎯 CAS D'USAGE RÉEL : Classification d'Images (MNIST et CIFAR-10)

CONTEXTE :
Système de reconnaissance automatique d'images pour :
- Reconnaissance de chiffres manuscrits (MNIST)
- Classification d'objets (CIFAR-10)

POURQUOI CNN ?
- STANDARD pour traitement d'images
- Détecte patterns locaux (edges, textures, objets)
- Invariance aux translations
- Moins de paramètres que Dense Networks

Ce tutoriel couvre :
1. POURQUOI CNN vs Dense Networks pour images
2. Architecture CNN : Convolution, Pooling, Dense
3. Fonctionnement de la Convolution (expliqué simplement)
4. Construire un CNN from scratch
5. Architectures célèbres (LeNet, AlexNet, VGG, ResNet)
6. Data Augmentation (crucial pour images)
7. Transfer Learning (utiliser modèles pré-entraînés)
8. Visualisation des features apprises
9. Optimisation et diagnostic

Chaque étape explique CE QU'IL FAUT OBSERVER et LES CONCLUSIONS.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, cifar10

from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
tf.random.set_seed(42)
np.random.seed(42)

print("="*80)
print("TUTORIEL : CNN (CONVOLUTIONAL NEURAL NETWORKS)")
print("="*80)
print(f"TensorFlow version : {tf.__version__}")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 1 : COMPRENDRE LES CNN
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 1 : POURQUOI CNN POUR LES IMAGES ?")
print("="*80)

print("""
🖼️  LE PROBLÈME DES DENSE NETWORKS SUR IMAGES
──────────────────────────────────────────────

EXEMPLE : Image 28×28 pixels (MNIST)
   Dense Network : 28 × 28 = 784 inputs

   Si 1ère hidden layer = 128 neurons :
   → Paramètres = 784 × 128 = 100,352 !

   Pour image 224×224 RGB (ImageNet) :
   → Inputs = 224 × 224 × 3 = 150,528
   → Avec 512 neurons : 150,528 × 512 = 77M paramètres !

❌ PROBLÈMES :
   1. Trop de paramètres → Overfitting
   2. Pas de notion de LOCALITÉ (pixels voisins reliés)
   3. Pas d'INVARIANCE (même objet déplacé = input différent)
   4. Pas de HIÉRARCHIE (low-level → high-level features)


✅ SOLUTION : CONVOLUTIONAL NEURAL NETWORKS
────────────────────────────────────────────

PRINCIPE : Détecter patterns LOCAUX via filtres partagés

1. CONVOLUTION : Filtres glissants détectent patterns
   → Edges, textures, formes, objets

2. POOLING : Réduction dimensionnelle
   → Robustesse aux translations

3. HIÉRARCHIE : Layers progressives
   → Layer 1 : Edges (lignes, courbes)
   → Layer 2 : Textures (grilles, motifs)
   → Layer 3 : Parties (yeux, roues)
   → Layer 4 : Objets complets (chat, voiture)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 CNN vs DENSE NETWORKS : COMPARAISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌──────────────────┬────────────────┬────────────────────────┐
│  Critère         │  Dense NN      │  CNN                   │
├──────────────────┼────────────────┼────────────────────────┤
│ Paramètres       │ ÉNORME         │ Réduit (poids partagés)│
│ Images           │ ★☆☆☆☆          │ ★★★★★                  │
│ Localité         │ ❌ Ignorée     │ ✅ Capturée            │
│ Invariance       │ ❌ Aucune      │ ✅ Translation         │
│ Hiérarchie       │ ❌ Plate       │ ✅ Hiérarchique        │
│ Tabulaire        │ ★★★★☆          │ ★☆☆☆☆                  │
│ Performance      │ ★★☆☆☆          │ ★★★★★                  │
└──────────────────┴────────────────┴────────────────────────┘


🏗️  ARCHITECTURE CNN TYPIQUE
─────────────────────────────

Input Image (28×28×1)
   ↓
[Conv 3×3, 32 filtres] → Feature Maps (26×26×32)
   ↓
[ReLU Activation]
   ↓
[MaxPooling 2×2] → (13×13×32)
   ↓
[Conv 3×3, 64 filtres] → (11×11×64)
   ↓
[ReLU]
   ↓
[MaxPooling 2×2] → (5×5×64)
   ↓
[Flatten] → Vector (1600)
   ↓
[Dense 128]
   ↓
[Dense 10] → Classes


🎯 QUAND UTILISER CNN ?
───────────────────────

✅ UTILISER CNN QUAND :
   - Données = IMAGES
   - Patterns locaux importants
   - Invariance spatiale nécessaire
   - Classification, détection, segmentation

❌ NE PAS UTILISER CNN QUAND :
   - Données tabulaires (colonnes indépendantes)
   - Pas de structure spatiale
   - Ordre des features arbitraire

💼 APPLICATIONS :
   - Classification d'images
   - Détection d'objets (YOLO, R-CNN)
   - Segmentation sémantique
   - Reconnaissance faciale
   - Médical (radiographies, IRM)
   - Véhicules autonomes
""")

input("\n▶ Appuyez sur Entrée pour continuer...")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 2 : COMPRENDRE LA CONVOLUTION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 2 : FONCTIONNEMENT DE LA CONVOLUTION")
print("="*80)

print("""
🔍 QU'EST-CE QU'UNE CONVOLUTION ?
─────────────────────────────────

PRINCIPE : Filtre (kernel) glisse sur l'image et calcule produit scalaire

EXEMPLE SIMPLE : Détection de bord vertical

Image 5×5 :            Filtre 3×3 :          Résultat :
┌─────────────┐        ┌─────────┐
│ 0 0 1 1 0 │        │ -1  0  1│         Output = Σ(Image × Filtre)
│ 0 0 1 1 0 │    ×   │ -1  0  1│    →   Valeur élevée si edge détecté
│ 0 0 1 1 0 │        │ -1  0  1│
│ 0 0 1 1 0 │        └─────────┘
│ 0 0 1 1 0 │
└─────────────┘


PARAMÈTRES CLÉS :
─────────────────

1. TAILLE DU FILTRE (kernel_size)
   - 1×1 : Mixe channels, pas de spatial
   - 3×3 : STANDARD (bon compromis)
   - 5×5 : Plus large, coûteux
   - 7×7 : Très large, rare

   💡 Recommandation : 3×3 (99% des cas)


2. NOMBRE DE FILTRES (filters)
   - Plus de filtres = Plus de patterns détectés
   - Layer 1 : 32-64 filtres
   - Layer 2 : 64-128 filtres
   - Layer 3+ : 128-512 filtres

   💡 Double généralement à chaque layer


3. STRIDE (pas de glissement)
   - stride=1 : Glisse d'1 pixel (défaut)
   - stride=2 : Glisse de 2 pixels → Réduit taille 2×

   💡 stride=1 presque toujours, pooling pour réduire


4. PADDING (remplissage)
   - valid : Pas de padding → Output plus petit
   - same : Padding → Output même taille que input

   💡 'same' permet de contrôler taille output


CALCUL DE LA TAILLE OUTPUT :
─────────────────────────────

Output size = (Input - Kernel + 2×Padding) / Stride + 1

Exemple : Input 28×28, Kernel 3×3, Padding 0, Stride 1
   Output = (28 - 3 + 0) / 1 + 1 = 26×26

Avec padding='same' et stride=1 :
   Output = Input size (28×28)


🌊 POOLING (RÉDUCTION)
──────────────────────

PRINCIPE : Réduit dimensionnalité en prenant max ou moyenne

MaxPooling 2×2 :
   ┌─────┐
   │ 1 3 │ → max = 6
   │ 2 6 │
   └─────┘

AVANTAGES :
✅ Réduit calculs (2× moins de pixels)
✅ Invariance aux petites translations
✅ Augmente champ réceptif

TYPES :
- MaxPooling : Prend maximum (STANDARD)
- AveragePooling : Prend moyenne (rare)

💡 Recommandation : MaxPooling 2×2


📊 NOMBRE DE PARAMÈTRES
───────────────────────

Conv2D(32 filtres, 3×3, input_channels=1) :
   Paramètres = (3 × 3 × 1 + 1) × 32 = 320

   Formule : (kernel_h × kernel_w × input_channels + 1) × nb_filtres

Dense(128, input=784) :
   Paramètres = (784 + 1) × 128 = 100,480

→ Conv a 300× MOINS de paramètres !
""")

# Démonstration visuelle de la convolution
print("\n📊 Démonstration : Détection de bords...\n")

# Créer image simple avec un bord
test_image = np.zeros((10, 10))
test_image[:, 5:] = 1  # Bord vertical au milieu

# Filtre de détection de bord vertical
edge_filter = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# Appliquer convolution manuellement (simplifié)
from scipy.signal import correlate2d
filtered_image = correlate2d(test_image, edge_filter, mode='valid')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(test_image, cmap='gray')
axes[0].set_title('Image Originale\n(Bord vertical au centre)')
axes[0].axis('off')

axes[1].imshow(edge_filter, cmap='gray', vmin=-1, vmax=1)
axes[1].set_title('Filtre de Détection\nde Bord Vertical')
axes[1].axis('off')

axes[2].imshow(filtered_image, cmap='gray')
axes[2].set_title('Résultat de la Convolution\n(Bord détecté = valeurs élevées)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/cnn_convolution_demo.png', dpi=100)
plt.show()

print("✓ Graphique sauvegardé : cnn_convolution_demo.png")

print("""
🔍 OBSERVATION #1 : Convolution en Action
──────────────────────────────────────────
CE QU'IL FAUT OBSERVER :

1. IMAGE ORIGINALE :
   Bord net entre noir (0) et blanc (1)

2. FILTRE :
   Valeurs négatives à gauche, positives à droite
   → Détecte transitions sombre → clair

3. RÉSULTAT :
   Valeurs élevées AU NIVEAU DU BORD
   Valeurs faibles ailleurs

💡 CONCLUSION :
   Le filtre a APPRIS à détecter ce pattern spécifique.
   CNN apprend automatiquement ces filtres pendant entraînement !

   Layer 1 : Apprend bords simples
   Layer 2 : Combine bords → Textures
   Layer 3 : Combine textures → Formes
   Layer 4 : Combine formes → Objets
""")

input("\n▶ Appuyez sur Entrée pour continuer...")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 3 : DONNÉES - MNIST
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 3 : CHARGEMENT DES DONNÉES - MNIST")
print("="*80)

print("\n📊 Chargement du dataset MNIST...\n")

# Charger MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"✓ Données chargées")
print(f"  Train : {X_train.shape[0]} images")
print(f"  Test : {X_test.shape[0]} images")
print(f"  Shape : {X_train.shape[1:]} (28×28 pixels, grayscale)")

# Visualiser quelques exemples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f'Label: {y_train[i]}')
    ax.axis('off')
plt.suptitle('Exemples MNIST', fontsize=14)
plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/cnn_mnist_samples.png', dpi=100)
plt.show()

print("\n✓ Graphique sauvegardé : cnn_mnist_samples.png")

# Préparation
print("\n⚙️  Préparation des données...\n")

# Normaliser pixels [0, 255] → [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape pour ajouter dimension channel
# (60000, 28, 28) → (60000, 28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Split train/val
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

print(f"✓ Préparation terminée")
print(f"  Train : {X_train.shape}")
print(f"  Val : {X_val.shape}")
print(f"  Test : {X_test.shape}")

print("""
💡 PRÉPARATION DES IMAGES
─────────────────────────

1. NORMALISATION [0, 1] :
   Pixels entre 0-255 → Diviser par 255
   ✅ Crucial pour convergence des NN

2. RESHAPE POUR CHANNEL :
   CNN attend format : (batch, height, width, channels)
   - Grayscale : channels = 1
   - RGB : channels = 3

3. LABELS :
   MNIST : 10 classes (chiffres 0-9)
   → Pas besoin one-hot encoding si sparse_categorical_crossentropy
   → Besoin one-hot si categorical_crossentropy
""")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 4 : CNN SIMPLE (BASELINE)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 4 : CNN SIMPLE (BASELINE)")
print("="*80)

print("\n🏗️  Construction du CNN simple...\n")

def create_simple_cnn():
    """
    CNN simple : 2 Conv + 2 Dense
    """
    model = models.Sequential([
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),

        # Dense
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dense(10, activation='softmax', name='output')
    ])

    return model

model_simple = create_simple_cnn()

model_simple.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Labels entiers (0-9)
    metrics=['accuracy']
)

print(model_simple.summary())

print(f"""
🔍 OBSERVATION #2 : Architecture CNN
─────────────────────────────────────
CE QU'IL FAUT OBSERVER DANS LE SUMMARY :

1. PROGRESSION DES SHAPES :
   Input : (28, 28, 1)
   Conv1 : (26, 26, 32) → 32 feature maps
   Pool1 : (13, 13, 32) → Divisé par 2
   Conv2 : (11, 11, 64) → 64 feature maps
   Pool2 : (5, 5, 64)
   Flatten : (1600) → 5×5×64 = 1600

2. NOMBRE DE PARAMÈTRES :
   Total : {model_simple.count_params():,} paramètres

   Conv1 : (3×3×1 + 1) × 32 = 320
   Conv2 : (3×3×32 + 1) × 64 = 18,496
   Dense1 : (1600 + 1) × 128 = 204,928
   Output : (128 + 1) × 10 = 1,290

   → Conv layers : Peu de paramètres (~20k)
   → Dense layers : Beaucoup de paramètres (~200k)

3. INTERPRÉTATION :
   - Conv capture patterns avec peu de paramètres
   - Mais Dense après Flatten reste coûteux
   - Solution moderne : Global Average Pooling au lieu de Flatten

💡 RÈGLE EMPIRIQUE :
   - Augmenter nb filtres en descendant
   - Réduire taille spatiale en descendant
   - Structure pyramidale : Large & peu profond → Étroit & profond
""")

# Entraîner
print("\n🚀 Entraînement du CNN simple...\n")

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history_simple = model_simple.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=128,
    callbacks=[early_stop],
    verbose=0
)

print(f"✓ Entraînement terminé (epoch {len(history_simple.history['loss'])})")

# Évaluer
train_acc = history_simple.history['accuracy'][-1]
val_acc = history_simple.history['val_accuracy'][-1]
test_loss, test_acc = model_simple.evaluate(X_test, y_test, verbose=0)

print(f"\n📊 Performances :")
print(f"  Train Accuracy : {train_acc:.4f}")
print(f"  Val Accuracy : {val_acc:.4f}")
print(f"  Test Accuracy : {test_acc:.4f}")

# Courbes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_simple.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history_simple.history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('CNN Simple - Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_simple.history['accuracy'], label='Train Acc', linewidth=2)
axes[1].plot(history_simple.history['val_accuracy'], label='Val Acc', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('CNN Simple - Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/cnn_simple_training.png', dpi=100)
plt.show()

print("\n✓ Graphique sauvegardé : cnn_simple_training.png")

print(f"""
🔍 OBSERVATION #3 : Performance CNN Simple
───────────────────────────────────────────
Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)

CE QU'IL FAUT OBSERVER :

1. PERFORMANCE :
   - >95% : Bon modèle
   - >97% : Très bon
   - >99% : Excellent (MNIST est "facile")

   Notre modèle : {test_acc*100:.2f}% → {"Excellent" if test_acc > 0.99 else "Très bon" if test_acc > 0.97 else "Bon"}

2. OVERFITTING :
   Écart Train - Val : {train_acc - val_acc:.4f}
   → {"Pas d'overfitting" if train_acc - val_acc < 0.02 else "Léger overfitting"}

3. CONVERGENCE :
   Arrêt epoch {len(history_simple.history['loss'])} / 20
   → {"Convergence rapide" if len(history_simple.history['loss']) < 15 else "Convergence lente"}

💡 CONCLUSION :
   CNN simple atteint déjà excellente performance sur MNIST.
   MNIST est dataset "facile" (images propres, centrées).

   Pour améliorer :
   1. Data Augmentation
   2. Architecture plus profonde
   3. Dropout / Batch Normalization
   4. Augmenter nombre de filtres
""")

input("\n▶ Appuyez sur Entrée pour continuer...")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 5 : DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 5 : DATA AUGMENTATION")
print("="*80)

print("""
🔄 POURQUOI DATA AUGMENTATION ?
────────────────────────────────

PROBLÈME :
   Dataset limité → Overfitting
   Modèle voit toujours mêmes images → Mémorise

SOLUTION :
   Créer variations artificielles des images pendant entraînement
   → Modèle voit images légèrement différentes à chaque epoch
   → Généralise mieux

TRANSFORMATIONS COURANTES :
───────────────────────────

1. ROTATION (rotation_range=10°)
   Tourne image jusqu'à ±10°
   Usage : Objets pas toujours droits

2. TRANSLATION (width_shift, height_shift=0.1)
   Décale image horizontalement/verticalement
   Usage : Objet pas toujours centré

3. ZOOM (zoom_range=0.1)
   Zoom in/out jusqu'à 10%
   Usage : Objet à différentes distances

4. FLIP (horizontal_flip=True)
   Miroir horizontal
   Usage : Objets symétriques (chat, voiture)
   ⚠️ Pas pour texte ou chiffres !

5. SHEAR (shear_range=0.1)
   Cisaillement (déformation)
   Usage : Perspectives différentes

6. BRIGHTNESS (brightness_range=[0.8, 1.2])
   Ajuste luminosité
   Usage : Conditions d'éclairage variables

⚠️  ATTENTION :
   Transformations doivent être RÉALISTES
   Ne pas augmenter si transformations changent classe !
   Ex : Flip horizontal d'un '6' → ressemble à '9'

🎯 IMPACT ATTENDU :
   - Réduit overfitting (écart train-val ↓)
   - Améliore généralisation (test acc ↑)
   - Équivalent à avoir plus de données
""")

print("\n🔄 Création du générateur d'augmentation...\n")

# Data Augmentation pour MNIST (léger car chiffres)
datagen = ImageDataGenerator(
    rotation_range=10,  # Rotation ±10°
    width_shift_range=0.1,  # Translation horizontale 10%
    height_shift_range=0.1,  # Translation verticale 10%
    zoom_range=0.1,  # Zoom ±10%
    # Pas de flip pour chiffres !
)

datagen.fit(X_train)

# Visualiser augmentation
print("📊 Exemples d'augmentation...\n")

fig, axes = plt.subplots(3, 6, figsize=(15, 8))

# Image originale
original_image = X_train[0:1]

for i, ax in enumerate(axes.flat):
    if i == 0:
        ax.imshow(original_image[0, :, :, 0], cmap='gray')
        ax.set_title('Original')
    else:
        # Générer image augmentée
        augmented = next(datagen.flow(original_image, batch_size=1))
        ax.imshow(augmented[0, :, :, 0], cmap='gray')
        ax.set_title(f'Augmenté {i}')
    ax.axis('off')

plt.suptitle('Data Augmentation - Variations d\'une même image', fontsize=14)
plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/cnn_data_augmentation.png', dpi=100)
plt.show()

print("✓ Graphique sauvegardé : cnn_data_augmentation.png")

print("""
🔍 OBSERVATION #4 : Variations Générées
────────────────────────────────────────
CE QU'IL FAUT OBSERVER :

1. VARIATIONS LÉGÈRES :
   Chiffre toujours reconnaissable
   → Transformations RÉALISTES

2. DIVERSITÉ :
   Chaque image légèrement différente
   → Modèle ne voit jamais exactement la même

3. CLASSE PRÉSERVÉE :
   Toujours le même chiffre
   → Transformations ne changent pas le label

💡 CONCLUSION :
   Data Augmentation = Régularisation puissante
   Équivalent à avoir 10-100× plus de données !

⚠️  SI TRANSFORMATIONS TROP FORTES :
   - Chiffre illisible → Peut nuire performance
   - Règle : Humain doit encore reconnaître l'objet
""")

# Entraîner avec augmentation
print("\n🚀 Entraînement avec Data Augmentation...\n")

# Nouveau modèle
model_augmented = create_simple_cnn()
model_augmented.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entraîner avec générateur
history_augmented = model_augmented.fit(
    datagen.flow(X_train, y_train, batch_size=128),
    validation_data=(X_val, y_val),
    epochs=20,
    steps_per_epoch=len(X_train) // 128,
    callbacks=[early_stop],
    verbose=0
)

print(f"✓ Entraînement terminé (epoch {len(history_augmented.history['loss'])})")

# Évaluer
test_loss_aug, test_acc_aug = model_augmented.evaluate(X_test, y_test, verbose=0)

print(f"\n📊 Comparaison :")
print(f"  Sans augmentation : {test_acc:.4f}")
print(f"  Avec augmentation : {test_acc_aug:.4f}")
print(f"  Amélioration : {(test_acc_aug - test_acc)*100:+.2f}%")

print(f"""
🔍 OBSERVATION #5 : Impact de l'Augmentation
─────────────────────────────────────────────
CE QU'IL FAUT OBSERVER :

1. TEST ACCURACY :
   Sans : {test_acc:.4f}
   Avec : {test_acc_aug:.4f}

   {"✅ Amélioration significative" if test_acc_aug > test_acc + 0.005 else "⚠️ Peu d'amélioration"}

2. OVERFITTING :
   Avec augmentation, écart train-val devrait DIMINUER
   → Modèle généralise mieux

3. CONVERGENCE :
   Peut être PLUS LENTE (images plus variées)
   → Normal, continuer plus longtemps si nécessaire

💡 INTERPRÉTATION :

Si amélioration faible :
   - MNIST est déjà "facile" (98%+ sans augmentation)
   - Augmentation plus utile sur datasets complexes (CIFAR, ImageNet)
   - Ou transformations pas assez fortes

Si amélioration significative :
   - Augmentation efficace
   - Modèle moins overfitté
   - Continue avec cette approche !

🎯 RÈGLE D'OR :
   TOUJOURS utiliser Data Augmentation sur images en production !
   Sauf si dataset ÉNORME (>1M images) et très diversifié.
""")

input("\n▶ Appuyez sur Entrée pour continuer...")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 6 : CNN PROFOND ET ARCHITECTURES CÉLÈBRES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 6 : ARCHITECTURES CNN CÉLÈBRES")
print("="*80)

print("""
🏛️  ÉVOLUTION DES ARCHITECTURES CNN
────────────────────────────────────

1️⃣  LeNet-5 (1998) - Yann LeCun
────────────────────────────────
Architecture : Conv → Pool → Conv → Pool → Dense → Dense
Usage : MNIST, reconnaissance de chiffres
Innovation : Première CNN efficace

2️⃣  AlexNet (2012) - ImageNet Winner
─────────────────────────────────────
Architecture : 5 Conv + 3 Dense
Innovation :
- ReLU (au lieu de tanh)
- Dropout
- Data Augmentation
- GPU training

Impact : Révolution Deep Learning !

3️⃣  VGG (2014)
───────────────
Architecture : Blocs de Conv 3×3 répétés
Innovation :
- PROFONDEUR (16-19 layers)
- Uniquement Conv 3×3
- Architecture simple et uniforme

Limitation : Beaucoup de paramètres (~140M)

4️⃣  ResNet (2015) - He et al.
──────────────────────────────
Architecture : Residual Blocks avec skip connections
Innovation :
- Connexions résiduelles : x + F(x)
- Permet réseaux TRÈS PROFONDS (50, 101, 152 layers)
- Résout "gradient vanishing"

Impact : Standard actuel !

5️⃣  EfficientNet (2019)
────────────────────────
Architecture : Scaling composé (depth, width, resolution)
Innovation :
- Optimisation automatique architecture
- Meilleur compromis performance/efficacité

Usage : Production avec ressources limitées


┌─────────────┬────────┬────────────┬──────────┬─────────┐
│ Architecture│ Année  │ Layers     │ Params   │ Top-1   │
├─────────────┼────────┼────────────┼──────────┼─────────┤
│ LeNet       │ 1998   │ 5          │ 60K      │ -       │
│ AlexNet     │ 2012   │ 8          │ 60M      │ 63.3%   │
│ VGG-16      │ 2014   │ 16         │ 138M     │ 71.5%   │
│ ResNet-50   │ 2015   │ 50         │ 25M      │ 76.2%   │
│ EfficientNet│ 2019   │ Variable   │ 5-66M    │ 84.4%   │
└─────────────┴────────┴────────────┴──────────┴─────────┘

Top-1 = Accuracy sur ImageNet


🎯 QUELLE ARCHITECTURE CHOISIR ?
─────────────────────────────────

SCRATCH (entraînement from scratch) :
   - Petites images (MNIST) → CNN simple
   - Images moyennes (CIFAR) → VGG-like ou ResNet
   - Grandes images → ResNet ou EfficientNet

TRANSFER LEARNING (réutiliser modèle pré-entraîné) :
   - Dataset petit (<10k) → ResNet50 pré-entraîné
   - Dataset moyen → ResNet ou EfficientNet pré-entraîné
   - Production → EfficientNet (efficace)

💡 RECOMMANDATION 2024 :
   - From scratch : Inspiration ResNet (residual blocks)
   - Transfer learning : EfficientNet ou ResNet50
   - Recherche : Vision Transformers (ViT)
""")

print("\n🏗️  Construction d'un CNN inspiré VGG...\n")

def create_vgg_like_cnn():
    """
    CNN inspiré de VGG : Blocs de Conv répétées
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Dense
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    return model

model_vgg = create_vgg_like_cnn()
model_vgg.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model_vgg.summary())

print(f"\n🚀 Entraînement du CNN profond...\n")

history_vgg = model_vgg.fit(
    datagen.flow(X_train, y_train, batch_size=128),
    validation_data=(X_val, y_val),
    epochs=20,
    steps_per_epoch=len(X_train) // 128,
    callbacks=[early_stop],
    verbose=0
)

test_loss_vgg, test_acc_vgg = model_vgg.evaluate(X_test, y_test, verbose=0)

print(f"✓ Entraînement terminé")
print(f"\n📊 Comparaison Finale :")
print(f"  CNN Simple : {test_acc:.4f}")
print(f"  + Augmentation : {test_acc_aug:.4f}")
print(f"  VGG-like : {test_acc_vgg:.4f}")

print(f"""
🔍 OBSERVATION #6 : Architecture Profonde vs Simple
────────────────────────────────────────────────────
CE QU'IL FAUT OBSERVER :

1. PERFORMANCE :
   Simple → VGG : {test_acc:.4f} → {test_acc_vgg:.4f}
   Amélioration : {(test_acc_vgg - test_acc)*100:+.2f}%

2. PARAMÈTRES :
   Simple : {model_simple.count_params():,}
   VGG-like : {model_vgg.count_params():,}
   Ratio : {model_vgg.count_params() / model_simple.count_params():.1f}×

3. TEMPS D'ENTRAÎNEMENT :
   Plus profond = Plus lent (mais plus puissant)

💡 INTERPRÉTATION :

Sur MNIST (simple) :
   - Gain modeste car dataset "facile"
   - CNN simple déjà >98%
   - VGG utile surtout si besoin 99%+

Sur datasets complexes (CIFAR, ImageNet) :
   - Gain MAJEUR avec architecture profonde
   - Nécessité de ResNet/VGG pour bonne performance

🎯 RÈGLE DE DÉCISION :
   Dataset simple (MNIST) → CNN simple suffit
   Dataset complexe (CIFAR, ImageNet) → Architecture profonde nécessaire

💡 OVERFITTING ?
   Si VGG overfit :
   - Augmenter Dropout (0.25 → 0.4)
   - Plus de Data Augmentation
   - Batch Normalization
   - Plus de données
""")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 7 : VISUALISATION DES FEATURES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PARTIE 7 : VISUALISATION DES FEATURES APPRISES")
print("="*80)

print("\n🔍 Visualisation des activations...\n")

# Prendre une image de test
test_image = X_test[0:1]
true_label = y_test[0]

# Créer modèle pour extraire activations intermédiaires
layer_outputs = [layer.output for layer in model_vgg.layers[:6]]  # Premiers 6 layers
activation_model = models.Model(inputs=model_vgg.input, outputs=layer_outputs)

# Obtenir activations
activations = activation_model.predict(test_image, verbose=0)

# Visualiser
layer_names = ['conv2d', 'conv2d_1', 'max_pooling2d', 'conv2d_2', 'conv2d_3', 'max_pooling2d_1']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, (ax, activation, name) in enumerate(zip(axes.flat, activations, layer_names)):
    if 'conv' in name:
        # Montrer premier filtre
        ax.imshow(activation[0, :, :, 0], cmap='viridis')
        ax.set_title(f'{name}\nFiltre 0')
    else:
        # Pooling
        ax.imshow(activation[0, :, :, 0], cmap='viridis')
        ax.set_title(f'{name}')
    ax.axis('off')

plt.suptitle(f'Activations pour chiffre {true_label}', fontsize=14)
plt.tight_layout()
plt.savefig('e:/Nicolas/MIAGE/M2/BigData/FORMATION_ML/TUTORIELS/cnn_activations.png', dpi=100)
plt.show()

print("✓ Graphique sauvegardé : cnn_activations.png")

print("""
🔍 OBSERVATION #7 : Features Apprises
──────────────────────────────────────
CE QU'IL FAUT OBSERVER :

1. LAYER 1 (Conv) :
   Détecte bords, lignes, courbes simples
   → Features BAS NIVEAU (low-level)

2. LAYER 2 (Conv) :
   Combine bords → Textures, motifs
   → Features INTERMÉDIAIRES

3. APRÈS POOLING :
   Résolution réduite mais information préservée
   → Abstraction progressive

4. LAYERS PROFONDS :
   Features de plus en plus abstraites
   → Représentations HAUT NIVEAU (high-level)

💡 INTERPRÉTATION :
   CNN apprend HIÉRARCHIE de features :
   Pixels → Bords → Textures → Parties → Objets

   C'est pourquoi CNN excelle sur images :
   Mimique système visuel humain !

🎯 DEBUGGING :
   Si activations nulles (toutes noires) :
   → Neurones "morts" (dying ReLU)
   → Solutions :
     - Utiliser LeakyReLU
     - Réduire learning rate
     - Vérifier normalisation input
""")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 8 : RÉSUMÉ ET CONCLUSIONS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🎉 RÉSUMÉ ET CONCLUSIONS")
print("="*80)

print(f"""
📚 CE QUE NOUS AVONS APPRIS
───────────────────────────

1️⃣  POURQUOI CNN POUR IMAGES
   ✅ Détecte patterns locaux (bords, textures)
   ✅ Poids partagés → Moins de paramètres
   ✅ Invariance aux translations
   ✅ Hiérarchie de features (bas → haut niveau)
   ❌ Pas pour données tabulaires

2️⃣  COMPOSANTS CNN
   🔷 Conv2D : Détection de patterns
      - Filters : 32 → 64 → 128 (doubler)
      - Kernel : 3×3 (standard)
      - Padding : 'same' (garder taille)

   🔷 MaxPooling : Réduction dimensionnelle
      - Taille : 2×2 (standard)
      - Réduit overfitting
      - Invariance spatiale

   🔷 Flatten : Conv → Dense
      - Ou Global Average Pooling (moderne)

   🔷 Dense : Classification finale

3️⃣  DATA AUGMENTATION
   🔄 ESSENTIEL pour images !
   🔄 Transformations réalistes
   🔄 Équivalent à 10-100× plus de données
   🔄 Réduit overfitting significativement

4️⃣  ARCHITECTURES
   🏛️  Simple : 2-3 Conv blocks
   🏛️  VGG : Conv répétées + profondeur
   🏛️  ResNet : Skip connections (moderne)
   🏛️  EfficientNet : Optimisé (production)

5️⃣  RÉSULTATS SUR MNIST
   📊 CNN Simple : {test_acc*100:.2f}%
   📊 + Augmentation : {test_acc_aug*100:.2f}%
   📊 VGG-like : {test_acc_vgg*100:.2f}%

   → CNN atteint facilement >98% sur MNIST


✅ CHECKLIST CNN
────────────────
✓ Images normalisées [0, 1]
✓ Shape correcte (batch, H, W, C)
✓ Architecture progressive (filtres ↑, taille ↓)
✓ Data Augmentation configurée
✓ Dropout entre Dense layers
✓ Early Stopping
✓ Learning rate scheduling
✓ Visualisation des activations (debugging)


🎯 RÈGLES D'OR CNN
──────────────────
1. "TOUJOURS normaliser images [0, 1]"
2. "Conv 3×3 est le standard (99% des cas)"
3. "Doubler filtres, diviser taille par 2"
4. "Data Augmentation est NON-NÉGOCIABLE"
5. "Start simple, go deep si underfitting"


🚀 PROCHAINES ÉTAPES
────────────────────
1. Transfer Learning (réutiliser modèles pré-entraînés)
2. CIFAR-10 (images couleur, plus complexe)
3. Object Detection (YOLO, Faster R-CNN)
4. Segmentation Sémantique (U-Net)
5. GANs (Generative Adversarial Networks)
6. Vision Transformers (ViT)


💡 TRANSFER LEARNING (APERÇU)
──────────────────────────────
Au lieu d'entraîner from scratch :

```python
# Charger ResNet50 pré-entraîné sur ImageNet
base_model = applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Geler les layers
base_model.trainable = False

# Ajouter classification personnalisée
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```

AVANTAGES :
✅ Converge 10-100× plus vite
✅ Meilleure performance avec peu de données
✅ Features génériques déjà apprises

USAGE :
- Dataset < 10k images : ESSENTIEL
- Dataset > 100k : Optionnel mais accélère


🔧 HYPERPARAMÈTRES À TUNER (PAR PRIORITÉ)
──────────────────────────────────────────
1. Data Augmentation (rotation, shift, zoom)
2. Architecture (nb layers, filtres)
3. Dropout rate (0.25-0.5)
4. Learning rate (0.0001-0.01)
5. Batch size (32, 64, 128)
6. Optimizer (Adam, SGD+momentum)


🐛 DEBUGGING CNN
────────────────
❌ Accuracy stagne ~10% (random) :
   → Modèle ne converge pas
   → Vérifier normalisation images [0, 1]
   → Vérifier labels corrects
   → Réduire learning rate

❌ Loss = NaN :
   → Learning rate trop grand
   → Gradient exploding
   → Vérifier pas de valeurs infinies dans data

❌ Overfitting fort :
   → Plus de Data Augmentation
   → Dropout (0.3-0.5)
   → L2 regularization
   → Réduire nombre de filtres
   → Plus de données

❌ Underfitting :
   → Architecture trop simple
   → Augmenter filtres (32 → 64 → 128)
   → Ajouter Conv layers
   → Réduire Dropout
   → Entraîner plus longtemps


📊 PERFORMANCES ATTENDUES
─────────────────────────
MNIST : 99%+ (facile)
Fashion-MNIST : 90-93%
CIFAR-10 : 85-95% (selon architecture)
ImageNet : 75-85% (state-of-the-art)


💪 VOUS MAÎTRISEZ MAINTENANT :
──────────────────────────────
✅ Principes des CNN
✅ Architecture (Conv, Pooling, Dense)
✅ Data Augmentation
✅ Entraînement et optimisation
✅ Visualisation des features
✅ Debugging et amélioration

→ Prêt pour projets de vision par ordinateur !
""")

print("="*80)
print("✨ TUTORIEL TERMINÉ AVEC SUCCÈS ! ✨")
print("="*80)
print("\n🎨 Vous maîtrisez maintenant les CNN !")
print("📚 Dernier tutoriel : Clustering (Apprentissage Non Supervisé)")
