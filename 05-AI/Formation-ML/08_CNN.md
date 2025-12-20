# Module 8 : Réseaux de Neurones Convolutifs (CNN)

## 📋 Table des Matières

1. [Introduction](#introduction)
2. [Limites des MLP pour les Images](#limites-des-mlp-pour-les-images)
3. [Opération de Convolution](#opération-de-convolution)
4. [Pooling](#pooling)
5. [Architecture Complète d'un CNN](#architecture-complète-dun-cnn)
6. [Architectures Célèbres](#architectures-célèbres)
7. [Transfer Learning](#transfer-learning)
8. [Implémentation Pratique](#implémentation-pratique)
9. [Projet : Classification d'Images](#projet--classification-dimages)
10. [Résumé](#résumé)

---

## Introduction

Les **Réseaux de Neurones Convolutifs** (Convolutional Neural Networks, CNN) sont spécialisés pour le traitement de données structurées en grille, notamment les **images**.

### Pourquoi les CNN ?

**Révolutions** :

- **2012** : AlexNet gagne ImageNet (classification d'images)
- **Vision par ordinateur** : State-of-the-art sur reconnaissance, détection, segmentation
- **Applications** : Voitures autonomes, diagnostic médical, reconnaissance faciale

### Applications

| Tâche                  | Description                   | Exemples           |
| ---------------------- | ----------------------------- | ------------------ |
| **Classification**     | Quelle est la classe ?        | ImageNet, CIFAR-10 |
| **Détection d'objets** | Où sont les objets ?          | YOLO, Faster R-CNN |
| **Segmentation**       | Délimiter chaque pixel        | U-Net, Mask R-CNN  |
| **Style Transfer**     | Appliquer un style artistique | DeepArt            |
| **Super-résolution**   | Améliorer la résolution       | SRGAN              |

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow version: {tf.__version__}")
```

---

## Limites des MLP pour les Images

### Problèmes avec MLP

Considérons une image $32 \times 32$ RGB :

- **Dimensions** : $32 \times 32 \times 3 = 3072$ pixels
- **Fully connected** : Chaque neurone connecté à tous les pixels

**Inconvénients** :

1. **Nombre de paramètres explosif** :

   - Première couche (1000 neurones) : $3072 \times 1000 = 3\,000\,000$ poids
   - Pour image $224 \times 224$ : $150\,528 \times 1000 = 150\,000\,000$ poids !

2. **Perte de structure spatiale** :

   - MLP traite l'image comme un vecteur 1D
   - Ignore la proximité des pixels

3. **Pas d'invariance** :
   - Un objet décalé est considéré comme différent

### Solution : CNN

**Principes** :

- **Connectivité locale** : Chaque neurone connecté à une petite région
- **Partage de poids** : Même filtre appliqué sur toute l'image
- **Hiérarchie** : Features simples → Features complexes

```python
# Exemple de dimensions
print("MLP sur image 224×224 RGB:")
print(f"  Input: {224*224*3} = 150,528 features")
print(f"  Première couche dense (1000 neurones): {150_528 * 1000:,} poids")

print("\nCNN sur même image:")
print(f"  Input: (224, 224, 3)")
print(f"  Conv2D(32 filters, 3×3): {3*3*3*32 + 32} = 896 poids")
print(f"  Réduction de facteur: {150_528_000 / 896:.0f}x !")
```

---

## Opération de Convolution

### Principe

La **convolution** applique un **filtre** (kernel) sur l'image pour détecter des features locales.

### Formule Mathématique

Pour une image $I$ et un filtre $K$ de taille $k \times k$ :

$$
(I * K)(i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I(i+m, j+n) \cdot K(m, n)
$$

### Exemple Visuel

```
Image (5×5):          Filtre (3×3):       Output (3×3):
┌─────────────┐      ┌───────┐
│ 1 2 1 0 1   │      │ 1 0 1 │           ┌───────┐
│ 0 1 2 1 0   │  *   │ 0 1 0 │    →      │ a b c │
│ 1 0 1 0 1   │      │ 1 0 1 │           │ d e f │
│ 2 1 0 1 2   │      └───────┘           │ g h i │
│ 1 0 1 2 1   │                          └───────┘
└─────────────┘

Calcul de 'e' (centre):
┌─────┐
│ 1 2 │ × │ 1 0 1 │ = 1×1 + 2×0 + 1×1 + 0×0 + 1×1 + 0×0 + 1×1 + 0×0 + 1×1 = 5
│ 0 1 │   │ 0 1 0 │
│ 1 0 │   │ 1 0 1 │
└─────┘
```

### Implémentation NumPy

```python
def conv2d_simple(image, kernel):
    """Convolution 2D simple (sans padding)"""
    h, w = image.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output

# Exemple
image = np.array([
    [1, 2, 1, 0, 1],
    [0, 1, 2, 1, 0],
    [1, 0, 1, 0, 1],
    [2, 1, 0, 1, 2],
    [1, 0, 1, 2, 1]
], dtype=np.float32)

# Filtre de détection de bords (Sobel horizontal)
kernel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
], dtype=np.float32)

result = conv2d_simple(image, kernel)

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Image Originale')
axes[0].axis('off')

axes[1].imshow(kernel, cmap='gray')
axes[1].set_title('Filtre (Kernel)')
axes[1].axis('off')

axes[2].imshow(result, cmap='gray')
axes[2].set_title('Après Convolution')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

### Hyperparamètres de Convolution

#### 1. Padding

**Problème** : La taille de l'output diminue.

**Solution** : Ajouter des zéros autour de l'image.

- **Valid** : Pas de padding → taille réduite
- **Same** : Padding pour garder même taille

$$
\text{Output size} = \frac{n + 2p - f}{s} + 1
$$

où :

- $n$ : taille de l'input
- $p$ : padding
- $f$ : taille du filtre
- $s$ : stride

```python
# Keras/TensorFlow
layers.Conv2D(32, (3, 3), padding='same')   # Garde taille
layers.Conv2D(32, (3, 3), padding='valid')  # Réduit taille
```

#### 2. Stride

**Stride** : Nombre de pixels dont on décale le filtre.

- **Stride = 1** : Déplacement pixel par pixel
- **Stride = 2** : On saute un pixel → réduit taille par 2

```python
layers.Conv2D(64, (3, 3), strides=2)  # Stride = 2
```

#### 3. Nombre de Filtres

Chaque filtre détecte une feature différente.

```python
layers.Conv2D(64, (3, 3))  # 64 filtres = 64 feature maps en sortie
```

### Filtres Classiques

```python
# Exemples de filtres
import numpy as np
import matplotlib.pyplot as plt

filtres = {
    'Identité': np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]),

    'Bords Horizontal': np.array([[ 1,  2,  1],
                                   [ 0,  0,  0],
                                   [-1, -2, -1]]),

    'Bords Vertical': np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]]),

    'Flou': np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]) / 9,

    'Sharpen': np.array([[ 0, -1,  0],
                          [-1,  5, -1],
                          [ 0, -1,  0]])
}

# Charger une image exemple
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Ou créer une image simple
img = np.random.rand(100, 100)

# Appliquer chaque filtre
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Image Originale')
axes[0].axis('off')

for idx, (name, kernel) in enumerate(filtres.items(), 1):
    result = conv2d_simple(img, kernel)
    axes[idx].imshow(result, cmap='gray')
    axes[idx].set_title(name)
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
```

---

## Pooling

Le **pooling** réduit la dimensionnalité spatiale tout en conservant les features importantes.

### Max Pooling

Prendre le **maximum** dans chaque région.

```
Input (4×4):          Max Pool 2×2:    Output (2×2):
┌─────────────┐      (stride=2)
│ 1 3 2 4 │                         ┌─────┐
│ 5 6 7 8 │      →                  │ 6 8 │
│ 9 2 1 3 │                         │ 9 7 │
│ 4 5 7 2 │                         └─────┘
└─────────────┘
```

**Avantages** :

- Réduit le calcul
- Invariance aux petites translations
- Conserve les features les plus fortes

### Average Pooling

Prendre la **moyenne** dans chaque région.

### Implémentation

```python
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D

# Max Pooling
layers.MaxPooling2D(pool_size=(2, 2), strides=2)

# Average Pooling
layers.AveragePooling2D(pool_size=(2, 2))
```

### Comparaison

```python
# Exemple
input_data = np.array([[
    [1, 3, 2, 4],
    [5, 6, 7, 8],
    [9, 2, 1, 3],
    [4, 5, 7, 2]
]], dtype=np.float32)

input_tensor = tf.constant(input_data.reshape(1, 4, 4, 1))

# Max Pooling
max_pool = tf.nn.max_pool2d(input_tensor, ksize=2, strides=2, padding='VALID')
print("Max Pooling:\n", max_pool.numpy().squeeze())

# Average Pooling
avg_pool = tf.nn.avg_pool2d(input_tensor, ksize=2, strides=2, padding='VALID')
print("\nAverage Pooling:\n", avg_pool.numpy().squeeze())
```

---

## Architecture Complète d'un CNN

### Structure Typique

```
Input Image (H×W×C)
    ↓
[Conv → ReLU → Conv → ReLU → MaxPool] × N
    ↓
Flatten
    ↓
[Dense → ReLU → Dropout] × M
    ↓
Dense (Softmax pour classification)
    ↓
Output (Classes)
```

### Exemple Concret

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    # Classifier
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())
```

### Évolution des Dimensions

```
Input:  (28, 28, 1)
    ↓
Conv2D(32, 3×3, same):  (28, 28, 32)
Conv2D(32, 3×3, same):  (28, 28, 32)
MaxPool(2×2):           (14, 14, 32)
    ↓
Conv2D(64, 3×3, same):  (14, 14, 64)
Conv2D(64, 3×3, same):  (14, 14, 64)
MaxPool(2×2):           (7, 7, 64)
    ↓
Conv2D(128, 3×3, same): (7, 7, 128)
MaxPool(2×2):           (3, 3, 128)
    ↓
Flatten:                (1152,)  # 3×3×128
Dense(256):             (256,)
Dense(10):              (10,)
```

---

## Architectures Célèbres

### 1. LeNet-5 (1998)

**Première architecture CNN** par Yann LeCun pour reconnaissance de chiffres.

```python
def LeNet5():
    model = Sequential([
        Conv2D(6, (5, 5), activation='tanh', input_shape=(28, 28, 1)),
        AveragePooling2D((2, 2)),
        Conv2D(16, (5, 5), activation='tanh'),
        AveragePooling2D((2, 2)),
        Flatten(),
        Dense(120, activation='tanh'),
        Dense(84, activation='tanh'),
        Dense(10, activation='softmax')
    ])
    return model
```

### 2. AlexNet (2012)

**Révolution ImageNet** - Première victoire d'un CNN profond.

**Innovations** :

- ReLU (au lieu de Tanh)
- Dropout
- Data Augmentation
- GPU Training

**Architecture** :

- 5 couches convolutives
- 3 couches fully connected
- 60 millions de paramètres

### 3. VGG-16 (2014)

**Principe** : Profondeur + petits filtres (3×3)

```python
from tensorflow.keras.applications import VGG16

vgg = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
print(vgg.summary())
```

**Caractéristiques** :

- 16 couches avec poids
- Filtres 3×3 uniquement
- MaxPool après chaque bloc
- 138 millions de paramètres

**Architecture** :

```
64 → 64 → MaxPool →
128 → 128 → MaxPool →
256 → 256 → 256 → MaxPool →
512 → 512 → 512 → MaxPool →
512 → 512 → 512 → MaxPool →
FC → FC → FC (4096 → 4096 → 1000)
```

### 4. ResNet (2015)

**Innovation** : **Residual Connections** (Skip Connections)

**Problème** : Réseaux très profonds difficiles à entraîner (vanishing gradients)

**Solution** : Connexions résiduelles

$$
\mathbf{y} = F(\mathbf{x}) + \mathbf{x}
$$

```python
from tensorflow.keras.applications import ResNet50

resnet = ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
```

**Variantes** :

- ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152

**Impact** : Permet de construire des réseaux très profonds (>100 couches)

### 5. Inception (GoogLeNet, 2014)

**Principe** : **Modules Inception** - convolutions parallèles de différentes tailles.

```
Input
  ├─ 1×1 Conv
  ├─ 1×1 Conv → 3×3 Conv
  ├─ 1×1 Conv → 5×5 Conv
  └─ 3×3 MaxPool → 1×1 Conv
  ↓
Concatenate
```

**Avantage** : Capture features à différentes échelles

```python
from tensorflow.keras.applications import InceptionV3

inception = InceptionV3(weights='imagenet', include_top=True, input_shape=(299, 299, 3))
```

### Comparaison

| Architecture    | Année | Profondeur | Paramètres | Top-5 Error (ImageNet) |
| --------------- | ----- | ---------- | ---------- | ---------------------- |
| LeNet-5         | 1998  | 7          | 60K        | -                      |
| AlexNet         | 2012  | 8          | 60M        | 16.4%                  |
| VGG-16          | 2014  | 16         | 138M       | 7.3%                   |
| GoogLeNet       | 2014  | 22         | 4M         | 6.7%                   |
| ResNet-152      | 2015  | 152        | 60M        | 3.57%                  |
| EfficientNet-B7 | 2019  | -          | 66M        | 2.9%                   |

---

## Transfer Learning

**Principe** : Utiliser un modèle pré-entraîné et l'adapter à une nouvelle tâche.

### Pourquoi Transfer Learning ?

**Avantages** :

- **Moins de données** nécessaires
- **Entraînement plus rapide**
- **Meilleures performances** (features génériques déjà apprises)

**Cas d'usage** :

- Dataset petit (<10,000 images)
- Ressources de calcul limitées
- Domaine proche d'ImageNet (objets, scènes)

### Stratégies

#### 1. Feature Extractor (Frozen)

Utiliser le modèle comme **extracteur de features** fixe.

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

# Charger modèle pré-entraîné (sans le top)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Geler les poids
base_model.trainable = False

# Ajouter classifier personnalisé
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())
```

#### 2. Fine-Tuning

Dégeler et réentraîner les dernières couches.

```python
# D'abord entraîner avec base frozen
model.fit(X_train, y_train, epochs=5)

# Puis dégeler les dernières couches
base_model.trainable = True

# Geler seulement les premières couches
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Recompiler avec learning rate plus faible
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # LR plus faible
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune
model.fit(X_train, y_train, epochs=10)
```

### Exemple Complet : Classification Chats vs Chiens

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Modèle pré-entraîné
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# 2. Ajouter classifier
model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binaire : chat (0) ou chien (1)
])

# 3. Compiler
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# 5. Charger données
train_generator = train_datagen.flow_from_directory(
    'data/cats_dogs/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'data/cats_dogs/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 6. Entraîner
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

---

## Implémentation Pratique

### Prétraitement d'Images

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# 1. Chargement
img = load_img('image.jpg', target_size=(224, 224))
img_array = img_to_array(img)  # (224, 224, 3)
img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

# 2. Normalisation
img_array = img_array / 255.0  # [0, 1]

# 3. Prétraitement spécifique au modèle
from tensorflow.keras.applications.vgg16 import preprocess_input
img_preprocessed = preprocess_input(img_array)
```

### Data Augmentation

```python
datagen = ImageDataGenerator(
    rotation_range=40,        # Rotation aléatoire ±40°
    width_shift_range=0.2,    # Décalage horizontal ±20%
    height_shift_range=0.2,   # Décalage vertical ±20%
    shear_range=0.2,          # Cisaillement
    zoom_range=0.2,           # Zoom ±20%
    horizontal_flip=True,     # Flip horizontal
    fill_mode='nearest'       # Remplissage des pixels
)

# Générer des images augmentées
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=32):
    # Entraîner sur batch augmenté
    model.fit(X_batch, y_batch)
```

### Visualisation des Features Maps

```python
from tensorflow.keras import Model

# Créer modèle pour extraire features intermédiaires
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Prédire
activations = activation_model.predict(img_array)

# Visualiser première couche Conv
first_layer_activation = activations[0]

plt.figure(figsize=(15, 15))
for i in range(min(32, first_layer_activation.shape[-1])):
    plt.subplot(6, 6, i+1)
    plt.imshow(first_layer_activation[0, :, :, i], cmap='viridis')
    plt.axis('off')
plt.suptitle('Feature Maps - Première Couche Conv')
plt.show()
```

---

## Projet : Classification d'Images

### CIFAR-10 Classification

```python
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Charger données
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"X_train shape: {X_train.shape}")  # (50000, 32, 32, 3)
print(f"X_test shape: {X_test.shape}")    # (10000, 32, 32, 3)

# Visualiser quelques images
plt.figure(figsize=(12, 6))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(X_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')
plt.tight_layout()
plt.show()

# 2. Prétraitement
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. Créer modèle CNN
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Classifier
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

print(model.summary())

# 4. Compiler
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
]

# 6. Entraîner
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# 7. Évaluer
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# 8. Visualiser apprentissage
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Evolution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Evolution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. Prédictions
predictions = model.predict(X_test[:20])
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test[:20], axis=1)

# Afficher
plt.figure(figsize=(15, 6))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(X_test[i])
    color = 'green' if predicted_classes[i] == true_classes[i] else 'red'
    plt.title(f'P: {class_names[predicted_classes[i]]}\nT: {class_names[true_classes[i]]}',
              color=color, fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show()
```

---

## Résumé

### Points Clés

#### Opérations Fondamentales

| Opération       | Rôle                      | Paramètres               |
| --------------- | ------------------------- | ------------------------ |
| **Convolution** | Extraction de features    | Filtres, stride, padding |
| **Activation**  | Non-linéarité             | ReLU généralement        |
| **Pooling**     | Réduction dimensionnalité | Max ou Average           |

#### Architecture Typique

```
Input → [Conv-ReLU-Conv-ReLU-Pool]×N → Flatten → Dense → Output
```

#### Architectures Célèbres

| Modèle    | Innovation              | Année |
| --------- | ----------------------- | ----- |
| LeNet-5   | Première CNN            | 1998  |
| AlexNet   | ReLU, Dropout, GPU      | 2012  |
| VGG       | Profondeur, 3×3 filtres | 2014  |
| ResNet    | Skip connections        | 2015  |
| Inception | Multi-échelle           | 2014  |

#### Transfer Learning

**Stratégies** :

1. **Feature Extractor** : Geler base, entraîner classifier
2. **Fine-Tuning** : Dégeler dernières couches, LR faible

**Avantages** :

- Moins de données
- Convergence rapide
- Meilleures performances

### Code Type CNN

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(H,W,C)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

### Bonnes Pratiques

- [ ] Normaliser images (0-1 ou standardiser)
- [ ] Data augmentation pour éviter overfitting
- [ ] BatchNormalization après Conv2D
- [ ] Dropout avant couches denses
- [ ] MaxPooling pour réduire dimensions
- [ ] Transfer learning si dataset petit
- [ ] Learning rate scheduling
- [ ] Early stopping sur validation

### Prochaine Étape

**Module 9 : Apprentissage Non Supervisé** - Clustering, PCA, autoencodeurs

---

**Navigation :**

- [⬅️ Module 7 : Réseaux de Neurones Profonds](07_Reseaux_Neurones_Profonds.md)
- [🏠 Retour au Sommaire](README.md)
- [➡️ Module 9 : Apprentissage Non Supervisé](09_Apprentissage_Non_Supervise.md)
