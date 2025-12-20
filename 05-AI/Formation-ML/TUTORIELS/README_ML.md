# Tutoriels Machine Learning - Scripts Détaillés

Ce dossier contient des tutoriels complets en Python pour implémenter, optimiser et valider différents modèles ML.

## 📚 Tutoriels Disponibles

### ✅ Tuto_01_Regression_Lineaire.py (~800 lignes)
**Régression linéaire, Ridge et Lasso**
- Théorie et formules mathématiques (OLS, L1, L2)
- Pourquoi normaliser les données
- Régression linéaire simple, Ridge, Lasso
- Cross-validation et choix d'hyperparamètres
- Analyse des résidus et diagnostics
- Feature importance et interprétation des coefficients
- Visualisations complètes (résidus, Q-Q plot, prédictions)
- **6 observations détaillées** sur normalisation, régularisation, overfitting

### ✅ Tuto_02_Classification_Complete.py (~1000 lignes)
**Logistic Regression, Decision Trees**
- Pourquoi et quand utiliser chaque modèle (tableau décisionnel)
- Cas d'usage : prédiction de risque crédit
- Logistic Regression avec régularisation
- Decision Trees avec contrôle de profondeur
- Gestion du déséquilibre de classes (class_weight='balanced')
- Métriques : Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix et interprétation métier
- **12 observations détaillées** sur :
  - Normalisation pour Logistic Regression
  - Interprétation probabilités et coefficients
  - Overfitting des arbres
  - Class imbalance et impact Precision/Recall

### ✅ Tuto_03_Random_Forest_XGBoost.py (~1100 lignes)
**Méthodes d'ensemble : Random Forest, XGBoost, LightGBM**
- Pourquoi et quand utiliser ensembles vs modèles simples
- Cas d'usage : prédiction de churn client
- Random Forest : n_estimators, max_depth, feature importance
- XGBoost : learning_rate, early stopping, tree-based models
- LightGBM : rapidité sur gros volumes
- Comparaison des 3 méthodes (performance, rapidité)
- **7 observations détaillées** sur :
  - Baseline importance (modèle simple)
  - Feature importance et redondance
  - Early stopping pour éviter overfitting
  - Trade-off vitesse/performance

### ✅ Tuto_04_Neural_Networks.py (~1300 lignes)
**Réseaux de neurones denses (MLP)**
- Pourquoi NN vs modèles classiques (relations non-linéaires complexes)
- Quand utiliser : grandes datasets, features nombreuses, non-linéarité
- Architecture : couches cachées, nombre de neurones
- Fonctions d'activation : ReLU, Sigmoid, Tanh
- Optimiseurs : SGD, Adam, RMSprop (comparaison détaillée)
- Régularisation : Dropout, L2, Batch Normalization
- Learning rate et convergence
- **6 observations détaillées** sur :
  - Importance CRUCIALE de la normalisation
  - Choix de l'optimiseur (Adam > SGD en général)
  - Impact de Dropout sur overfitting
  - Learning rate et oscillations

### ✅ Tuto_05_CNN_Images.py (~1400 lignes)
**Réseaux de neurones convolutifs pour images**
- Pourquoi CNN pour images vs Dense Networks
- Principe de la convolution (explication visuelle avec démonstration code)
- Architecture : Convolution → Pooling → Dense
- Data Augmentation : rotation, shift, zoom (avec visualisations)
- Cas d'usage : classification MNIST (chiffres manuscrits)
- Architecture VGG-like pour MNIST
- Visualisation des feature maps (ce que le réseau "voit")
- **7 observations détaillées** sur :
  - Avantages convolution (invariance, partage de poids)
  - Pooling et réduction de dimensionnalité
  - Data Augmentation pour généralisation
  - Interprétation des feature maps

### ✅ Tuto_06_Clustering.py (~1200 lignes)
**Apprentissage non supervisé : K-Means, DBSCAN, Hiérarchique**
- Différence supervisé/non supervisé
- Pourquoi et quand utiliser chaque algorithme (tableaux comparatifs)
- Cas d'usage : segmentation client e-commerce
- K-Means : Elbow Method, Silhouette Score pour choisir k
- DBSCAN : k-distance plot pour eps optimal, détection outliers
- Clustering hiérarchique : dendrogramme, linkage methods
- PCA pour visualisation 2D
- Évaluation : Silhouette, Davies-Bouldin Index
- Interprétation métier : nommer et actionner les segments
- **7 observations détaillées** sur :
  - Interprétation Elbow Method et Silhouette
  - Profils de clusters et stratégies marketing
  - DBSCAN vs K-Means (outliers, forme clusters)
  - Validation métier vs métriques techniques

## 🎯 Format des Tutoriels

Chaque tutoriel suit cette structure :

```
1. THÉORIE
   - Formules mathématiques
   - Principes et hypothèses
   - Cas d'usage

2. PRÉPARATION DES DONNÉES
   - Chargement
   - Exploration (EDA)
   - Nettoyage
   - Feature engineering
   - Normalisation

3. BASELINE
   - Modèle simple de référence

4. MODÉLISATION
   - Entraînement
   - Paramètres expliqués
   - Prédictions

5. VALIDATION
   - Cross-validation
   - Métriques détaillées

6. OPTIMISATION
   - Hyperparamètres
   - Comparaison de variantes

7. ÉVALUATION FINALE
   - Test set
   - Analyse des erreurs

8. INTERPRÉTATION
   - Importance des features
   - Diagnostics

9. SAUVEGARDE
   - Sérialisation du modèle
```

## 💡 Comment Utiliser les Tutoriels

### Option 1 : Exécuter comme script Python

```bash
python Tuto_01_Regression_Lineaire.py
```

### Option 2 : Convertir en Jupyter Notebook

```bash
# Installer p2j si nécessaire
pip install p2j

# Convertir
p2j Tuto_01_Regression_Lineaire.py
```

### Option 3 : Copier dans Jupyter

1. Ouvrir Jupyter Notebook
2. Créer un nouveau notebook
3. Copier le code par sections
4. Exécuter cellule par cellule

## 📖 Concepts Clés Expliqués dans les Tutoriels

### 🔧 Fonctions et Paramètres Détaillés

**Preprocessing :**
- `StandardScaler()` : Pourquoi normaliser, impact sur convergence
- `train_test_split()` : Stratification, random_state
- `SimpleImputer()` : Gestion valeurs manquantes

**Modèles Classiques :**
- `LinearRegression()` : OLS, hypothèses, interprétation coefficients
- `Ridge(alpha=...)` : Régularisation L2, choix d'alpha, cross-validation
- `Lasso(alpha=...)` : Régularisation L1, sélection de features
- `LogisticRegression()` : Probabilités, C (inverse régularisation), class_weight
- `DecisionTreeClassifier()` : max_depth, min_samples_split, overfitting
- `RandomForestClassifier()` : n_estimators, max_features, feature importance
- `XGBClassifier()` : learning_rate, n_estimators, early_stopping_rounds
- `LGBMClassifier()` : num_leaves, boosting_type, rapidité

**Méthodes d'Ensemble :**
- Bagging (Random Forest) vs Boosting (XGBoost/LightGBM)
- Early stopping : éviter overfitting
- Feature importance : interpretation et diagnostics

**Deep Learning :**
- `Sequential()` : Construction modèle Keras
- `Dense(units, activation)` : Couches fully connected
- `Conv2D(filters, kernel_size)` : Convolution pour images
- `MaxPooling2D()` : Réduction dimensionnalité
- `Dropout(rate)` : Régularisation
- `BatchNormalization()` : Stabilité entraînement
- Optimizers : `Adam`, `SGD`, `RMSprop` (comparaison)
- `ImageDataGenerator()` : Data augmentation
- `fit()` : batch_size, epochs, validation_split, callbacks

**Clustering :**
- `KMeans(n_clusters)` : Centroïdes, inertie
- `DBSCAN(eps, min_samples)` : Densité, outliers
- `AgglomerativeClustering()` : Hiérarchique, linkage methods
- `PCA(n_components)` : Réduction dimensionnalité, variance expliquée
- `silhouette_score()` : Qualité clusters
- `dendrogram()` : Visualisation hiérarchie

### 📊 Métriques d'Évaluation

**Régression :**
- MSE, RMSE, MAE : Interprétation et unités
- R² : Variance expliquée
- Analyse résidus : homoscédasticité, normalité

**Classification :**
- Accuracy : Quand utiliser (classes équilibrées)
- Precision : Minimiser faux positifs
- Recall : Minimiser faux négatifs
- F1-Score : Compromis Precision/Recall
- ROC-AUC : Discrimination classes
- Confusion Matrix : Interprétation métier
- Log-Loss : Qualité probabilités

**Clustering :**
- Silhouette Score : Séparation clusters
- Davies-Bouldin Index : Chevauchement
- Inertie : Distance aux centroïdes
- Elbow Method : Choix k optimal

### 📊 Visualisations Générées par Tutoriel

**Tuto 01 (Régression) :**
- Distribution features et target
- Matrice de corrélation
- Prédictions vs réalité
- Résidus (distribution, Q-Q plot)
- Cross-validation scores
- Comparaison coefficients Ridge/Lasso
- Feature importance

**Tuto 02 (Classification) :**
- Distribution classes
- Courbes ROC (avec AUC)
- Confusion matrices
- Feature importance
- Frontières de décision (2D)
- Comparaison Logistic vs Tree

**Tuto 03 (Ensembles) :**
- Comparaison performances (barplot)
- Feature importance (3 modèles)
- Learning curves (early stopping)
- Temps d'entraînement
- Confusion matrices comparées

**Tuto 04 (Neural Networks) :**
- Loss curves (train/validation)
- Comparaison optimizers
- Impact Dropout
- Impact learning rate
- Architecture réseau (diagramme)

**Tuto 05 (CNN) :**
- Démonstration convolution (visuelle)
- Exemples data augmentation (avant/après)
- Architecture CNN (diagramme)
- Feature maps (visualisation)
- Loss/Accuracy curves
- Erreurs de classification

**Tuto 06 (Clustering) :**
- Elbow Method + Silhouette
- Silhouette plot par cluster
- PCA 2D (visualisation clusters)
- k-distance plot (DBSCAN)
- Dendrogramme (hiérarchique)
- Boxplots features par cluster
- Comparaison 3 méthodes

## 🔧 Dépendances

### Installation Complète

```bash
# Core ML et Data Science
pip install numpy pandas matplotlib seaborn scikit-learn scipy

# Deep Learning (Tuto 04, 05)
pip install tensorflow keras

# Gradient Boosting (Tuto 03)
pip install xgboost lightgbm

# Optionnel : Sauvegarde modèles
pip install joblib
```

### Versions Recommandées

- Python : 3.8+
- NumPy : 1.21+
- Pandas : 1.3+
- Scikit-learn : 1.0+
- TensorFlow : 2.8+
- XGBoost : 1.5+
- LightGBM : 3.3+

## 📊 Sorties Générées

Chaque tutoriel génère :
- **Graphiques** (.png) sauvegardés dans le même dossier
- **Modèles** (.pkl) prêts pour déploiement
- **Rapports** dans la console avec métriques détaillées

## 🎓 Utilisation Pédagogique

Ces tutoriels sont conçus pour :
- ✅ Apprendre en pratiquant
- ✅ Comprendre chaque paramètre et fonction
- ✅ Voir l'impact de chaque choix
- ✅ Suivre les meilleures pratiques
- ✅ Avoir un template réutilisable

## 💡 Conseils d'Utilisation

1. **Débutant** : Exécuter le script complet et lire les sorties
2. **Intermédiaire** : Modifier les paramètres et observer les changements
3. **Avancé** : Adapter le code à vos propres données

### Parcours d'Apprentissage Recommandé

```
┌─────────────────────────────────────────────────────────┐
│  PHASE 1 : FONDAMENTAUX                                 │
├─────────────────────────────────────────────────────────┤
│  1. Tuto_01_Regression_Lineaire.py                      │
│     → Comprendre normalisation, train/test, métriques   │
│  2. Tuto_02_Classification_Complete.py                  │
│     → Métriques classification, class imbalance         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  PHASE 2 : MÉTHODES AVANCÉES                            │
├─────────────────────────────────────────────────────────┤
│  3. Tuto_03_Random_Forest_XGBoost.py                    │
│     → Ensembles, feature importance, early stopping     │
│  4. Tuto_06_Clustering.py                               │
│     → Non supervisé, segmentation, PCA                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  PHASE 3 : DEEP LEARNING                                │
├─────────────────────────────────────────────────────────┤
│  5. Tuto_04_Neural_Networks.py                          │
│     → Architecture NN, optimizers, régularisation       │
│  6. Tuto_05_CNN_Images.py                               │
│     → Convolution, augmentation, feature maps           │
└─────────────────────────────────────────────────────────┘
```

## 📊 Récapitulatif des Tutoriels

| Tutoriel | Lignes | Observations | Cas d'Usage | Temps Estimé |
|----------|--------|--------------|-------------|--------------|
| 01 - Régression | ~800 | 6 | Prix immobilier | 1-2h |
| 02 - Classification | ~1000 | 12 | Risque crédit | 1.5-2h |
| 03 - Ensembles | ~1100 | 7 | Churn client | 1.5-2h |
| 04 - Neural Networks | ~1300 | 6 | Classification générale | 2-3h |
| 05 - CNN | ~1400 | 7 | Chiffres MNIST | 2-3h |
| 06 - Clustering | ~1200 | 7 | Segmentation client | 1.5-2h |
| **TOTAL** | **~6800** | **45** | **6 domaines** | **10-14h** |

## 🎯 Compétences Acquises

Après avoir complété les 6 tutoriels, vous maîtriserez :

✅ **Preprocessing et Feature Engineering**
- Normalisation (StandardScaler, MinMaxScaler)
- Gestion valeurs manquantes
- Création de features
- Train/test split stratifié

✅ **Modèles de Machine Learning**
- Régression : Linear, Ridge, Lasso
- Classification : Logistic, Trees, Random Forest, XGBoost
- Clustering : K-Means, DBSCAN, Hiérarchique
- Deep Learning : Dense Networks, CNN

✅ **Optimisation et Validation**
- Cross-validation (K-Fold, Stratified)
- Grid Search hyperparamètres
- Early stopping
- Régularisation (L1, L2, Dropout)

✅ **Évaluation et Diagnostics**
- Métriques : R², MSE, Accuracy, Precision, Recall, F1, ROC-AUC, Silhouette
- Confusion Matrix
- Analyse résidus
- Feature importance
- Overfitting/Underfitting detection

✅ **Visualisation**
- Matplotlib et Seaborn
- Courbes ROC, learning curves
- Feature maps (CNN)
- Dendrogrammes, PCA
- Diagnostics visuels

✅ **Production**
- Sauvegarde modèles (joblib, pickle)
- Pipeline de preprocessing
- Scoring nouveaux points
- Interprétation métier

## 🔗 Ressources Complémentaires

- [00_Guide_Projet_ML.md](../00_Guide_Projet_ML.md) - Checklist complète projet ML
- [00_Guide_Decision_ML.md](../00_Guide_Decision_ML.md) - Quel modèle choisir ?
- [00_Workflows_ML.md](../00_Workflows_ML.md) - Workflows étape par étape

## 📧 Support

Pour toute question sur les tutoriels :
- Consulter la documentation Scikit-Learn : https://scikit-learn.org/
- TensorFlow/Keras : https://www.tensorflow.org/
- Lire les commentaires détaillés dans le code (sections "OBSERVATION")
- Vérifier les guides principaux (00_Guide_*.md)

## 🏆 Prochaines Étapes

Après avoir maîtrisé ces tutoriels :

1. **Appliquer sur vos propres données**
   - Adapter les scripts à votre contexte métier
   - Expérimenter avec différents paramètres

2. **Participer à des compétitions Kaggle**
   - Mettre en pratique sur problèmes réels
   - Apprendre des kernels de la communauté

3. **Approfondir des sujets spécifiques**
   - Transfer Learning (VGG, ResNet, BERT)
   - Séries temporelles (LSTM, Prophet)
   - NLP (Transformers, BERT, GPT)
   - Reinforcement Learning

4. **Déployer en production**
   - APIs avec Flask/FastAPI
   - Conteneurisation (Docker)
   - MLOps (MLflow, Kubeflow)

---

**🎯 Formation complète de 6 tutoriels totalisant ~6800 lignes de code commenté et 45 observations détaillées !**

**Objectif : Maîtriser le ML en comprenant chaque détail, paramètre et décision !**
