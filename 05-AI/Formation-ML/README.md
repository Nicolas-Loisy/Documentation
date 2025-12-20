# Formation Complète en Machine Learning

## 📚 À Propos

Cette formation complète en Machine Learning a été créée à partir de mes notes des cours magistraux et travaux pratiques de mon Master. Elle combine théorie mathématique et pratique Python pour une compréhension approfondie du Machine Learning et du Deep Learning.

## 🎯 Objectifs Pédagogiques

À l'issue de cette formation, vous serez capable de :

- Maîtriser les fondamentaux mathématiques du Machine Learning
- Manipuler et visualiser des données avec Python (NumPy, Pandas, Matplotlib)
- Implémenter des modèles de Machine Learning supervisé et non supervisé
- Concevoir et entraîner des réseaux de neurones profonds
- Appliquer les techniques de Deep Learning (CNN, réseaux profonds)
- Évaluer et optimiser les performances des modèles

## 📖 Structure de la Formation

### Module 1 : [Introduction et Motivation](01_Introduction_et_Motivation.md)

- Définition du Machine Learning
- Applications concrètes (reconnaissance d'images, NLP, prévisions)
- Types d'apprentissage (supervisé, non supervisé, par renforcement)
- Environnement de travail Python

### Module 2 : [Algèbre Linéaire](02_Algebre_Lineaire.md)

- Vecteurs et matrices
- Opérations matricielles
- Valeurs et vecteurs propres
- Projections orthogonales
- Décomposition SVD

### Module 3 : [Probabilités](03_Probabilites.md)

- Théorie des probabilités
- Variables aléatoires
- Lois de probabilité courantes
- Théorème de Bayes
- Espérance et variance

### Module 4 : [Statistiques Descriptives](04_Statistiques_Descriptives.md)

- Mesures de tendance centrale
- Mesures de dispersion
- Visualisation de données
- Corrélation et régression linéaire
- Analyse exploratoire des données

### Module 5 : [Optimisation Numérique](05_Optimisation_Numerique.md)

- Gradient et dérivées
- Descente de gradient
- Méthodes d'optimisation (SGD, Adam, RMSprop)
- Convergence et taux d'apprentissage
- Optimisation sous contraintes

### Module 6 : [Apprentissage Supervisé](06_Apprentissage_Supervise.md)

- Régression linéaire et logistique
- Arbres de décision
- Forêts aléatoires
- SVM (Support Vector Machines)
- Évaluation des modèles (accuracy, précision, recall, F1-score)

### Module 7 : [Réseaux de Neurones Profonds](07_Reseaux_Neurones_Profonds.md)

- Perceptron et perceptron multicouche
- Fonction d'activation
- Backpropagation
- Régularisation (Dropout, L1/L2)
- Batch Normalization

### Module 8 : [Réseaux de Neurones Convolutifs (CNN)](08_CNN.md)

- Couches de convolution
- Pooling
- Architectures célèbres (LeNet, AlexNet, VGG, ResNet)
- Transfer Learning
- Applications en vision par ordinateur

### Module 9 : [Apprentissage Non Supervisé](09_Apprentissage_Non_Supervise.md)

- Clustering (K-means, DBSCAN, clustering hiérarchique)
- Réduction de dimensionnalité (PCA, t-SNE)
- Autoencodeurs
- Détection d'anomalies
- Apprentissage par renforcement (introduction)

---

## 📘 Guides Pratiques et Méthodologiques

### [Guide Complet : Démarrer un Projet ML](00_Guide_Projet_ML.md)

**Checklist complète pour mener un projet ML de A à Z**

- ✅ Checklist des 6 phases (Compréhension → Déploiement)
- 📋 Questions critiques à se poser
- 📊 Scripts d'exploration de données (EDA)
- 🔧 Stratégies de traitement des données
- 📈 Feature engineering
- 🎯 Validation et évaluation
- 📝 Templates de documentation

### [Guide de Décision : Quel Modèle ML pour Quel Problème ?](00_Guide_Decision_ML.md)

**Arbre de décision pour choisir le bon modèle**

- 🌳 Arbres de décision par type de problème
- 📊 Tableaux comparatifs détaillés
- 🔍 Quand utiliser chaque modèle (avantages/inconvénients)
- ⚙️ Guide des techniques d'optimisation
- 💡 À quoi sert la descente de gradient ?
- 🎯 Exemples de code pour chaque modèle

**Contenu détaillé :**

- Classification (Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, Naive Bayes, KNN)
- Régression (Linear, Ridge, Lasso, ElasticNet, Random Forest, XGBoost)
- Clustering (K-Means, DBSCAN, Hiérarchique, Gaussian Mixture)
- Réduction de dimensionnalité (PCA, t-SNE, UMAP, Autoencoders)

### [Workflows ML : Construire, Optimiser, Valider et Tester](00_Workflows_ML.md)

**Diagrammes et workflows étape par étape**

- 🔄 Workflow complet d'un projet ML
- 🏗️ Construction d'un modèle (code complet)
- ⚡ Optimisation (Grid Search, Random Search, diagnostics)
- ✅ Validation (K-Fold, Stratified, Time Series Split)
- 🧠 Workflow Deep Learning spécifique
- 🚀 Pipeline de production
- 🔧 Résolution de problèmes (overfitting/underfitting)

---

## 💻 Tutoriels Pratiques Détaillés

### Dossier [TUTORIELS/](TUTORIELS/)

**Tutoriels complets en Python avec explications détaillées de chaque étape, paramètre et fonction.**

#### ✅ Disponible :

- **[Tuto_01_Regression_Lineaire.py](TUTORIELS/Tuto_01_Regression_Lineaire.py)**
  - Régression linéaire simple (OLS)
  - Ridge Regression (L2)
  - Lasso Regression (L1)
  - Cross-validation
  - Analyse des résidus
  - Feature importance
  - Visualisations complètes
  - **~800 lignes de code commenté**

#### 🔜 À venir :

- Classification (Logistic Regression, Trees, Random Forest, XGBoost)
- Réseaux de neurones (Dense networks, optimisation)
- CNN pour images
- Clustering et non-supervisé

**Format des tutoriels :**

1. Théorie et formules mathématiques
2. Préparation des données
3. Entraînement du modèle
4. Optimisation des hyperparamètres
5. Validation et cross-validation
6. Évaluation finale
7. Interprétation et diagnostics
8. Sauvegarde du modèle

---

## 🛠️ Prérequis Techniques

### Logiciels

- **Python 3.7+** (recommandé : Python 3.10+)
- **Anaconda** ou **Miniconda**
- **Jupyter Notebook** ou **JupyterLab**

### Bibliothèques Python Essentielles

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

#### Manipulation de données

- **NumPy** : Calcul numérique et manipulation de tableaux
- **Pandas** : Manipulation et analyse de données
- **Matplotlib** : Visualisation de base
- **Seaborn** : Visualisation statistique avancée

#### Machine Learning

- **Scikit-learn** : Algorithmes de ML classiques
- **TensorFlow** : Framework de Deep Learning
- **Keras** : API haut niveau pour réseaux de neurones

## 📚 Ressources Complémentaires

### Livres de Référence

1. **"The Elements of Statistical Learning"** - Hastie, Tibshirani, Friedman
2. **"Pattern Recognition and Machine Learning"** - Christopher Bishop
3. **"Machine Learning: A Probabilistic Perspective"** - Kevin Murphy
4. **"Deep Learning"** - Goodfellow, Bengio, Courville
5. **"Reinforcement Learning: An Introduction"** - Sutton & Barto

### Cours en Ligne

- [Machine Learning - Andrew Ng (Coursera)](https://www.coursera.org/learn/machine-learning)
- [Deep Learning Specialization - Andrew Ng](https://www.coursera.org/specializations/deep-learning)
- [Reinforcement Learning - David Silver (UCL)](https://www.davidsilver.uk/teaching/)

### Plateformes Pratiques

- [Kaggle](https://www.kaggle.com/) - Compétitions et datasets
- [Towards Data Science](https://towardsdatascience.com/) - Articles techniques
- [Papers with Code](https://paperswithcode.com/) - Papiers de recherche avec implémentations

## 🎓 Évaluation et Pratique

Chaque module contient :

- ✅ **Théorie** : Concepts mathématiques et fondamentaux
- 💻 **Exemples pratiques** : Code Python commenté
- 🔧 **Exercices** : Applications concrètes avec solutions
- 📝 **Résumé** : Points clés à retenir

## 🚀 Comment Utiliser Cette Formation

1. **Parcours Linéaire** : Suivez les modules dans l'ordre pour une progression cohérente
2. **Parcours Thématique** : Consultez directement les modules qui vous intéressent
3. **Pratique Intensive** : Exécutez tous les exemples de code dans Jupyter Notebook
4. **Projets Personnels** : Appliquez les concepts sur vos propres données

## 📄 Licence

Ce matériel pédagogique est destiné à un usage éducatif.

---

**Bon apprentissage ! 🎉**
