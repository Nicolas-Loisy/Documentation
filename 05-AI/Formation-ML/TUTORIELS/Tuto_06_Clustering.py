"""
═══════════════════════════════════════════════════════════════════════════════
TUTORIEL 06 : CLUSTERING ET APPRENTISSAGE NON SUPERVISÉ
═══════════════════════════════════════════════════════════════════════════════

📚 OBJECTIFS :
    - Comprendre les différents algorithmes de clustering
    - Savoir choisir entre K-Means, DBSCAN, et Clustering Hiérarchique
    - Évaluer la qualité des clusters (Silhouette, Elbow Method)
    - Réduire la dimensionnalité pour visualiser (PCA)
    - Interpréter les segments obtenus

🎯 CAS D'USAGE :
    - Segmentation client (marketing)
    - Détection d'anomalies
    - Organisation de documents
    - Analyse de comportements
    - Compression d'images

═══════════════════════════════════════════════════════════════════════════════
PARTIE 1 : THÉORIE ET DÉCISION
═══════════════════════════════════════════════════════════════════════════════
"""

print("="*80)
print("PARTIE 1 : POURQUOI UTILISER LE CLUSTERING ?")
print("="*80)

print("""
🔍 CLUSTERING = APPRENTISSAGE NON SUPERVISÉ

   Pas de labels (y) ! On cherche des groupes naturels dans les données.

   ┌─────────────────────────────────────────────────────────────┐
   │  DIFFÉRENCE FONDAMENTALE                                     │
   ├─────────────────────────────────────────────────────────────┤
   │  Supervisé   : X, y  → Modèle prédit y                      │
   │  Non supervisé : X   → Modèle trouve des groupes            │
   └─────────────────────────────────────────────────────────────┘

💡 POURQUOI UTILISER LE CLUSTERING ?

   1. SEGMENTATION CLIENT
      → Identifier des profils clients similaires
      → Marketing ciblé, personnalisation

   2. DÉTECTION D'ANOMALIES
      → Points qui n'appartiennent à aucun cluster
      → Fraude, défaillances techniques

   3. COMPRESSION/ORGANISATION
      → Regrouper documents similaires
      → Compression d'images (K-means sur couleurs)

   4. PREPROCESSING
      → Créer de nouvelles features (cluster_id)
      → Identifier des sous-populations avant modèle supervisé
""")

print("\n" + "="*80)
print("COMPARAISON DES ALGORITHMES DE CLUSTERING")
print("="*80)

print("""
┌─────────────────┬──────────────┬──────────────┬─────────────────┐
│   ALGORITHME    │   K-MEANS    │   DBSCAN     │  HIÉRARCHIQUE   │
├─────────────────┼──────────────┼──────────────┼─────────────────┤
│ Principe        │ Centroïdes   │ Densité      │ Agglomération   │
│                 │ + distance   │ spatiale     │ successive      │
├─────────────────┼──────────────┼──────────────┼─────────────────┤
│ Forme clusters  │ Sphériques   │ Arbitraires  │ Arbitraires     │
├─────────────────┼──────────────┼──────────────┼─────────────────┤
│ Nb clusters     │ À SPÉCIFIER  │ AUTOMATIQUE  │ À SPÉCIFIER     │
├─────────────────┼──────────────┼──────────────┼─────────────────┤
│ Outliers        │ NON          │ OUI (noise)  │ NON (ou isolés) │
├─────────────────┼──────────────┼──────────────┼─────────────────┤
│ Scalabilité     │ ✓✓✓ Rapide   │ ✓✓ Moyen     │ ✗ Lent (>10k)   │
├─────────────────┼──────────────┼──────────────┼─────────────────┤
│ Complexité      │ O(n·k·i)     │ O(n log n)   │ O(n²) ou O(n³)  │
└─────────────────┴──────────────┴──────────────┴─────────────────┘

✅ UTILISER K-MEANS QUAND :
   - Nombre de clusters connu approximativement
   - Clusters de forme sphérique/convexe
   - Besoin de RAPIDITÉ (gros volumes)
   - Toutes les données sont pertinentes (pas d'outliers)

✅ UTILISER DBSCAN QUAND :
   - Nombre de clusters INCONNU
   - Clusters de forme ARBITRAIRE (spirales, anneaux...)
   - Présence d'OUTLIERS à identifier
   - Densité variable dans l'espace

✅ UTILISER CLUSTERING HIÉRARCHIQUE QUAND :
   - Besoin de VISUALISER la hiérarchie (dendrogramme)
   - Petit dataset (<10 000 points)
   - Exploration : tester plusieurs nb de clusters
   - Besoin de clusters imbriqués
""")

print("\n" + "="*80)
print("COMMENT CHOISIR LE NOMBRE DE CLUSTERS ?")
print("="*80)

print("""
🔧 MÉTHODES POUR K-MEANS :

1. ELBOW METHOD (Méthode du coude)
   → Tracer inertie (somme distances²) vs k
   → Chercher le "coude" où la décroissance ralentit

   Inertie
      │    \\
      │     \\___    ← "Coude" ici : k optimal
      │          \\___
      └──────────────────> k

2. SILHOUETTE SCORE
   → Mesure combien chaque point est proche de son cluster
      vs clusters voisins
   → Score de -1 (mal classé) à +1 (bien classé)
   → Viser score > 0.5

   Formule : s = (b - a) / max(a, b)
   où :
     a = distance moyenne intra-cluster
     b = distance moyenne au cluster voisin le plus proche

3. BUSINESS KNOWLEDGE
   → Le nombre de segments doit avoir du SENS métier
   → Ex : 3-5 segments clients (Bronze/Silver/Gold/Platinum)

🔧 POUR DBSCAN : CHOISIR eps ET min_samples

   eps         : Rayon de voisinage
   min_samples : Nb min de points pour former un cluster

   Méthode :
   1. Calculer k-distance plot (distance au k-ème voisin)
   2. Chercher le "coude" → valeur eps
   3. min_samples ≈ 2 × nb_features (règle empirique)
""")

print("\n" + "="*80)
print("PARTIE 2 : PRÉPARATION DES DONNÉES")
print("="*80)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Configuration
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')

print("""
📊 CAS D'USAGE : SEGMENTATION CLIENT

Contexte : Entreprise e-commerce avec données clients
Objectif : Identifier des segments pour stratégie marketing ciblée

Features disponibles :
  - Age : Âge du client
  - Income : Revenu annuel (k€)
  - SpendingScore : Score de dépenses (1-100)
  - Recency : Jours depuis dernier achat
  - Frequency : Nombre d'achats sur l'année
""")

# Génération de données réalistes de segmentation client
np.random.seed(42)
n_samples = 500

# Segment 1 : Jeunes, revenus moyens, dépensiers
age_1 = np.random.normal(28, 5, n_samples//5)
income_1 = np.random.normal(45, 10, n_samples//5)
spending_1 = np.random.normal(75, 10, n_samples//5)
recency_1 = np.random.normal(15, 5, n_samples//5)
frequency_1 = np.random.normal(25, 5, n_samples//5)

# Segment 2 : Âge moyen, hauts revenus, très dépensiers
age_2 = np.random.normal(45, 7, n_samples//5)
income_2 = np.random.normal(85, 15, n_samples//5)
spending_2 = np.random.normal(85, 8, n_samples//5)
recency_2 = np.random.normal(10, 3, n_samples//5)
frequency_2 = np.random.normal(35, 7, n_samples//5)

# Segment 3 : Seniors, revenus élevés, peu dépensiers
age_3 = np.random.normal(60, 8, n_samples//5)
income_3 = np.random.normal(75, 12, n_samples//5)
spending_3 = np.random.normal(35, 10, n_samples//5)
recency_3 = np.random.normal(45, 15, n_samples//5)
frequency_3 = np.random.normal(8, 3, n_samples//5)

# Segment 4 : Jeunes, faibles revenus, économes
age_4 = np.random.normal(25, 4, n_samples//5)
income_4 = np.random.normal(30, 8, n_samples//5)
spending_4 = np.random.normal(25, 8, n_samples//5)
recency_4 = np.random.normal(60, 20, n_samples//5)
frequency_4 = np.random.normal(5, 2, n_samples//5)

# Segment 5 : Âge moyen, revenus moyens, modérés
age_5 = np.random.normal(40, 10, n_samples//5)
income_5 = np.random.normal(55, 12, n_samples//5)
spending_5 = np.random.normal(50, 12, n_samples//5)
recency_5 = np.random.normal(30, 10, n_samples//5)
frequency_5 = np.random.normal(15, 5, n_samples//5)

# Combiner tous les segments
X = np.column_stack([
    np.concatenate([age_1, age_2, age_3, age_4, age_5]),
    np.concatenate([income_1, income_2, income_3, income_4, income_5]),
    np.concatenate([spending_1, spending_2, spending_3, spending_4, spending_5]),
    np.concatenate([recency_1, recency_2, recency_3, recency_4, recency_5]),
    np.concatenate([frequency_1, frequency_2, frequency_3, frequency_4, frequency_5])
])

# Créer DataFrame
df = pd.DataFrame(X, columns=['Age', 'Income', 'SpendingScore', 'Recency', 'Frequency'])

# Ajouter du bruit pour rendre plus réaliste
df = df + np.random.normal(0, 2, df.shape)
df = df.clip(lower=0)  # Pas de valeurs négatives

print("\n📊 APERÇU DES DONNÉES :")
print(df.head(10))
print(f"\nShape : {df.shape}")
print(f"\nStatistiques descriptives :")
print(df.describe())

print("\n🔍 ANALYSE EXPLORATOIRE :")
print(f"Valeurs manquantes : {df.isnull().sum().sum()}")
print(f"Duplicatas : {df.duplicated().sum()}")

# Matrice de corrélation
print("\n📈 CORRÉLATIONS ENTRE VARIABLES :")
print(df.corr().round(3))

print("""
💡 CE QU'IL FAUT OBSERVER DANS LES CORRÉLATIONS :

   - Income vs SpendingScore : Corrélation positive attendue
     (revenus ↑ → dépenses ↑)

   - Recency vs Frequency : Corrélation négative attendue
     (achats fréquents → recency faible)

   - Age vs Income : Peut être corrélé (carrière → revenus)

   ⚠️ Si corrélations TRÈS ÉLEVÉES (>0.9) :
      → Redondance, considérer supprimer une feature
      → Ou utiliser PCA pour décorréler
""")

print("\n" + "="*80)
print("NORMALISATION DES DONNÉES")
print("="*80)

print("""
⚠️ NORMALISATION EST CRUCIALE POUR LE CLUSTERING !

Pourquoi ?
   - K-Means, DBSCAN utilisent des DISTANCES
   - Si Income (0-100k) et Age (20-80) : Income domine !
   - Clusters biaisés par les features à grande échelle

Méthode : StandardScaler (z-score)

   z = (x - μ) / σ

   Transforme chaque feature : moyenne=0, écart-type=1
""")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

print(f"✓ Données normalisées : shape {X_scaled.shape}")
print(f"  Moyennes après scaling : {X_scaled.mean(axis=0).round(3)}")
print(f"  Écarts-types après scaling : {X_scaled.std(axis=0).round(3)}")

print("\n" + "="*80)
print("PARTIE 3 : K-MEANS CLUSTERING")
print("="*80)

print("""
🔧 ALGORITHME K-MEANS

1. Initialiser k centroïdes aléatoirement
2. RÉPÉTER jusqu'à convergence :
   a) Assigner chaque point au centroïde le plus proche
   b) Recalculer centroïdes = moyenne des points assignés

Métriques :
   - Inertie : Somme des distances² aux centroïdes
   - Silhouette : Qualité de séparation des clusters
""")

# ÉTAPE 1 : ELBOW METHOD (trouver k optimal)
print("\n📊 ÉTAPE 1 : ELBOW METHOD")

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

    print(f"k={k} → Inertie: {kmeans.inertia_:.2f}, Silhouette: {silhouette_score(X_scaled, kmeans.labels_):.3f}")

# Visualisation Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Nombre de clusters (k)', fontsize=12)
axes[0].set_ylabel('Inertie (Within-Cluster Sum of Squares)', fontsize=12)
axes[0].set_title('Elbow Method : Trouver k optimal', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=5, color='red', linestyle='--', label='k optimal suggéré')
axes[0].legend()

# Silhouette plot
axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Nombre de clusters (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score par k', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0.5, color='orange', linestyle='--', label='Seuil acceptable (0.5)')
axes[1].axvline(x=5, color='red', linestyle='--', label='k optimal suggéré')
axes[1].legend()

plt.tight_layout()
plt.savefig('E:\\Nicolas\\MIAGE\\M2\\BigData\\FORMATION_ML\\TUTORIELS\\06_elbow_silhouette.png', dpi=100, bbox_inches='tight')
print("\n✓ Graphique sauvegardé : 06_elbow_silhouette.png")
plt.close()

print("""
═══════════════════════════════════════════════════════════════════════════════
📊 OBSERVATION #1 : INTERPRÉTATION ELBOW METHOD
═══════════════════════════════════════════════════════════════════════════════

CE QU'IL FAUT OBSERVER :

1. COURBE INERTIE (Elbow Plot)
   → Décroissance forte au début, puis ralentissement
   → Chercher le "COUDE" : point où pente change drastiquement

   k=2 → k=3 : Grosse baisse (beaucoup d'information gagnée)
   k=5 → k=6 : Faible baisse (peu de gain)

2. SILHOUETTE SCORE
   → Maximum autour de k=5 (probablement)
   → Score > 0.5 : Clusters bien séparés
   → Score < 0.3 : Mauvaise séparation

💡 CONCLUSION :

   ✓ k=5 semble optimal (coude + silhouette max)
   ✓ Correspond aux 5 segments générés (validation !)

   En pratique :
   - Tester k=4, k=5, k=6
   - Interpréter les clusters avec BUSINESS KNOWLEDGE
   - Un k avec moins bon silhouette mais meilleure interprétation
     métier peut être préférable !

⚠️ ATTENTION :
   Elbow pas toujours clair (courbe lisse)
   → Combiner avec silhouette ET connaissance métier
═══════════════════════════════════════════════════════════════════════════════
""")

# ÉTAPE 2 : ENTRAÎNER K-MEANS AVEC k OPTIMAL
print("\n📊 ÉTAPE 2 : ENTRAÎNEMENT K-MEANS FINAL (k=5)")

kmeans_final = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters_kmeans = kmeans_final.fit_predict(X_scaled)

print(f"✓ K-Means entraîné avec k=5")
print(f"  Inertie finale : {kmeans_final.inertia_:.2f}")
print(f"  Silhouette score : {silhouette_score(X_scaled, clusters_kmeans):.3f}")
print(f"  Davies-Bouldin Index : {davies_bouldin_score(X_scaled, clusters_kmeans):.3f}")
print("    (Plus bas = meilleur, mesure chevauchement clusters)")

# Ajouter les clusters au DataFrame
df['Cluster_KMeans'] = clusters_kmeans

print("\n📊 DISTRIBUTION DES CLUSTERS :")
print(df['Cluster_KMeans'].value_counts().sort_index())

print("\n📊 PROFIL DES CLUSTERS (Moyennes par segment) :")
cluster_profiles = df.groupby('Cluster_KMeans').mean().round(2)
print(cluster_profiles)

print("""
═══════════════════════════════════════════════════════════════════════════════
📊 OBSERVATION #2 : INTERPRÉTATION DES PROFILS DE CLUSTERS
═══════════════════════════════════════════════════════════════════════════════

CE QU'IL FAUT OBSERVER :

1. CARACTÉRISTIQUES DISTINCTIVES de chaque cluster
   → Quelles features diffèrent le plus entre clusters ?
   → Chercher des patterns cohérents

2. TAILLE DES CLUSTERS
   → Clusters très déséquilibrés ? (ex: 400 vs 20 points)
   → Peut indiquer outliers ou segment de niche

3. INTERPRÉTATION MÉTIER

   Exemple de profils typiques :

   Cluster 0 : "JEUNES DÉPENSIERS"
     Age faible, Income moyen, SpendingScore élevé
     → Stratégie : Offres tendance, réseaux sociaux

   Cluster 1 : "PREMIUM"
     Age moyen, Income haut, Frequency élevée
     → Stratégie : Programme fidélité premium, services VIP

   Cluster 2 : "DORMANTS"
     Recency élevé, Frequency faible
     → Stratégie : Campagne de réactivation

   Cluster 3 : "ÉCONOMES"
     SpendingScore faible, Recency élevé
     → Stratégie : Promotions, codes promo

   Cluster 4 : "STABLES"
     Valeurs moyennes sur toutes features
     → Stratégie : Marketing générique

💡 CONCLUSION :

   ✓ Nommer chaque cluster selon profil
   ✓ Définir stratégie marketing par segment
   ✓ Calculer LTV (Lifetime Value) par segment

⚠️ ATTENTION :
   - Ne pas sur-interpréter de petits clusters (<5% données)
   - Valider avec équipes métier (Marketing, Sales)
═══════════════════════════════════════════════════════════════════════════════
""")

# ÉTAPE 3 : VISUALISATION AVEC PCA
print("\n📊 ÉTAPE 3 : RÉDUCTION DIMENSIONNELLE POUR VISUALISATION (PCA)")

print("""
❓ POURQUOI PCA ?

   - Nos données : 5 dimensions (Age, Income, ...)
   - Impossible de visualiser en 5D !
   - PCA réduit à 2D en conservant maximum de variance

   PCA (Principal Component Analysis) :

   1. Trouve directions de variance maximale
   2. Projette données sur ces directions (composantes principales)
   3. PC1 = direction de variance max, PC2 = 2ème direction, etc.
""")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"✓ PCA appliquée : 5D → 2D")
print(f"  Variance expliquée par PC1 : {pca.explained_variance_ratio_[0]:.2%}")
print(f"  Variance expliquée par PC2 : {pca.explained_variance_ratio_[1]:.2%}")
print(f"  Variance totale conservée : {pca.explained_variance_ratio_.sum():.2%}")

# Visualisation des clusters en 2D
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_kmeans,
                      cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
plt.title('K-Means Clustering (k=5) - Vue PCA', fontsize=13, fontweight='bold')
plt.colorbar(scatter, label='Cluster')

# Ajouter les centroïdes
centroids_pca = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            c='red', marker='X', s=300, edgecolors='black', linewidth=2, label='Centroïdes')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Silhouette diagram
from matplotlib import cm
silhouette_vals = silhouette_samples(X_scaled, clusters_kmeans)
y_lower = 10
for i in range(5):
    cluster_silhouette_vals = silhouette_vals[clusters_kmeans == i]
    cluster_silhouette_vals.sort()
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.viridis(float(i) / 5)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                      facecolor=color, edgecolor=color, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=12, fontweight='bold')
    y_lower = y_upper + 10

plt.axvline(x=silhouette_score(X_scaled, clusters_kmeans), color="red", linestyle="--",
            label=f'Silhouette moyenne: {silhouette_score(X_scaled, clusters_kmeans):.3f}')
plt.xlabel('Coefficient de Silhouette', fontsize=11)
plt.ylabel('Cluster', fontsize=11)
plt.title('Silhouette Plot par Cluster', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('E:\\Nicolas\\MIAGE\\M2\\BigData\\FORMATION_ML\\TUTORIELS\\06_kmeans_visualization.png', dpi=100, bbox_inches='tight')
print("\n✓ Graphique sauvegardé : 06_kmeans_visualization.png")
plt.close()

print("""
═══════════════════════════════════════════════════════════════════════════════
📊 OBSERVATION #3 : QUALITÉ DES CLUSTERS (SILHOUETTE)
═══════════════════════════════════════════════════════════════════════════════

CE QU'IL FAUT OBSERVER :

1. SILHOUETTE PLOT (Graphique de droite)

   → Chaque "étage" = un cluster
   → Largeur = taille du cluster
   → Longueur des barres = coefficient de silhouette individuel

   ✓ BON SIGNE :
     - Toutes les barres dépassent la ligne rouge (moyenne)
     - Barres longues et uniformes (cluster cohésif)
     - Peu de barres négatives

   ✗ MAUVAIS SIGNE :
     - Barres très courtes ou négatives (points mal classés)
     - Variation forte au sein d'un cluster (hétérogène)
     - Clusters de tailles très inégales

2. VISUALISATION PCA (Graphique de gauche)

   → Clusters bien séparés visuellement ?
   → Centroïdes (X rouges) au centre de leurs clusters ?

   ⚠️ ATTENTION : PCA ne conserve que ~60-70% variance
      → Séparation en 2D peut être trompeuse
      → Clusters peuvent se chevaucher en 2D mais être distincts en 5D

💡 CONCLUSION :

   ✓ Si silhouette moyenne > 0.5 ET barres uniformes :
     → Clusters de bonne qualité, bien séparés

   ✓ Si certains clusters ont silhouette faible :
     → Peuvent être fusionnés (réduire k)
     → Ou contenir outliers (considérer DBSCAN)

   ✓ Variance PCA > 60% :
     → Visualisation 2D représentative

⚠️ SI SILHOUETTE < 0.3 :
   → Revoir le nombre de clusters
   → Ou les données ne sont pas naturellement clusterisables
   → Essayer DBSCAN ou clustering hiérarchique
═══════════════════════════════════════════════════════════════════════════════
""")

print("\n" + "="*80)
print("PARTIE 4 : DBSCAN CLUSTERING")
print("="*80)

print("""
🔧 ALGORITHME DBSCAN (Density-Based Spatial Clustering)

Principe :
   - Grouper points dans régions DENSES
   - Identifier OUTLIERS (points isolés)

Paramètres :
   eps         : Rayon de voisinage (epsilon)
   min_samples : Nb min de points pour former un cluster

Types de points :
   - CORE : ≥ min_samples voisins dans rayon eps
   - BORDER : < min_samples voisins, mais proche d'un core point
   - NOISE : Ni core ni border (OUTLIER)

Avantages :
   ✓ Détecte nombre de clusters automatiquement
   ✓ Forme de clusters arbitraires
   ✓ Identifie outliers (label = -1)

Inconvénients :
   ✗ Sensible aux paramètres eps et min_samples
   ✗ Difficile si densité très variable
""")

print("\n📊 ÉTAPE 1 : TROUVER eps OPTIMAL (k-distance plot)")

# Calculer distance au 4ème voisin le plus proche
from sklearn.neighbors import NearestNeighbors

k = 4  # min_samples recommandé
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)

# Distances au k-ème voisin (triées)
distances = np.sort(distances[:, k-1], axis=0)

plt.figure(figsize=(10, 5))
plt.plot(distances, linewidth=2)
plt.xlabel('Points (triés par distance)', fontsize=11)
plt.ylabel(f'Distance au {k}-ème voisin le plus proche', fontsize=11)
plt.title('k-Distance Plot : Trouver eps optimal pour DBSCAN', fontsize=13, fontweight='bold')
plt.axhline(y=0.8, color='red', linestyle='--', label='eps suggéré = 0.8')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('E:\\Nicolas\\MIAGE\\M2\\BigData\\FORMATION_ML\\TUTORIELS\\06_kdistance_plot.png', dpi=100, bbox_inches='tight')
print("\n✓ Graphique sauvegardé : 06_kdistance_plot.png")
plt.close()

print("""
💡 COMMENT LIRE LE k-DISTANCE PLOT ?

   1. Chercher le "COUDE" où distance augmente brusquement
   2. Ce point sépare :
      - Points dans clusters denses (avant le coude)
      - Outliers (après le coude, distances élevées)
   3. eps ≈ distance au coude

   Ici : coude vers 0.8 → eps = 0.8
""")

# ÉTAPE 2 : ENTRAÎNER DBSCAN
print("\n📊 ÉTAPE 2 : ENTRAÎNEMENT DBSCAN")

dbscan = DBSCAN(eps=0.8, min_samples=4)
clusters_dbscan = dbscan.fit_predict(X_scaled)

n_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
n_noise = list(clusters_dbscan).count(-1)

print(f"✓ DBSCAN entraîné avec eps=0.8, min_samples=4")
print(f"  Nombre de clusters détectés : {n_clusters_dbscan}")
print(f"  Nombre d'outliers (noise) : {n_noise} ({n_noise/len(clusters_dbscan)*100:.1f}%)")

if n_clusters_dbscan > 1:
    # Silhouette sans les outliers
    mask_not_noise = clusters_dbscan != -1
    if mask_not_noise.sum() > 0:
        silhouette_dbscan = silhouette_score(X_scaled[mask_not_noise], clusters_dbscan[mask_not_noise])
        print(f"  Silhouette score (sans outliers) : {silhouette_dbscan:.3f}")

print("\n📊 DISTRIBUTION DES CLUSTERS DBSCAN :")
unique, counts = np.unique(clusters_dbscan, return_counts=True)
for cluster_id, count in zip(unique, counts):
    if cluster_id == -1:
        print(f"  Noise (outliers) : {count} points")
    else:
        print(f"  Cluster {cluster_id} : {count} points")

# Visualisation DBSCAN
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_dbscan,
                      cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
plt.title(f'DBSCAN Clustering - {n_clusters_dbscan} clusters, {n_noise} outliers',
          fontsize=13, fontweight='bold')
plt.colorbar(scatter, label='Cluster (-1 = noise)')
plt.grid(True, alpha=0.3)

# Comparaison K-Means vs DBSCAN
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_kmeans,
            cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5, label='K-Means')
plt.scatter(X_pca[clusters_dbscan == -1, 0], X_pca[clusters_dbscan == -1, 1],
            c='red', marker='x', s=100, linewidth=2, label='Outliers DBSCAN')
plt.xlabel(f'PC1', fontsize=11)
plt.ylabel(f'PC2', fontsize=11)
plt.title('K-Means avec Outliers DBSCAN superposés', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('E:\\Nicolas\\MIAGE\\M2\\BigData\\FORMATION_ML\\TUTORIELS\\06_dbscan_visualization.png', dpi=100, bbox_inches='tight')
print("\n✓ Graphique sauvegardé : 06_dbscan_visualization.png")
plt.close()

print("""
═══════════════════════════════════════════════════════════════════════════════
📊 OBSERVATION #4 : DBSCAN VS K-MEANS
═══════════════════════════════════════════════════════════════════════════════

CE QU'IL FAUT OBSERVER :

1. NOMBRE DE CLUSTERS

   K-Means : k fixé à 5 (imposé)
   DBSCAN  : Détecté automatiquement (peut différer)

   → Si DBSCAN trouve beaucoup plus ou moins de clusters :
     - Peut indiquer que k=5 n'est pas naturel dans les données
     - Ou que eps/min_samples mal choisis

2. OUTLIERS (DBSCAN uniquement)

   → Points marqués en ROUGE (label = -1)
   → Représentent combien % du dataset ?

   ✓ Si < 5% outliers :
     → Probablement vrais outliers à investiguer
     → Clients atypiques, erreurs de données ?

   ✗ Si > 20% outliers :
     → eps trop petit (augmenter eps)
     → Ou min_samples trop élevé

3. FORME DES CLUSTERS

   K-Means : Clusters sphériques/convexes
   DBSCAN  : Forme arbitraire (peut suivre densité)

   → Si données ont forme complexe (spirales, anneaux) :
     DBSCAN > K-Means

💡 CONCLUSION :

   ✓ Utiliser DBSCAN si :
     - Outliers sont informatifs (fraude, anomalies)
     - Nombre de clusters inconnu
     - Forme de clusters complexe

   ✓ Utiliser K-Means si :
     - Besoin de k segments précis (métier)
     - Toutes les données doivent être assignées
     - Rapidité cruciale (gros volumes)

⚠️ POINTS D'ATTENTION :

   - Outliers DBSCAN peuvent être des micro-segments ignorés par K-Means
   - Investiguer les outliers : erreurs données ? Clients VIP uniques ?
   - Combiner les deux : K-Means pour segmentation, DBSCAN pour outliers
═══════════════════════════════════════════════════════════════════════════════
""")

print("\n" + "="*80)
print("PARTIE 5 : CLUSTERING HIÉRARCHIQUE")
print("="*80)

print("""
🔧 CLUSTERING HIÉRARCHIQUE (Agglomerative)

Principe :
   1. Départ : chaque point = un cluster
   2. RÉPÉTER jusqu'à un seul cluster :
      - Fusionner les 2 clusters les plus proches
      - Recalculer distances

Méthodes de linkage (calcul distance inter-clusters) :

   - WARD : Minimise variance intra-cluster
     → Clusters équilibrés, sphériques (similaire K-Means)
     → LE PLUS UTILISÉ

   - AVERAGE : Distance moyenne entre tous points
     → Compromis

   - COMPLETE : Distance max entre points les plus éloignés
     → Clusters compacts

   - SINGLE : Distance min entre points les plus proches
     → Peut créer chaînes (effet "chaining")

Avantage :
   ✓ DENDROGRAMME : Visualise toute la hiérarchie
   ✓ Choisir k APRÈS clustering (couper dendrogramme)

Inconvénient :
   ✗ Complexité O(n²) ou O(n³)
   ✗ Ne scale pas (max ~10k points)
""")

print("\n📊 ÉTAPE 1 : CONSTRUCTION DU DENDROGRAMME")

# Utiliser un échantillon pour le dendrogramme (trop lent sinon)
sample_size = 200
idx_sample = np.random.choice(len(X_scaled), size=sample_size, replace=False)
X_sample = X_scaled[idx_sample]

# Calculer linkage matrix
linkage_matrix = linkage(X_sample, method='ward')

plt.figure(figsize=(14, 6))
dendrogram(linkage_matrix,
           truncate_mode='lastp',  # Montrer seulement derniers p clusters
           p=20,
           leaf_font_size=10,
           show_leaf_counts=True)
plt.xlabel('Cluster Index', fontsize=11)
plt.ylabel('Distance (Ward)', fontsize=11)
plt.title(f'Dendrogramme Hiérarchique (échantillon de {sample_size} points)',
          fontsize=13, fontweight='bold')
plt.axhline(y=15, color='red', linestyle='--', label='Coupure suggérée (5 clusters)')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('E:\\Nicolas\\MIAGE\\M2\\BigData\\FORMATION_ML\\TUTORIELS\\06_dendrogram.png', dpi=100, bbox_inches='tight')
print("\n✓ Graphique sauvegardé : 06_dendrogram.png")
plt.close()

print("""
💡 COMMENT LIRE UN DENDROGRAMME ?

   1. Axe Y : Distance de fusion
      → Plus haut = clusters plus distants

   2. Branches : Représentent fusions successives
      → Largeur = nombre de points dans le cluster

   3. COUPER LE DENDROGRAMME :
      → Tracer ligne horizontale → nb clusters
      → Ici : couper à y=15 → 5 clusters

   4. Chercher le plus GRAND SAUT VERTICAL
      → Indique séparation naturelle
      → Couper juste avant ce saut
""")

# ÉTAPE 2 : ENTRAÎNER CLUSTERING HIÉRARCHIQUE
print("\n📊 ÉTAPE 2 : CLUSTERING HIÉRARCHIQUE (n_clusters=5)")

hierarchical = AgglomerativeClustering(n_clusters=5, linkage='ward')
clusters_hierarchical = hierarchical.fit_predict(X_scaled)

print(f"✓ Clustering hiérarchique entraîné")
print(f"  Silhouette score : {silhouette_score(X_scaled, clusters_hierarchical):.3f}")
print(f"  Davies-Bouldin Index : {davies_bouldin_score(X_scaled, clusters_hierarchical):.3f}")

print("\n📊 DISTRIBUTION DES CLUSTERS HIÉRARCHIQUES :")
unique_hier, counts_hier = np.unique(clusters_hierarchical, return_counts=True)
for cluster_id, count in zip(unique_hier, counts_hier):
    print(f"  Cluster {cluster_id} : {count} points")

# Visualisation
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_kmeans,
                      cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.xlabel('PC1', fontsize=10)
plt.ylabel('PC2', fontsize=10)
plt.title('K-Means (k=5)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_dbscan,
                      cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.xlabel('PC1', fontsize=10)
plt.ylabel('PC2', fontsize=10)
plt.title(f'DBSCAN ({n_clusters_dbscan} clusters + outliers)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_hierarchical,
                      cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.xlabel('PC1', fontsize=10)
plt.ylabel('PC2', fontsize=10)
plt.title('Hiérarchique (n_clusters=5)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('E:\\Nicolas\\MIAGE\\M2\\BigData\\FORMATION_ML\\TUTORIELS\\06_comparison_all.png', dpi=100, bbox_inches='tight')
print("\n✓ Graphique sauvegardé : 06_comparison_all.png")
plt.close()

print("""
═══════════════════════════════════════════════════════════════════════════════
📊 OBSERVATION #5 : COMPARAISON DES 3 MÉTHODES
═══════════════════════════════════════════════════════════════════════════════

CE QU'IL FAUT OBSERVER :

1. COHÉRENCE ENTRE MÉTHODES

   → Les 3 méthodes identifient des clusters similaires ?
     ✓ OUI : Clusters robustes, bien définis
     ✗ NON : Structure ambiguë, clusters artificiels

   → Comparer visuellement les frontières

2. DIFFÉRENCES SPÉCIFIQUES

   K-Means vs Hiérarchique (Ward) :
     → Souvent très similaires (les deux minimisent variance)
     → Différences aux frontières uniquement

   DBSCAN vs autres :
     → Outliers = points où K-Means/Hiérarchique sont incertains
     → DBSCAN peut détecter moins de clusters (plus conservateur)

3. MÉTRIQUES DE QUALITÉ

   Silhouette Score :
     K-Means        : ~0.XX
     DBSCAN         : ~0.XX (sans outliers)
     Hiérarchique   : ~0.XX

   → Quelle méthode a le meilleur score ?
   → Différence > 0.1 : significative
   → Différence < 0.05 : équivalentes

💡 CONCLUSION : QUELLE MÉTHODE CHOISIR ?

   ✅ ACCORD FORT (3 méthodes similaires) :
      → Utiliser K-Means (rapidité + interprétabilité)
      → Ou Hiérarchique si besoin dendrogramme

   ✅ DBSCAN TRÈS DIFFÉRENT :
      → Investiguer les outliers DBSCAN
      → Peut révéler structure ignorée par K-Means
      → Considérer méthode hybride

   ✅ DÉSACCORD ENTRE TOUTES :
      → Structure de clustering faible dans les données
      → Revoir features (feature engineering ?)
      → Ou données pas naturellement clusterisables
      → Considérer segmentation basée sur règles métier

⚠️ CHECKLIST FINALE :

   [ ] Silhouette > 0.4 pour méthode choisie
   [ ] Clusters équilibrés en taille (sauf si intentionnel)
   [ ] Profils de clusters interprétables métier
   [ ] Validation avec stakeholders (Marketing, etc.)
   [ ] Outliers investigués (erreurs données ? VIP ?)
═══════════════════════════════════════════════════════════════════════════════
""")

print("\n" + "="*80)
print("PARTIE 6 : ANALYSE APPROFONDIE DES SEGMENTS")
print("="*80)

print("\n📊 PROFILS DÉTAILLÉS DES SEGMENTS (K-Means)")

# Ajouter clusters aux données originales (non normalisées)
df['Cluster'] = clusters_kmeans

# Statistiques par cluster
print("\n1️⃣ MOYENNES PAR CLUSTER :")
print(df.groupby('Cluster').mean().round(2))

print("\n2️⃣ MÉDIANES PAR CLUSTER :")
print(df.groupby('Cluster').median().round(2))

print("\n3️⃣ TAILLES DES CLUSTERS :")
cluster_sizes = df['Cluster'].value_counts().sort_index()
print(cluster_sizes)
print(f"\nProportion de chaque cluster :")
print((cluster_sizes / len(df) * 100).round(1).astype(str) + '%')

# Visualisation boxplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
features = ['Age', 'Income', 'SpendingScore', 'Recency', 'Frequency']

for idx, feature in enumerate(features):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    df.boxplot(column=feature, by='Cluster', ax=ax)
    ax.set_title(f'{feature} par Cluster', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=10)
    ax.set_ylabel(feature, fontsize=10)
    ax.grid(True, alpha=0.3)

# Supprimer le subplot vide
fig.delaxes(axes[1, 2])

# Ajouter un texte récapitulatif
axes[1, 2] = fig.add_subplot(2, 3, 6)
axes[1, 2].axis('off')
recap_text = f"""
RÉCAPITULATIF
{'='*30}

Nombre de clusters : {len(cluster_sizes)}

Silhouette Score : {silhouette_score(X_scaled, clusters_kmeans):.3f}

Tailles des clusters :
{chr(10).join([f'  Cluster {i}: {size} ({size/len(df)*100:.1f}%)' for i, size in cluster_sizes.items()])}

Prochaines étapes :
1. Nommer les clusters (profils)
2. Valider avec métier
3. Définir stratégies par segment
4. Suivre évolution dans le temps
"""
axes[1, 2].text(0.1, 0.5, recap_text, fontsize=10, family='monospace', verticalalignment='center')

plt.suptitle('Distribution des Features par Cluster', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('E:\\Nicolas\\MIAGE\\M2\\BigData\\FORMATION_ML\\TUTORIELS\\06_boxplots_clusters.png', dpi=100, bbox_inches='tight')
print("\n✓ Graphique sauvegardé : 06_boxplots_clusters.png")
plt.close()

print("""
═══════════════════════════════════════════════════════════════════════════════
📊 OBSERVATION #6 : BOXPLOTS ET PROFILS DÉTAILLÉS
═══════════════════════════════════════════════════════════════════════════════

CE QU'IL FAUT OBSERVER DANS LES BOXPLOTS :

1. SÉPARATION DES CLUSTERS

   → Boîtes qui se chevauchent PEU : Bonne séparation
   → Chevauchement fort : Clusters ambigus sur cette feature

   Exemple :
     - Age : Si boîtes bien séparées → Age discriminant
     - Income : Si chevauchement → Pas discriminant seul

2. DISPERSION INTRA-CLUSTER

   → Boîtes ÉTROITES : Cluster homogène (bonne cohésion)
   → Boîtes LARGES : Cluster hétérogène (revoir k ?)

3. OUTLIERS (points au-delà des moustaches)

   → Beaucoup d'outliers dans un cluster :
     - Cluster mal défini
     - Ou vraie diversité dans le segment

   → Investiguer ces points extrêmes

4. FEATURES DISCRIMINANTES

   → Identifier quelles features séparent le MIEUX les clusters
   → Celles avec séparation claire = clés pour nommage

   Exemple profil :
     Cluster 0 : Age BAS + SpendingScore HAUT
                 → "Jeunes Dépensiers"
     Cluster 2 : Recency HAUT + Frequency BAS
                 → "Clients Dormants"

💡 CONCLUSION POUR LA SEGMENTATION CLIENT :

   ✓ NOMMAGE DES SEGMENTS (exemples)

   Segment 0 : "Jeunes Actifs"
     - Age : 25-35 ans
     - Income : Moyen
     - Stratégie : Offres lifestyle, réseaux sociaux

   Segment 1 : "VIP Premium"
     - Income : Élevé
     - Frequency : Très élevée
     - Stratégie : Services exclusifs, early access

   Segment 2 : "En Sommeil"
     - Recency : > 60 jours
     - Frequency : Faible
     - Stratégie : Réactivation, promotions agressives

   Segment 3 : "Économes"
     - SpendingScore : Faible
     - Stratégie : Coupons, soldes, rapport qualité/prix

   Segment 4 : "Matures Stables"
     - Age : > 50 ans
     - Behaviour : Stable, prévisible
     - Stratégie : Fidélisation, service client premium

⚠️ ACTIONS POST-CLUSTERING :

   [ ] Présenter profils à équipes Marketing/Sales
   [ ] Valider cohérence avec connaissance métier
   [ ] Calculer métriques business par segment :
       - LTV (Lifetime Value)
       - Taux de conversion
       - Panier moyen
   [ ] Définir KPI de suivi par segment
   [ ] Implémenter scoring pour nouveaux clients
   [ ] Monitorer évolution des segments dans le temps
═══════════════════════════════════════════════════════════════════════════════
""")

print("\n" + "="*80)
print("PARTIE 7 : ÉVALUATION ET VALIDATION")
print("="*80)

print("""
📊 RÉCAPITULATIF DES MÉTRIQUES DE QUALITÉ

1. SILHOUETTE SCORE [-1, 1]

   Formule : s = (b - a) / max(a, b)
   où a = distance moyenne intra-cluster
       b = distance moyenne inter-cluster

   Interprétation :
     > 0.7  : Séparation forte, clusters excellents
     0.5-0.7: Séparation raisonnable, structure claire
     0.25-0.5: Séparation faible, clusters moyens
     < 0.25 : Pas de structure claire, reconsidérer k

2. DAVIES-BOULDIN INDEX [0, +∞]

   Mesure : Ratio similarité intra-cluster / inter-cluster

   Interprétation :
     Plus BAS = Meilleur
     < 1.0 : Excellente séparation
     1.0-2.0 : Séparation acceptable
     > 2.0 : Clusters se chevauchent beaucoup

3. INERTIE (K-Means uniquement)

   Somme des distances² aux centroïdes

   ⚠️ NE PAS COMPARER entre datasets différents !
      Seulement pour Elbow Method (même dataset, différents k)
""")

# Tableau récapitulatif
print("\n📊 COMPARAISON FINALE DES MÉTHODES :")
print("="*70)
print(f"{'Méthode':<20} {'Silhouette':>12} {'Davies-Bouldin':>15} {'Nb Clusters':>12}")
print("="*70)

sil_kmeans = silhouette_score(X_scaled, clusters_kmeans)
db_kmeans = davies_bouldin_score(X_scaled, clusters_kmeans)
print(f"{'K-Means':<20} {sil_kmeans:>12.3f} {db_kmeans:>15.3f} {5:>12}")

if n_clusters_dbscan > 1 and mask_not_noise.sum() > 0:
    sil_dbscan = silhouette_score(X_scaled[mask_not_noise], clusters_dbscan[mask_not_noise])
    db_dbscan = davies_bouldin_score(X_scaled[mask_not_noise], clusters_dbscan[mask_not_noise])
    print(f"{'DBSCAN':<20} {sil_dbscan:>12.3f} {db_dbscan:>15.3f} {n_clusters_dbscan:>12}")
else:
    print(f"{'DBSCAN':<20} {'N/A':>12} {'N/A':>15} {n_clusters_dbscan:>12}")

sil_hier = silhouette_score(X_scaled, clusters_hierarchical)
db_hier = davies_bouldin_score(X_scaled, clusters_hierarchical)
print(f"{'Hiérarchique':<20} {sil_hier:>12.3f} {db_hier:>15.3f} {5:>12}")
print("="*70)

print("""
═══════════════════════════════════════════════════════════════════════════════
📊 OBSERVATION #7 : VALIDATION ET DÉCISION FINALE
═══════════════════════════════════════════════════════════════════════════════

CE QU'IL FAUT OBSERVER DANS LES MÉTRIQUES :

1. CONVERGENCE DES MÉTHODES

   → Silhouette scores similaires entre méthodes ?
     ✓ OUI : Structure de clustering robuste
     ✗ NON : Une méthode capte mieux la structure

   → Davies-Bouldin cohérent ?
     ✓ Tous < 1.5 : Bonne séparation quelle que soit méthode
     ✗ Un score >> autres : Méthode inadaptée

2. CRITÈRES DE DÉCISION

   Méthode A : K-Means
     ✓ Silhouette élevé + DB faible
     ✓ Rapidité (production sur millions de clients)
     ✓ Nombre de clusters fixe acceptable (métier)
     → RECOMMANDÉ pour production

   Méthode B : DBSCAN
     ✓ Détecte outliers importants (fraude, VIP)
     ✓ Forme de clusters complexe
     → RECOMMANDÉ pour exploration + détection anomalies

   Méthode C : Hiérarchique
     ✓ Besoin de visualiser hiérarchie
     ✓ Petit dataset
     → RECOMMANDÉ pour analyse exploratoire

3. VALIDATION BUSINESS

   ⚠️ MÉTRIQUES ≠ SUCCÈS BUSINESS !

   Un clustering avec silhouette 0.45 mais segments
   métier cohérents > clustering silhouette 0.65 ininterprétables

   Questions clés :

   [ ] Les profils de clusters ont du SENS ?
   [ ] Actionnables pour Marketing/Sales ?
   [ ] Stables dans le temps (re-clustering mensuel) ?
   [ ] Améliorent KPI business (conversion, LTV) ?

💡 DÉCISION FINALE (pour ce cas d'usage) :

   MÉTHODE RETENUE : K-Means (k=5)

   Raisons :
   ✓ Silhouette satisfaisant (> 0.4)
   ✓ 5 segments cohérents métier
   ✓ Scalable pour scoring temps réel
   ✓ Facile à expliquer aux stakeholders

   USAGE COMPLÉMENTAIRE : DBSCAN

   Pour :
   ✓ Détection clients VIP uniques (outliers)
   ✓ Fraude / comportements anormaux

   PROCHAINES ÉTAPES :

   1. [ ] Nommer les 5 segments
   2. [ ] Calculer LTV par segment
   3. [ ] Définir stratégies marketing ciblées
   4. [ ] Implémenter scoring nouveaux clients
   5. [ ] Dashboard suivi segments (Power BI, Tableau)
   6. [ ] A/B testing stratégies par segment
   7. [ ] Re-clustering mensuel (évolution segments)
   8. [ ] Analyser transitions inter-segments

⚠️ MONITORING EN PRODUCTION :

   - Surveiller distribution clusters (dérive ?)
   - Recalculer centroïdes régulièrement
   - Alertes si % outliers DBSCAN change brutalement
   - Valider que nouvelles données similaires (distribution)
═══════════════════════════════════════════════════════════════════════════════
""")

print("\n" + "="*80)
print("RÉSUMÉ FINAL : QUAND UTILISER QUEL ALGORITHME ?")
print("="*80)

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    ARBRE DE DÉCISION CLUSTERING                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

Vous connaissez le NOMBRE DE CLUSTERS ?
│
├─ OUI ──────────────────────────────────────────────────────┐
│                                                              │
│  Les clusters sont de forme SPHÉRIQUE/CONVEXE ?            │
│  │                                                           │
│  ├─ OUI ──> K-MEANS                                        │
│  │          ✓ Rapide, scalable                             │
│  │          ✓ Interprétable (centroïdes)                   │
│  │          ✓ Production                                    │
│  │                                                           │
│  └─ NON ──> CLUSTERING HIÉRARCHIQUE (si n < 10k)           │
│             ✓ Forme arbitraire                              │
│             ✓ Dendrogramme (visualisation)                 │
│             ✗ Lent sur gros volumes                        │
│                                                              │
└─ NON ──────────────────────────────────────────────────────┐
                                                               │
   Présence d'OUTLIERS importants ?                          │
   │                                                           │
   ├─ OUI ──> DBSCAN                                         │
   │          ✓ Détecte outliers automatiquement             │
   │          ✓ Nombre de clusters automatique               │
   │          ✓ Forme arbitraire                             │
   │          ⚠️ Sensible aux paramètres eps/min_samples     │
   │                                                           │
   └─ NON ──> Essayer K-MEANS avec Elbow Method              │
              Ou HIÉRARCHIQUE + Dendrogramme                  │


╔═══════════════════════════════════════════════════════════════════════════╗
║                    CAS D'USAGE PAR ALGORITHME                             ║
╚═══════════════════════════════════════════════════════════════════════════╝

🎯 K-MEANS

   ✅ Segmentation client (marketing)
   ✅ Compression d'images (réduire palette couleurs)
   ✅ Organisation de documents (topics similaires)
   ✅ Analyse de séries temporelles (patterns)
   ✅ Recommandation (groupes de produits similaires)

   ⚠️ Limitations :
   - Clusters non-sphériques mal détectés
   - Sensible aux outliers (fausse centroïdes)
   - Résultats dépendent initialisation (utiliser n_init=10+)

🎯 DBSCAN

   ✅ Détection de fraude (outliers = fraudes potentielles)
   ✅ Analyse spatiale (densité géographique)
   ✅ Analyse de réseaux sociaux (communautés)
   ✅ Détection d'anomalies (maintenance prédictive)
   ✅ Traitement d'images (segmentation)

   ⚠️ Limitations :
   - Difficile si densité très variable
   - Paramètres eps/min_samples à tuner finement
   - Peut créer UN SEUL gros cluster si mal paramétré

🎯 HIÉRARCHIQUE

   ✅ Analyse phylogénétique (biologie)
   ✅ Taxonomie (classification hiérarchique)
   ✅ Exploration de données (comprendre structure)
   ✅ Petits datasets avec besoin visualisation

   ⚠️ Limitations :
   - NE SCALE PAS (max 10-20k points)
   - Décisions de fusion irréversibles
   - Plusieurs linkages possibles (ward, complete, average...)


╔═══════════════════════════════════════════════════════════════════════════╗
║                    CHECKLIST AVANT PRODUCTION                             ║
╚═══════════════════════════════════════════════════════════════════════════╝

📋 QUALITÉ DES DONNÉES

   [ ] Données normalisées (StandardScaler, MinMaxScaler)
   [ ] Valeurs manquantes traitées (imputation ou suppression)
   [ ] Outliers investigués (erreurs ? vrais outliers ?)
   [ ] Features corrélées identifiées (considérer PCA ?)
   [ ] Échelle des features cohérente

📋 QUALITÉ DES CLUSTERS

   [ ] Silhouette score > 0.4 (idéalement > 0.5)
   [ ] Davies-Bouldin index < 1.5
   [ ] Clusters équilibrés en taille (sauf si intentionnel)
   [ ] Profils de clusters INTERPRÉTABLES
   [ ] Validation avec équipes métier

📋 ROBUSTESSE

   [ ] Tester stabilité (re-run avec random_state différent)
   [ ] Cross-validation si possible (inertie, silhouette)
   [ ] Tester sur échantillon hold-out (généralisation)
   [ ] Comparer plusieurs algorithmes
   [ ] Documenter choix d'hyperparamètres

📋 PRODUCTION

   [ ] Pipeline de preprocessing sauvegardé (scaler, PCA)
   [ ] Modèle sauvegardé (joblib, pickle)
   [ ] Fonction de scoring nouveaux points implémentée
   [ ] Monitoring distribution clusters en production
   [ ] Plan de re-clustering régulier (mensuel, trimestriel)
   [ ] Dashboard visualisation segments
   [ ] Documentation pour utilisateurs métier


🎓 FIN DU TUTORIEL CLUSTERING !

   Vous savez maintenant :
   ✓ Choisir l'algorithme adapté au problème
   ✓ Trouver le nombre optimal de clusters
   ✓ Évaluer la qualité des clusters
   ✓ Interpréter et nommer les segments
   ✓ Valider avec métriques ET connaissance métier
   ✓ Déployer en production avec monitoring
""")

print("\n" + "="*80)
print("SAUVEGARDE DU MODÈLE")
print("="*80)

import joblib

# Sauvegarder le modèle et le scaler
joblib.dump(kmeans_final, 'E:\\Nicolas\\MIAGE\\M2\\BigData\\FORMATION_ML\\TUTORIELS\\kmeans_model.pkl')
joblib.dump(scaler, 'E:\\Nicolas\\MIAGE\\M2\\BigData\\FORMATION_ML\\TUTORIELS\\scaler_clustering.pkl')
joblib.dump(pca, 'E:\\Nicolas\\MIAGE\\M2\\BigData\\FORMATION_ML\\TUTORIELS\\pca_clustering.pkl')

print("\n✓ Modèle sauvegardé : kmeans_model.pkl")
print("✓ Scaler sauvegardé : scaler_clustering.pkl")
print("✓ PCA sauvegardé : pca_clustering.pkl")

print("""
📝 UTILISATION EN PRODUCTION :

import joblib
import numpy as np

# Charger modèle et transformers
scaler = joblib.load('scaler_clustering.pkl')
kmeans = joblib.load('kmeans_model.pkl')
pca = joblib.load('pca_clustering.pkl')

# Nouveau client
nouveau_client = np.array([[30, 50, 70, 20, 15]])  # Age, Income, SpendingScore, Recency, Frequency

# Preprocessing
client_scaled = scaler.transform(nouveau_client)

# Prédiction
segment = kmeans.predict(client_scaled)[0]
print(f"Client assigné au segment : {segment}")

# Distance aux centroïdes (confiance)
distances = kmeans.transform(client_scaled)[0]
print(f"Distance au centroïde assigné : {distances[segment]:.2f}")
print(f"Distance au centroïde le plus proche suivant : {np.sort(distances)[1]:.2f}")

# Si distance très élevée → client atypique, investiguer
if distances[segment] > 2.0:
    print("⚠️ Client atypique pour son segment !")
""")

print("\n" + "="*80)
print("🎉 TUTORIEL TERMINÉ AVEC SUCCÈS !")
print("="*80)
