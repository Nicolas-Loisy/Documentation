# Crypto & Blockchain

## Définitions

- **Stochastic** : Processus intégrant du hasard ou de la probabilité, souvent utilisé pour modéliser des phénomènes non déterministes.

## Cryptographie

### Types de chiffrement
- **Substitution** : Méthode où chaque lettre ou groupe de lettres est remplacé par un autre selon une règle définie.  
  - **César** : Décalage fixe des lettres de l’alphabet.  
  - **Enigma** : Chiffrement polyalphabétique utilisant des rotors pour changer dynamiquement les substitutions.  
- **Symétrique** : Utilise une seule clé pour chiffrer et déchiffrer les données.  
- **Asymétrique** : Utilise une paire de clés (publique et privée) où la clé publique chiffre et la clé privée déchiffre (ou inversement).  

---

### Cryptographie quantique
- **Quantique** : Exploite les propriétés de la mécanique quantique pour sécuriser les communications.  
  - *Protocole BB84* : Méthode d’échange de clés basée sur les états quantiques, garantissant la détection d’une interception.  
- **Post-quantique** : Algorithmes conçus pour résister aux attaques des ordinateurs quantiques.  
  - *NTRU* : Basé sur les polynômes et les réseaux.  
  - *McEliece* : Basé sur les codes correcteurs d’erreurs.  

---

### Cryptographie symétrique
- **AES (Advanced Encryption Standard)** : Algorithme de chiffrement par blocs utilisé pour sécuriser les données.  
  - Bloc : 128 bits.  
  - Tours : Nombre de cycles de transformation (10, 12 ou 14 selon la taille de la clé).  
- **Confusion et diffusion** (concepts fondamentaux de la cryptographie) :  
  - *Confusion* : Rend la relation entre la clé et le texte chiffré difficile à comprendre.  
  - *Diffusion* : Répartit l’information sur tout le texte chiffré pour rendre les motifs indiscernables.  

---

### Chiffrement symétrique
- **Homomorphique** : Permet d'effectuer des calculs directement sur des données chiffrées sans les déchiffrer, produisant un résultat qui, une fois déchiffré, correspond au résultat des mêmes calculs sur les données en clair.
    - Permet de poser une question sur les données chiffrées mais sans accès au contenu.

---

### Cryptographie asymétrique

- **RSA** : Utilisé pour les signatures numériques, garantissant l’authenticité et l’intégrité d’un message. Il repose sur la factorisation de grands nombres premiers. La sécurité du RSA dépend de la génération de deux nombres premiers très grands et bien éloignés l’un de l’autre, avec des tests de primalité pour assurer leur qualité.

    - **Clé RSA** : La taille minimale recommandée pour une clé RSA est de 1024 bits, mais il est fortement conseillé d'utiliser des tailles de 2048 bits ou 4096 bits pour assurer une sécurité.


## À savoir

Le fonctionnement d'Enigma et d'AES :  
- **Difficulté à casser** :  
  - **Enigma** : Utilise des rotors et un réflecteur pour créer un chiffrement polyalphabétique complexe, rendant la clé de chiffrement difficile à déduire sans connaître la configuration exacte des rotors.  
  - **AES** : Basé sur des transformations de blocs de données avec plusieurs tours, la sécurité repose sur la taille de la clé et l'absence de vulnérabilités connues dans l'algorithme.  
- **Cas d'usage** :  
  - **Enigma** : Principalement utilisé pendant la Seconde Guerre mondiale pour la communication militaire secrète.  
  - **AES** : Utilisé dans des applications modernes telles que le chiffrement de données sensibles (transactions bancaires, communications sécurisées, stockage de données).  
- **Tailles de clé recommandées** :  
  - **Enigma** : La sécurité dépend de la configuration des rotors, généralement 3 rotors pour les configurations de base.  
  - **AES** : Les tailles de clé recommandées sont de 128 bits (minimum), 192 bits et 256 bits pour une sécurité optimale contre les attaques par force brute.



## Projet : Émulateur Enigma

### Fonctionnalités
- Chiffrement et déchiffrement :  
  - Texte à chiffrer.  
  - Clé secrète.  
  - Configuration des rotors (position initiale et ordre).  
  - Support des couples clavier/affichage.  
- Configuration standard d’Enigma :  
  - 3 rotors avec réflecteur intégré.  

### Options supplémentaires
- Interface utilisateur (facultative).  

### Contraintes
- Travail individuel.  
- Langage au choix.  
