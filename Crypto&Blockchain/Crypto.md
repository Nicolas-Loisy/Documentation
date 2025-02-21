# Cryptographie

## Définitions

- **Stochastic** : Processus intégrant du hasard ou de la probabilité, souvent utilisé pour modéliser des phénomènes non déterministes.
- **Confidentialité** : 
- **Intégrité de la donnée** :
- **Authenticité** : 
- **Moyens d'implémentation des méthodes** : 

## Cryptographie

### Types de chiffrement
- **Substitution** : Méthode où chaque lettre ou groupe de lettres est remplacé par un autre selon une règle définie.  
  - **César** : Décalage fixe des lettres de l’alphabet.  
  - **Enigma** : Chiffrement polyalphabétique utilisant des rotors pour changer dynamiquement les substitutions.  
- **Symétrique** : Utilise une seule clé pour chiffrer et déchiffrer les données.  
- **Asymétrique** : Utilise une paire de clés (publique et privée) où la clé publique chiffre et la clé privée déchiffre (ou inversement).  

---

### Cryptographie quantique
- **Quantique** : Exploite les propriétés de la mécanique quantique pour sécuriser les communications ou casser plus rapidement les chiffrements grace à sa puissance de calcul.  
  - *Protocole BB84* : Méthode d’échange de clés basée sur les états quantiques, garantissant la détection d’une interception.  
- **Post-quantique** : Algorithmes conçus pour résister aux attaques des ordinateurs quantiques.  
  - *NTRU* : Basé sur les polynômes et les réseaux.  
  - *McEliece* : Basé sur les codes correcteurs d’erreurs.  

---

### Cryptographie symétrique
- **AES (Advanced Encryption Standard)** : Algorithme de chiffrement par blocs utilisé pour sécuriser les données.  
  - **Objectif d'AES** : Algorithme simple, rapide (temps quasi réel) que l'on peut réitérer un grand nombre de fois et qui fourni une très grande complexité calculatoire.
  - Bloc : 128 bits.  
  - Tours : Nombre de cycles de transformation (10, 12 ou 14 selon la taille de la clé).  
- **Confusion et diffusion** (concepts fondamentaux de la cryptographie) :  
  - *Confusion* (Table de substitution : S-box) : Rend la relation entre la clé et le texte chiffré difficile à comprendre.  
  - *Diffusion* (Table de permutation : P-box) : Répartit l’information sur tout le texte chiffré pour rendre les motifs indiscernables.

---

### Chiffrement symétrique
- **Homomorphique** : Permet d'effectuer des calculs directement sur des données chiffrées sans les déchiffrer, produisant un résultat qui, une fois déchiffré, correspond au résultat des mêmes calculs sur les données en clair.
    - Permet de poser une question sur les données chiffrées mais sans accès au contenu.
    - Avantage : Pas obligé de tout déchiffrer.

---

### Cryptographie asymétrique

- **RSA** : Utilisé pour les signatures numériques, garantissant l’authenticité et l’intégrité d’un message. Il repose sur la factorisation de grands nombres premiers. La sécurité du RSA dépend de la génération de deux nombres premiers très grands et bien éloignés l’un de l’autre, avec des tests de primalité pour assurer leur qualité.

    - **Clé RSA** : La taille minimale recommandée pour une clé RSA est de 1024 bits, mais il est fortement conseillé d'utiliser des tailles de 2048 bits ou 4096 bits pour assurer une sécurité.

---

### Cryptographie hybride

- **Modèle OSI**
    - Couche physique
    - Couche de liaison de données
    - Couche réseau
    - Couche de transport
    - Couche de session
    - Couche de présentation
    - Couche application

- **TLS** : 
    - Utilisé dans : messagerie, voix ip, https





## À savoir

Le fonctionnement d'Enigma et d'AES :  
- **Difficulté à casser** :  
  - **Enigma** : Utilise des rotors et un réflecteur pour créer un chiffrement polyalphabétique complexe, rendant la clé de chiffrement difficile à déduire sans connaître la configuration exacte des rotors.  
  - **AES** : Basé sur des transformations de blocs de données avec plusieurs tours, la sécurité repose sur la taille de la clé et l'absence de vulnérabilités connues dans l'algorithme.  
- **Cas d'usage** :  
  - **Enigma** : Principalement utilisé pendant la Seconde Guerre mondiale pour la communication militaire secrète.  
  - **AES** : Utilisé dans des applications modernes telles que le chiffrement de données sensibles (VPN, WI-FI, TLS, SSH, transactions bancaires, communications sécurisées, stockage de données).  
- **Tailles de clé recommandées** :  
  - **Enigma** : La sécurité dépend de la configuration des rotors, généralement 3 rotors pour les configurations de base.  
  - **AES** : Les tailles de clé recommandées sont de 128 bits (minimum), 192 bits et 256 bits pour une sécurité optimale contre les attaques par force brute.


**Différence entre codage et cryptographie** : 
- **Codage** = Transformation pour compatibilité (pas de sécurité), pas un but de sécurité.
- **Cryptographie** = Transformation pour sécurité (confidentialité et intégrité).
- **Cryptanalyse** = Science de casser le chiffrement (illégitime)
- **Cryptologie** = Ensemble de la **Cryptographie** et de la **Cryptanalyse** 
- **Stéganographie** = Science de la dissimulation de l'information 

**Différence entre permutation et substitution** : 
La substitution consiste à remplacer un élément (comme une lettre ou un chiffre) par un autre selon une règle fixe, tandis que la permutation implique un échange de positions entre les éléments sans modification de leur valeur.

**Asymétrique** : Utilise une paire de clés (publique et privée) où la clé publique est dérivée de grands nombres premiers, avec des tests de primalité pour assurer leur sécurité. La clé publique chiffre les données, et la clé privée les déchiffre.

**Handshake** : Premier échange entre deux parties pour établir une connexion sécurisée, incluant la transmission des paramètres nécessaires (clés, algorithmes, etc.) pour initier la communication.

**Partie commune entre TLS et SSH** : Les deux protocoles utilisent de la cryptographie asymétrique pour l'échange de clés, le processus de vérification de l'identité des parties et l'établissement d'une connexion sécurisée via un échange initial (handshake). Les deux sont sur la couche application.


**Différence entre Connexion à distance et Tunneling** :
- **Connexion à distance** = Accès direct à un système distant, prise de contrôle du PC.
- **Tunneling** = Création d'un tunnel sécurisé pour le transfert de données.

Hashage : 

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

