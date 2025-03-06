# Récapitulatif

### **AES (Advanced Encryption Standard)**  
- **Définition** : Algorithme de chiffrement symétrique, rapide et sécurisé, utilisé pour protéger des données.  
- **Objectif** : Garantir la confidentialité des données en les chiffrant avec une clé unique.  
- **Logique de fonctionnement** :  
  1. **SubWord** : Substitution de chaque octet via la S-Box.  
  2. **Rcon** : Valeur de ronde XOR avec une partie de la clé pour introduire de la variation.  
  3. **Expansion de clé** : Chaque tour génère une nouvelle clé à partir de la précédente.  

- **Utilisations** :  
  - Chiffrement des fichiers, bases de données, disques (BitLocker, VeraCrypt).  
  - Sécurisation des communications (VPN, Wi-Fi WPA2/WPA3).  
  - Protection des transactions bancaires et paiements en ligne.  

---

### **Modèle OSI (Open Systems Interconnection)**  
- Modèle en 7 couches décrivant le fonctionnement des réseaux informatiques :  
  1. **Physique** : Transmission des bits sur un support (câble, fibre, radio).  
  2. **Liaison de données** : Détection et correction des erreurs (MAC, Ethernet).  
  3. **Réseau** : Routage des paquets (IP).  
  4. **Transport** : Communication de bout en bout (TCP, UDP).  
  5. **Session** : Gestion des connexions entre applications.  
  6. **Présentation** : Format des données (chiffrement, encodage).  
  7. **Application** : Interface utilisateur (HTTP, FTP, SSH).  

---

### **SSH (Secure Shell)**  
- **Buts** :  
  - Exploiter des services réseau de manière sécurisée sur un réseau non sécurisé.  
  - Connexion à distance sécurisée à un serveur.  
  - Tunneling sécurisé de protocoles non sécurisés.  
  - Transfert sécurisé de fichiers (SFTP).  
  - Exécution de commandes à distance via terminal.  

- **Différence entre connexion à distance et tunneling** :  
  - **Connexion à distance** : Accès direct à un serveur distant via SSH.  
  - **Tunneling** : Encapsulation d’un flux réseau (HTTP, RDP...) dans SSH pour sécuriser la communication.  

---

### **Registres et architectures de stockage**  
- **Registre** : Base de données structurée stockant des informations de manière ordonnée.  
- **Systèmes de stockage** :  
  - **Centralisé** : Une seule entité contrôle les données (ex : base de données classique).  
  - **Décentralisé** : Plusieurs entités gèrent les données, mais avec une autorité centrale (ex : DNS, certaines blockchains privées).  
  - **Distribué** : Pas d’autorité centrale, chaque participant détient une copie des données (ex : Bitcoin, IPFS).  

---

### **Arbre de Merkle**  
- **Définition** : Structure de données permettant de vérifier l’intégrité d’un ensemble de données en utilisant des fonctions de hachage.  
- **Utilisation** :  
  - Optimisation et sécurisation des transactions blockchain.  
  - Vérification rapide de l’intégrité des fichiers dans un réseau distribué.  

---

### **Blockchain et Hard Fork**  
- **Blockchain** : Registre distribué et immuable, stockant des transactions sécurisées et vérifiées par un consensus.  
- **Hard Fork** : Scission d’une blockchain en deux versions incompatibles, souvent à la suite d’un changement de protocole (ex : Bitcoin vs Bitcoin Cash).  

---

### **Contrat Intelligent (Smart Contract)**  
- **Définition** : Programme exécuté automatiquement sur une blockchain dès que certaines conditions sont remplies.  
- **Caractéristiques** :  
  - **Immuable** : Une fois déployé, il ne peut pas être modifié.  
  - **Distribué** : Exécuté sur plusieurs nœuds du réseau blockchain.  

- **Applications** :  
  - Finance décentralisée (DeFi).  
  - Automatisation des paiements et des accords (assurance, supply chain).  
  - Gestion des droits d’auteur et des royalties.  

---

### **NFT (Non-Fungible Token)**  
- **Définition** : Jeton unique prouvant la propriété d’un actif numérique.  
- **Différence avec un asset numérique** :  
  - **NFT** = Droit de propriété sur un asset numérique.  
  - **L’asset numérique** (image, musique, vidéo...) peut être copié, mais seul le NFT atteste de la propriété.  

- **Utilisations** :  
  - Art numérique et collection.  
  - Jeux vidéo (skins, objets uniques).  
  - Billetterie et certificats numériques.  
