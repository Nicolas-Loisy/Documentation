# Yarn

Yarn est un gestionnaire de paquets JavaScript qui permet de gérer les dépendances de votre projet.

---

## 1. Installation de Yarn

### Installation globale de Yarn
Pour installer Yarn globalement, utilisez la commande suivante :
```bash
npm install --global yarn
```

### Installer les dépendances avec Yarn
Une fois Yarn installé, vous pouvez installer les dépendances de votre projet en exécutant :
```bash
yarn install
```

Cela va installer toutes les dépendances définies dans le fichier `package.json` du projet.

---

## 2. Lancer le serveur de développement

### Démarrer le serveur de développement
Pour démarrer le serveur de développement, utilisez la commande suivante :
```bash
yarn encore dev-server
```

Cela lancera le serveur Webpack, généralement utilisé pour la gestion des assets dans un projet Symfony avec Yarn et Webpack Encore.

### Démarrer en mode watch
Pour démarrer le serveur avec une surveillance des fichiers (en mode "watch"), utilisez :
```bash
yarn watch
```

Cela permet de recompiler automatiquement les fichiers lorsque des changements sont détectés.
