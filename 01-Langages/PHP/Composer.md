# Composer

### Commandes courantes

- **Installer les dépendances** :
  Installe les paquets à partir de `composer.json` et crée/actualise `composer.lock`.
  ```bash
  composer install
  ```

- **Mettre à jour les dépendances** :
  Met à jour les dépendances à leurs dernières versions compatibles.
  ```bash
  composer update
  ```

- **Mettre à jour Composer lui-même** :
  Met à jour Composer vers la dernière version stable.
  ```bash
  composer self-update
  ```

- **Créer un projet Symfony** :
  Crée un nouveau projet Symfony à partir du squelette de base.
  ```bash
  composer create-project symfony/skeleton my_project
  ```

### Gestion des composants

- **Ajouter un composant** :
  Installe un composant ou une bibliothèque.
  ```bash
  composer require <composant>
  ```

- **Supprimer un composant** :
  Supprime un composant du projet.
  ```bash
  composer remove <composant>
  ```

- **Ajouter un composant en mode dev** :
  Installe un composant pour l'environnement de développement uniquement.
  ```bash
  composer require <composant> --dev
  ```

### Générateur de données

- **Générer des données ORM** :
  Installe `orm-fixtures` pour générer des données de test pour la base de données.
  ```bash
  composer require orm-fixtures --dev
  ```

- **Zenstruck Foundry** :
  Installe `zenstruck/foundry` pour créer des générateurs de données.
  ```bash
  composer require zenstruck/foundry --dev
  ```
