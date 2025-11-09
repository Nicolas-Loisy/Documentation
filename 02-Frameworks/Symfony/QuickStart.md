# Mise en Place d'un Projet Symfony avec Docker

## 1. Création du Projet

- **Créer un dossier pour le projet** :
  - Créez un nouveau dossier dans `Ubuntu/opt/` pour votre projet, par exemple :
    `mkdir /opt/NomDuNewProjet`

- **Récupérer le dépôt Git** :
  - Clonez le repository avec la commande :
    ```bash
    git clone URL_DuRepoDuNewProjet
    ```

- **Créer les dossiers nécessaires et fichier `docker-compose.yml`** :
  - Dans le dossier du projet, créez les répertoires `etc/docker`.
  - Créez un fichier `docker-compose.yml` dans `etc/docker` pour la configuration Docker et des containers (voir Annexe 1 pour la configuration).

- **Modifier `docker-compose.yml`** :
  - Remplacez `starterkit_symfony` par le nom de votre projet dans le fichier `etc/docker/docker-compose.yml`.

---

## 2. Lancer Docker

- **Démarrer Docker** :
  - Accédez au dossier `etc/docker` et lancez la commande :
    ```bash
    docker-compose up -d
    ```

- **Accéder au container Docker** :
  - Une fois Docker démarré, entrez dans le container avec la commande suivante :
    ```bash
    docker exec -ti starterkit_symfony /bin/bash
    ```

- **Vérification de la structure du container** :
  - Dans le container, exécutez `ls` pour vérifier les fichiers suivants :
    ```bash
    app  index.html  info.php  supervisord.log  supervisord.pid
    ```

---

## 3. Initialiser le Projet Symfony

- **Vérifier si Composer est installé** :
  - Exécutez `composer --version` pour vérifier si Composer est installé.

- **Créer le squelette du projet Symfony** :
  - Créez un nouveau projet Symfony en utilisant la commande suivante dans le container :
    ```bash
    composer create-project symfony/skeleton:"6.2.*" app
    ```

- **Accéder au dossier du projet** :
  - Allez dans le dossier `app` et vérifiez la structure avec `ls` :
    ```bash
    bin  composer.json  composer.lock  config  public  src  symfony.lock  var  vendor
    ```

---

## 4. Ajouter des Composants Symfony

- **Installer des composants Symfony** :
  - Exécutez les commandes suivantes pour ajouter des composants au projet :
    ```bash
    composer require webapp
    composer require twig
    composer require logger
    ```
  - Lors de l'installation du package `webapp`, il vous sera demandé si vous voulez inclure une configuration Docker. Répondez `y`.

- **Vérifier la structure du projet** :
  - Après avoir ajouté les composants, exécutez `ls` pour voir la nouvelle structure :
    ```bash
    bin  composer.lock  docker-compose.override.yml  migrations  public  symfony.lock  tests  var  composer.json  config  docker-compose.yml  phpunit.xml.dist  src  templates  translations  vendor
    ```

---

## 5. Gérer les Permissions et Configurations

- **Accéder au dossier `src/` et vérifier sa structure** :
  - Exécutez les commandes suivantes :
    ```bash
    cd src
    ls
    ```
  - La structure devrait afficher : `Controller  Entity  Kernel.php  Repository`.

- **Appliquer les permissions sur le projet** :
  - Revenez au répertoire parent et appliquez les permissions :
    ```bash
    cd ..
    chmod 777 *
    ```

- **Arrêter le container Docker** :
  - Exécutez la commande suivante pour arrêter les containers :
    ```bash
    docker compose down
    ```

---

## 6. Configurations Docker (Fichiers à Modifier)

- **Modifier `docker-compose.yml`** :
  - Dans `etc/docker/docker-compose.yml`, décommentez les lignes nécessaires (voir Annexe 2).

- **Créer les fichiers de configuration dans Windows** :
  - Créez les fichiers suivants dans `Symfony-StarterKit/etc/docker/config/` :
    - `hosts` (Annexe 3)
    - `starterkit_symfony.conf` (Annexe 4)

- **Mettre à jour le fichier `hosts` de Windows** :
  - Si nécessaire, ajoutez la ligne suivante dans `C:/Windows/System32/drivers/etc/hosts` :
    ```bash
    127.0.0.1       starterkit_symfony.localhost
    ```

---

## 7. Lancer à Nouveau Docker

- **Redémarrer Docker** :
  - Exécutez à nouveau :
    ```bash
    docker-compose up -d
    ```

---

## 8. Tester le Projet

- **Ajouter les fichiers de test** :
  - Pour tester, ajoutez les fichiers suivants au projet :
    - `app/src/Controller/BaseController.php` (Annexe 5)
    - `app/templates/lucky/number.html.twig` (Annexe 6)

- **Accéder au projet via le navigateur** :
  - Ouvrez `http://www.starterkit_symfony.localhost/home` pour vérifier si le projet fonctionne.

---

## 9. Configuration de la Base de Données (BDD)

- **Créer un utilisateur et accorder les privilèges** :
  - Dans le container MySQL, exécutez :
    ```sql
    CREATE USER 'starterkit'@'localhost' IDENTIFIED BY 'starterkit';
    CREATE USER 'starterkit'@'%' IDENTIFIED BY 'starterkit';  -- Pour permettre la connexion depuis n'importe où
    GRANT ALL PRIVILEGES ON *.* TO 'starterkit'@'%' WITH GRANT OPTION;
    GRANT ALL PRIVILEGES ON *.* TO 'starterkit'@'localhost' WITH GRANT OPTION;
    ```

- **Créer la base de données** :
  - Dans le container, exécutez :
    ```bash
    php bin/console doctrine:database:create
    ```

---

## 10. Ajouter un Utilisateur et Authentification

- **Générer l'utilisateur et l'authentification** :
  - Exécutez les commandes suivantes pour configurer l'authentification :
    ```bash
    php bin/console make:user
    php bin/console make:auth
    php bin/console make:registration-form
    php bin/console make:controller ProfileController
    ```

---

## 11. Exécuter les Migrations Doctrine

- **Exécuter une migration spécifique** :
  - Pour effectuer une migration doctrine, exécutez la commande :
    ```bash
    php bin/console doctrine:migrations:exec 'DoctrineMigrations\Version20220201250222'
    ```

---

## Annexes

### **Annexe 1 : Configuration Docker `docker-compose.yml`**
```yaml
version: "3"
services:
    starterkit_symfony:
        container_name: starterkit_symfony
        image: TROUVER_UNE_IMAGE_UBUNTU_SUR_DOCKERHUB/ubuntu-php8:php81
        working_dir: /var/www/html
        volumes:
            - ../../app:/var/www/html/app
            - ../../etc:/mnt
        ports:
            - 80:80
            - 443:443
```

### **Annexe 2 : Configuration Docker (Windows) `docker-compose.yml`**
```yaml
version: "3"
services:
    starterkit_symfony:
        container_name: starterkit_symfony
        image: TROUVER_UNE_IMAGE_UBUNTU_SUR_DOCKERHUB/ubuntu-php8:php81
        working_dir: /var/www/html
        volumes:
            - ./config/starterkit_symfony.conf:/etc/apache2/sites-enabled/starterkit_symfony.conf
            - ./config/hosts:/etc/hosts
            - ../../app:/var/www/html/app
            - ../../etc:/mnt
        ports:
            - 80:80
            - 443:443
```

### **Annexe 3 : Fichier `hosts`**
```
127.0.0.1       starterkit_symfony.localhost
```

### **Annexe 4 : Configuration Apache `starterkit_symfony.conf`**
```apache
<VirtualHost *:80>
    ServerName starterkit_symfony.localhost
    ServerAlias www.starterkit_symfony.localhost
    DocumentRoot /var/www/html/app/public
    <Directory /var/www/html/app/public>
        AllowOverride None
        Order Allow,Deny
        Allow from All
        FallbackResource /index.php
    </Directory>
    <Directory /var/www/html/app/public/bundles>
        FallbackResource disabled
    </Directory>
    ErrorLog /var/log/apache2/starterkit_symfony_error.log
    CustomLog /var/log/apache2/starterkit_symfony.log combined
    SetEnvIf Authorization "(.*)" HTTP_AUTHORIZATION=$1
</VirtualHost>
```

### **Annexe 5 : BaseController**
```php
<?php
namespace App\Controller;

use Symfony\Bundle\FrameworkBundle\Controller\AbstractController;
use Symfony\Component\HttpFoundation\Response;
use Symfony\Component\Routing\Annotation\Route;

class BaseController extends AbstractController
{
    #[Route('/lucky/number')]
    public function number(): Response
    {
        $number = random_int(0, 100);
        return $this->render('lucky/number.html.twig', [
            'number' => $number,
        ]);
    }
}
```

### **Annexe 6 : `number.html.twig`**
```twig
<h1>Your lucky number is {{ number }}</h1>
```

---

Ce document vous guidera à travers les étapes de mise en place d'un projet Symfony avec Docker. Assurez-vous de suivre chaque étape attentivement pour éviter les erreurs.
