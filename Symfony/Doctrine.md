# Doctrine avec Symfony

Doctrine est un ORM (Object Relational Mapper) qui simplifie l’interaction avec la base de données en Symfony, en mappant les entités PHP aux tables de base de données. Ce guide présente sa configuration, l'utilisation des entités et des relations, ainsi que des requêtes de base et avancées.

---

## 1. Configuration de Doctrine

Doctrine est préconfiguré avec Symfony, mais voici les étapes clés pour vérifier et ajuster la configuration si nécessaire :

1. **Fichier de configuration** : La configuration principale se trouve dans `config/packages/doctrine.yaml` :
    ```yaml
    doctrine:
        dbal:
            url: '%env(resolve:DATABASE_URL)%'
        orm:
            auto_generate_proxy_classes: true
            naming_strategy: doctrine.orm.naming_strategy.underscore_number_aware
            auto_mapping: true
    ```

2. **Définition de l’URL de connexion** : Définissez `DATABASE_URL` dans le fichier `.env` pour indiquer la base de données à utiliser :
    ```env
    DATABASE_URL="mysql://db_user:db_password@127.0.0.1:3306/db_name"
    ```

---

## 2. Création d’une Entité

Les entités sont des classes qui représentent des tables en base de données et facilitent les opérations CRUD.

1. **Créer une entité** : Utilisez la commande suivante pour créer une entité :
    ```bash
    php bin/console make:entity
    ```

2. **Exemple d’entité Product** :
    ```php
    namespace App\Entity;

    use Doctrine\ORM\Mapping as ORM;

    #[ORM\Entity]
    #[ORM\Table(name: "product")]
    class Product
    {
        #[ORM\Id]
        #[ORM\GeneratedValue]
        #[ORM\Column(type: "integer")]
        private $id;

        #[ORM\Column(type: "string", length: 100)]
        private $name;

        #[ORM\Column(type: "decimal", precision: 10, scale: 2)]
        private $price;
        
        // Getters et Setters...
    }
    ```

3. **Créer et exécuter les migrations** : Générez et appliquez les migrations pour synchroniser la base de données avec l’entité.
    ```bash
    php bin/console make:migration
    php bin/console doctrine:migrations:migrate
    ```

---

## 3. Relations entre Entités

Doctrine permet de définir des relations entre entités, telles que `OneToOne`, `OneToMany`, et `ManyToMany`.

1. **OneToOne** : Relation entre `User` et `Profile`.
    ```php
    #[ORM\OneToOne(targetEntity: Profile::class, cascade: ["persist", "remove"])]
    #[ORM\JoinColumn(nullable: false)]
    private $profile;
    ```

2. **OneToMany / ManyToOne** : Relation entre `Category` et `Product`.
    ```php
    #[ORM\OneToMany(targetEntity: Product::class, mappedBy: "category")]
    private $products;
    
    #[ORM\ManyToOne(targetEntity: Category::class, inversedBy: "products")]
    private $category;
    ```

3. **ManyToMany** : Relation entre `Product` et `Tag`.
    ```php
    #[ORM\ManyToMany(targetEntity: Tag::class, inversedBy: "products")]
    #[ORM\JoinTable(name: "product_tag")]
    private $tags;
    ```

---

## 4. Requêtes avec Doctrine

Doctrine propose plusieurs méthodes pour exécuter des requêtes sur la base de données.

### Utilisation du Repository

Chaque entité a un repository qui gère les opérations de base. Pour créer un repository personnalisé :
```bash
php bin/console make:repository
```

1. **Requêtes simples** :
    ```php
    $product = $productRepository->find($id);
    $products = $productRepository->findAll();
    $productByName = $productRepository->findOneBy(['name' => 'example']);
    ```

2. **Requêtes avancées avec QueryBuilder** :
    ```php
    $query = $productRepository->createQueryBuilder('p')
        ->where('p.price > :price')
        ->setParameter('price', 10)
        ->orderBy('p.name', 'ASC')
        ->getQuery();

    $products = $query->getResult();
    ```

---

## 5. Commandes Doctrine Utiles

- **Créer une entité** : `php bin/console make:entity`
- **Créer une migration** : `php bin/console make:migration`
- **Exécuter les migrations** : `php bin/console doctrine:migrations:migrate`
- **Vider la base de données** : `php bin/console doctrine:schema:drop --force`
- **Synchroniser la base de données sans migration** : `php bin/console doctrine:schema:update --force`

### Exécuter une Migration Spécifique
Depuis Symfony 6, la commande `doctrine:migrations:exec` permet d'exécuter une migration spécifique :
```bash
php bin/console doctrine:migrations:exec "Migrations\Versionxxxxxxxxxxxxxxx"
php bin/console doctrine:migrations:exec "DoctrineMigrations\Versionxxxxxxxxxxxxxxx"
```

### Mettre à Jour la Base de Données
Pour mettre à jour la base de données complètement (sans utiliser les migrations), vous pouvez utiliser :
```bash
php bin/console doctrine:schema:update --dump-sql  # Affiche les SQL à exécuter
php bin/console doctrine:schema:update --force     # Applique les changements directement
```

---

## 6. Exemples de Requêtes SQL Personnalisées

Doctrine permet d'exécuter des requêtes SQL brutes ou de construire des requêtes personnalisées en DQL (Doctrine Query Language).

1. **Requête SQL brute** :
    ```php
    $connection = $entityManager->getConnection();
    $sql = 'SELECT * FROM product WHERE price > :price';
    $stmt = $connection->prepare($sql);
    $result = $stmt->executeQuery(['price' => 10])->fetchAllAssociative();
    ```

2. **Requête DQL** :
    ```php
    $query = $entityManager->createQuery(
        'SELECT p FROM App\Entity\Product p WHERE p.price > :price'
    )->setParameter('price', 10);

    $products = $query->getResult();
    ```
