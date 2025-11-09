# API Platform

### Installation d'API Platform

Pour installer API Platform dans votre projet Symfony, suivez ces étapes :

1. **Installer API Platform** :
    ```bash
    composer require api
    composer require api-platform/core
    ```

### Fichiers modifiés lors de l'installation

L'installation d'API Platform modifie plusieurs fichiers importants de votre projet :

- `.env`
- `composer.json`
- `composer.lock`
- `symfony.lock`
- `config/bundles.php`
- `config/packages/nelmio_cors.yaml`
- `config/packages/api_platform.yaml`

### Ressources de formation

Pour vous aider à bien comprendre et utiliser API Platform, voici quelques tutoriels recommandés :

- **SymfonyCasts** : [Tutoriel vidéo API Resource](https://symfonycasts.com/screencast/api-platform/api-resource)
- **Nouvelle Techno** : [Vidéo YouTube - Introduction à API Platform](https://www.youtube.com/watch?v=TCx2M6KRn9c)
- **Documentation officielle Symfony** : [Le Fast Track Symfony - API](https://symfony.com/doc/current/the-fast-track/fr/26-api.html)
- **Grafikart** : [Vidéo YouTube - API Platform](https://www.youtube.com/watch?v=Ap6l56bLQtQ&list=PLjwdMgw5TTLU7DcDwEt39EvPBi9EiJnF4)

---

### Utilisation d'API Platform avec des annotations sur les entités

API Platform permet de créer facilement des APIs avec des annotations sur vos entités.

#### 1. Ajouter API Platform à une entité

Pour que votre entité devienne une ressource API, ajoutez l'annotation suivante à la classe de l'entité :

```php
use ApiPlatform\Metadata\ApiResource;

#[ApiResource]
class Product
{
    // Définir vos propriétés et méthodes...
}
```

#### 2. Configurer les opérations de l'API

Vous pouvez personnaliser les opérations de votre API pour chaque entité. Par exemple, pour définir des opérations de `GET` avec des groupes de normalisation spécifiques, vous pouvez ajouter cette annotation :

```php
#[ApiResource(
    operations: [
        new Get(normalizationContext: ['groups' => 'product:item']),
        new GetCollection(normalizationContext: ['groups' => 'product:list'])
    ],
    order: ['name' => 'DESC', 'description' => 'ASC'],
    paginationEnabled: false
)]
class Product
{
    // Définir vos propriétés et méthodes...
}
```

Cela permet de configurer des groupes de données à exposer via l'API. Par exemple, l'API `GET` pour un produit renverra les données avec le groupe `product:item`, tandis que l'API `GET` pour la collection renverra les données avec le groupe `product:list`.

#### 3. Ajouter des groupes de normalisation sur les attributs

Pour inclure des attributs dans des groupes de normalisation spécifiques, utilisez l'annotation `#[Groups]` sur chaque propriété de l'entité :

```php
use Symfony\Component\Serializer\Annotation\Groups;

class Product
{
    #[Groups(['product:list', 'product:item'])]
    private $name;

    #[Groups(['product:list', 'product:item'])]
    private $description;

    // Autres propriétés et méthodes...
}
```

Les groupes permettent de contrôler quelle partie des données sera incluse dans la réponse de l'API, en fonction de l'opération demandée.

---

API Platform simplifie grandement la création d'APIs RESTful dans Symfony, en utilisant des annotations sur les entités pour exposer des ressources via une API. Avec les ressources et tutoriels ci-dessus, vous pouvez rapidement mettre en place des APIs performantes et personnalisées pour vos applications Symfony.
