# Symfony

## Gestion des Routes avec Symfony

### Exemple de Routes Structurées et Non Structurées
Symfony permet d’organiser les routes de manière structurée ou non selon le besoin.

- **Route Structurée** :
  ```php
  /**
   * @Route("/view_url/{documentId}", name="document_add_consultation", methods={"GET"})
   */
  ```
  Utilise un paramètre dynamique (`{documentId}`) pour accéder à des ressources spécifiques dans l'URL.

- **Route Non Structurée** :
  ```php
  /**
   * @Route("/services/questionnaire-satisfaction", name="questionnaire_satisfaction", methods={"GET", "POST"})
   */
  ```
  Route avec un chemin statique, sans paramètres dynamiques.

## Pages d'Erreur en Environnement de Développement
Symfony offre des pages d'erreur intégrées pour aider à diagnostiquer les problèmes lors du développement :

- Accéder aux pages d'erreur 403 et 404 :
  - `localhost/_error/403`
  - `localhost/_error/404`

## Commandes Symfony Utiles pour la Console
Quelques commandes essentielles pour manipuler et déboguer une application Symfony :

- **Lister les routes** :
  ```bash
  php bin/console debug:router
  ```

- **Vider le cache** :
  ```bash
  php bin/console cache:clear
  ```

## Exécution de Migrations Doctrine
Les migrations permettent de gérer les changements de schéma de base de données. Voici quelques commandes essentielles :

- **Exécuter les migrations** :
  ```bash
  php bin/console doctrine:migrations:migrate
  ```

- **Exécuter une migration spécifique** :
  ```bash
  php bin/console doctrine:migrations:exec "Migrations\Versionxxxxxxxxxxxxxxx"
  ```

## Gestion des Services et Injection de Dépendance

### Exemple de Déclaration de Service dans services.yaml
Pour enregistrer un service et l’injecter dans des contrôleurs ou autres services :

```yaml
# config/services.yaml
App\Service\CustomService:
    arguments:
        $parameter: '%custom_parameter%'
```

### Utiliser un Service dans un Contrôleur
Injectez un service en le passant dans le constructeur d’un contrôleur :

```php
use App\Service\CustomService;

class MyController extends AbstractController
{
    private $customService;

    public function __construct(CustomService $customService)
    {
        $this->customService = $customService;
    }

    public function index()
    {
        // Utilisation du service
        $this->customService->doSomething();
    }
}
```

## Les Profils de Logging (Environnements dev, test, prod)
Symfony utilise différents profils de configuration (environnements) pour ajuster la gestion des logs, performances et gestion d'erreurs.

- **Environnement de Développement** (`dev`) :
  - Logs détaillés, profilage activé, erreurs affichées dans le navigateur.
  
- **Environnement de Production** (`prod`) :
  - Logs minimaux, erreurs cachées, optimisations de performances activées.

## Bonnes Pratiques pour le Développement Symfony
1. **Respecter la Convention PSR-4** pour l'organisation des dossiers.
2. **Utiliser les Types** dans les signatures de méthodes pour un code plus clair et plus sûr.
3. **Tester le Code avec PHPUnit** et d'autres outils de test intégrés.

---

## Erreurs Courantes et Résolutions

### Erreur `Class Not Found`
- **Cause** : Mauvais namespace ou fichier manquant.
- **Solution** : Vérifiez les noms de classe, le namespace, et les importations (use).

### Erreur 500 lors du chargement de pages
- **Cause** : Espace superflu dans le fichier `.env.local`.
- **Solution** : Retirer les espaces autour des signes `=` dans les déclarations de variables d'environnement.

### Erreur de Cache en Développement
- **Symptôme** : Modifications non prises en compte.
- **Solution** : Exécuter `php bin/console cache:clear` pour forcer le rafraîchissement du cache.

---

## Ressources et Liens Utiles

- [Documentation Symfony](https://symfony.com/doc/current/index.html)
- [SymfonyCasts](https://symfonycasts.com/)
- [Packagist - Extensions Symfony](https://packagist.org/)

---