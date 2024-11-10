# Twig

## Syntaxe de base

### Affichage de variables
- Utilisez `{{ ... }}` pour afficher une variable.

    ```twig
    {{ variableName }}
    ```

### Exécution de code
- Utilisez `{% ... %}` pour exécuter des instructions logiques comme des boucles et des conditions.

    ```twig
    {% if condition %}
        <!-- code -->
    {% endif %}
    ```

### Commentaires
- Utilisez `{# ... #}` pour ajouter des commentaires qui ne seront pas affichés dans le rendu HTML.

    ```twig
    {# Ceci est un commentaire #}
    ```

## Fonctions et filtres utiles

### Join
- Utilisez `|join(', ')` pour joindre les éléments d'un tableau avec un séparateur.

    ```twig
    {{ arrayVariable|join(', ') }}
    ```

### Merge
- Utilisez `|merge(array2)` pour fusionner deux tableaux.

    ```twig
    {% set mergedArray = array1|merge(array2) %}
    ```

## Boucles et conditions

### Boucle `for`
- Structure de base pour boucler sur les éléments d’un tableau.

    ```twig
    {% for item in array %}
        {{ item }}
    {% endfor %}
    ```

- **Utilisation de propriétés dans une boucle :**
  - `loop.first` : vrai si c'est le premier élément.
  - `loop.last` : vrai si c'est le dernier élément.
  - `loop.index` : index courant (commence à 1).
  - `loop.index0` : index courant (commence à 0).

    ```twig
    {% for item in array %}
        {% if loop.first %}
            <!-- Code pour le premier élément -->
        {% elseif loop.last %}
            <!-- Code pour le dernier élément -->
        {% else %}
            <!-- Code pour les autres éléments -->
        {% endif %}
    {% endfor %}
    ```

## Filtres et fonctions additionnelles

### Exemple d'utilisation de filtres
- **Uppercase** : Convertit une chaîne en majuscules.
    ```twig
    {{ 'texte'|upper }}
    ```

- **Date formatting** : Formate une date selon un format donné.
    ```twig
    {{ dateVariable|date("d/m/Y") }}
    ```

### Conditions avancées
- **Condition ternaire** : Raccourci pour une condition simple.

    ```twig
    {{ variable ? 'True' : 'False' }}
    ```

- **Null coalescing operator** : Fournit une valeur par défaut si la variable est `null`.

    ```twig
    {{ variable ?? 'valeur_par_defaut' }}
    ```

---

### Notes
1. **Utiliser les filtres de sécurité** : Pour éviter les failles XSS, assurez-vous de bien échapper les variables si elles contiennent des données utilisateur.
2. **Débogage dans Twig** : Utilisez `dump(variable)` pour afficher la variable pendant le développement. Assurez-vous que le mode de débogage est activé dans Symfony.
