# Python - Fonctions et Modules

## Exceptions

**Gestion des erreurs:**

* `try`: Exécuter un bloc de code.
* `except`: Capturer une exception et exécuter un bloc de code.
* `else`: Exécuter un bloc de code si aucune exception n'est levée.
* `finally`: Exécuter un bloc de code quoi qu'il arrive.

**Exemple:**

```python
try:
    resultat = 3 / 0
except ZeroDivisionError:
    print("Division par zéro impossible !")
else:
    print(resultat)
```

**Lever une exception:**

* `raise ValueError("Message d'erreur")`

**Fonctions utiles:**

* `type(e)`: Obtenir le type de l'exception.
* `e.args`: Obtenir les arguments de l'exception.

**Conseils:**

* Utiliser des exceptions pour gérer les erreurs prévisibles.
* Documenter les exceptions que votre code peut lever.

## Fonctions

**Définir une fonction:**

```python
def ma_fonction():
    print("Bonjour !")
```

**Appeler une fonction:**

```python
ma_fonction()
```

**Arguments de la fonction:**

* **Arguments par défaut:**

```python
def ma_fonction(nom="inconnu"):
    print(f"Bonjour, {nom} !")

ma_fonction()  # Affiche "Bonjour, inconnu !"
ma_fonction("Pierre")  # Affiche "Bonjour, Pierre !"
```

**Valeurs de retour:**

```python
def ma_fonction():
    return "Bonjour !"

resultat = ma_fonction()
print(resultat)  # Affiche "Bonjour !"
```

## Modules

**Importer un module:**

```python
import math
```

**Fonctions du module `math`:**

* `math.ceil(-4.7)`: Arrondir à l'entier supérieur.
* `math.exp(2)`: Calculer l'exponentielle.
* `math.sqrt(2)`: Calculer la racine carrée.
* `math.pi`: Accéder à la constante π.

**Module `random`:**

* Générer des nombres aléatoires.
* `random.randint(0, 2)`: Générer un entier aléatoire entre 0 et 2.
* `random.uniform(0, 1)`: Générer un nombre réel aléatoire entre 0 et 1.
* `random.randrange(1)`: Générer un entier aléatoire entre 0 et 1 (exclu).
* `random.randrange(0, 101, 10)`: Générer un entier aléatoire entre 0 et 100 (inclus) avec un pas de 10.

**Module `os` (obsolète):**

* Fonctions liées au système d'exploitation.
* `os.path.join(chemin, "dossier", "test")`: Concaténer des chemins de fichiers.
* `os.makedirs(dossier, exist_ok=True)`: Créer un dossier (s'il n'existe pas déjà).
* `os.path.exists(dossier)`: Vérifier si un dossier existe.
* `os.removedirs(dossier)`: Supprimer un dossier vide.

**Module `pprint`:**

* Affiche des structures de données de manière plus lisible.
* `from pprint import pprint`
* `pprint(liste)`

**Module `json`:**

* Permet de lire et d'écrire des données JSON.

**Ecrire en JSON:**

```python
with open(chemin, "w") as f:
    json.dump(list(range(10)), f, indent=4)
```

**Récupérer des données JSON:**

```python
with open(chemin, "r") as f:
    liste = json.load(f)
    print(type(liste))
```

**Modifier des données JSON:**

* Ouvrir le fichier en mode lecture/écriture ("r+").
* Modifier les données.
* Ecraser le fichier avec les nouvelles données.

## `locals()` et `globals()`

* `locals()`: Obtenir un dictionnaire contenant les variables locales de la fonction courante.
* `globals()`: Obtenir un dictionnaire contenant les variables globales du programme.

**Exemple:**

```python
def ma_fonction():
    x = 1
    print(locals())

ma_fonction()

# Affiche : {'x': 1}

print(globals())

# Affiche : {'__name__': '__main__', '__doc__': None, ...}
```

## Annotations de typage

* Permettent de spécifier le type des variables et des valeurs de retour des fonctions.
* Améliorent la lisibilité du code et permettent de détecter des erreurs de typage à la compilation.

**Exemple:**

```python
def add(a: int, b: int) -> int:
    """
    Additionne deux nombres entiers.

    Args:
        a (int): Le premier nombre.
        b (int): Le deuxième nombre.

    Returns:
        int: La somme des deux nombres.
    """
    return a + b

result = add(2, 4)
print(result)  # Affiche 6
```

## Générateurs (`yield`)

* Permettent de générer des séquences de valeurs de manière itérative.
* Plus économes en mémoire que les listes pour les séquences de grande taille.

**Exemple:**

```python
def generateur_multiplication_par_2():
    """
    Génère les multiples de 2 jusqu'à un certain nombre.

    Args:
        n (int): Le nombre maximal de multiples à générer.

    Yields:
        int: Les multiples de 2.
    """
    for i in range(4):
        yield i * 2

mon_generateur = generateur_multiplication_par_2()

for i in mon_generateur:
    print(i)

# Affiche :
# 0
# 2
# 4
# 6
```

**Comparaison avec les listes:**

```python
ma_liste = (x * 2 for x in range(4))

for i in ma_liste:
    print(i)

# Affiche :
# 0
# 2
# 4
# 6
```

**Avantages des générateurs:**

* Plus économes en mémoire.
* Plus performants pour les séquences de grande taille.
* Permettent de générer des séquences infinies.
