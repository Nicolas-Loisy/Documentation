# Python

## VS Code
### Problème : Navigation dans les imports non fonctionnelle (Ctrl + Click) sous VS Code

Ajoutez cette ligne dans `settings.json` (**Ctrl+Shift+P** > `Settings`)  :

```json
"python.analysis.extraPaths": ["./src"]
```

Ensuite, rechargez la fenêtre avec **Ctrl+Shift+P** > `Reload Window` pour appliquer. Cela rétablit la navigation entre les fichiers.


## Types de données

**Chaînes de caractères (str):**

* `r"maString"` => Interprète en "raw", permet de mettre des /t, ...
* `.upper()`: Convertit la chaîne en majuscules
*`.lower()`: Convertit la chaîne en minuscules
* `.capitalize()`: Met la première lettre en majuscule et le rste en minuscules
* `.title()`: Met la première lettre de chaque mot en majuscule
* `"Le jour bonjour".count("jour")`: Compte le nombre d'occurences de "jour"
* `"Le jour bonjour".index("jour")`: Renvoie l'indexe de la première occurrence de "jour"
* `"Le jour bonjour".find("jour")`: Idem que `index()` mais renvoie -1 si non trouvé
* `"Le jour bonjour".rfind("jour")`: Renvoie l'indexe de la dernière occurrence de "jour"
* `"Le jour bonjour".lfind("jour")`: Renvoie l'indexe de la première occurrence de "jour" en partant de la gauche
* `"bonjour".replace("jour", "soir")`: Remplace "jour" par "soir"
* `"image.png".endswith(".png") => True`: Vérifie si la chaîne se termine par ".png"
* `"image.png".startswith(".png") => False`: Vérifie si la chaîne commence par ".png"
* `" bon jour ".strip() => "bon jour"`: Supprime les espaces blancs en début et fin de chaîne
* `" bonjour ".strip(" ujor") => "bon"`: Supprime les caractères "u", "j", "o" et "r" en début et fin de chaîne
* `"1, 2, 3, 4".split(", ") => ['1', '2', '3', '4']`: Divise la chaîne en une liste en utilisant "," comme séparateur
* `".".join(['1', '2', '3', '4']) => '1.2.3.4'`: Joint les éléments d'une liste en une chaîne en utilisant "." comme séparateur

**F-strings:**

* Permettent de formater des chaînes de manière plus lisible
* Exemple:

```python
protocole = "https://"
nom_du_site = "docstring"
extension = "fr"
page = "glossaire"
URL = f"{protocole}www.{nom_du_site}.{extension}/{page}/"
```

**Méthode `format() :`**

* Permet de formater des chaînes en utilisant des expressions
* Exemple:

```python
age = 26
phrase = "J'ai {age} ans".format(age=age)
```

**Entiers (int):**

* Nombres entiers

**Nombres à virgule flottante (float):**

* Nombres avec virgule

**Booléens (bool):**

* True ou False

## Variables

* `id(maVariable)` => Renvoie l'adresse mémoire de la variable
* `a, b, c = 5, 8, 3`: Déclaration de plusieurs variables en une seule ligne
* `a, b = b, a`: Permet d'échanger les valeurs de deux variables
* `a = b = c = 5`: Affecte la valeur 5 aux variables `a`, `b` et `c`

## Conditions

**Instruction `if`:**

* Permet d'exécuter du code en fonction d'une condition
* Exemple:

```python
if age >= 18:
    print("Vous êtes majeur")
elif age < 18:
    print("Vous êtes mineur")
else:
    print("Âge non valide")
```

**Opérateur ternaire:**

* Permet de simplifier une instruction `if`
* Exemple:

```python
majeur = True if age >= 18 else False
```

## Boucles

**Boucle `for`:**

* Permet d'exécuter un bloc de code un nombre défini de fois
* Exemples:

```python
for i in [0, 1, 4, 7, 8]:
    print(i)

for i in range(1000):
    print(i)
```

**Boucle `while`:**

* Permet d'exécuter un bloc de code tant qu'une condition est vraie
* Exemple:

```python
i = 0
while i < 100:
    print("Hello")
    i += 1
```



## Opérateurs

**Opérateurs arithmétiques:**

* `/`: Division. **Exemple:** `5 / 2 = 2.5`.
* `//`: Division entière. **Exemple:** `5 // 2 = 2`.
* `**`: Puissance. **Exemple:** `2 ** 3 = 8`.

**Opérateurs logiques:**

* `and`: Et logique. **Exemple:** `True and True = True`.
* `or`: Ou logique. **Exemple:** `False or True = True`.
* `not`: Non logique. **Exemple:** `not True = False`.
* `==`: Égalité. **Exemple:** `5 == 5 = True`.
* `!=`: Différence. **Exemple:** `5 != 5 = False`.

**Opérateurs d'appartenance:**

* `in`: Test d'appartenance. **Exemple:** `"a" in "abc" = True`.
* `not in`: Test de non-appartenance. **Exemple:** `"a" not in "bcd" = True`.

**Exemples:**

```python
# Division
5 / 2  # 2.5

# Division entière
5 // 2  # 2

# Puissance
2 ** 3  # 8

# Et logique
True and True  # True

# Ou logique
False or True  # True

# Non logique
not True  # False

# Égalité
5 == 5  # True

# Différence
5 != 5  # False

# Test d'appartenance
"a" in "abc"  # True

# Test de non-appartenance
"a" not in "bcd"  # True
```

## Listes

**Fonctions de base:**

* `list(...)`: Convertir un objet en liste. **Exemple:** `list("abc") = ['a', 'b', 'c']`.
* `liste.append("valeur")`: Ajouter une valeur à la fin de la liste. **Exemple:** `liste.append(4)`.
* `liste.extend([10, 34, 56])`: Ajouter plusieurs valeurs à la fin de la liste. **Exemple:** `liste.extend([10, 34, 56])`.
* `liste.remove(5)`: Supprimer la première occurrence de la valeur 5. **Exemple:** `liste.remove(5)`.
* `liste.index("valeur")`: Récuperer l'indexe de la première occurrence de la valeur. **Exemple:** `liste.index("a")`.
* `liste.count("valeur")`: Récuperer le nombre d'occurrences de la valeur. **Exemple:** `liste.count("a")`.
* `liste.sort()`: Trier la liste par ordre croissant. **Exemple:** `liste.sort()`.
* `sorted(liste)`: Idem que `liste.sort()` mais renvoie une nouvelle liste. **Exemple:** `sorted(liste)`.
* `liste.reverse()` : Inverser l'ordre de la liste. **Exemple:** `liste.reverse()`.
* `liste.pop(-1)` : Supprimer le dernier élément de la liste. **Exemple:** `liste.pop(-1)`.
* `liste.clear()` : Vider la liste. **Exemple:** `liste.clear()`.

**Fusion et découpage de chaînes:**

* `" ".join(["Bonjour", "le", "monde", "!"])`: Fusionner une liste de chaînes en une seule chaîne avec un séparateur. **Exemple:** `", ".join(["Bonjour", "le", "monde", "!"]) = "Bonjour le monde !"`.
* `"Riz, Pomme, Lait".split(", ")`: Découper une chaîne en une liste en utilisant un séparateur. **Exemple:** `"Riz, Pomme, Lait".split(", ") = ['Riz', 'Pomme', 'Lait']`.

**Slicing (tranches):**

* `liste[0:2]`: Accéder à une sous-liste de la liste (de l'indice 0 à 2 exclus). **Exemple:** `liste[0:2] = ['a', 'b']`.
* `liste[:-1]`: Accéder à une sous-liste de la liste (du début à l'avant-dernier élément). **Exemple:** `liste[:-1] = ['a', 'b', 'c']`.
* `liste[2:]`: Accéder à une sous-liste de la liste (du 3ème élément à la fin). **Exemple:** `liste[2:] = ['c', 'd', 'e']`.
* `liste[1:-2:2]`: Accéder à une sous-liste de la liste (du 2ème élément à l'avant-dernier élément avec un pas de 2). **Exemple:** `liste[1:-2:2] = ['b', 'd']`.
* `liste[::-1]`: Inverser l'ordre de la liste. **Exemple:** `liste[::-1] = ['e', 'd', 'c', 'b', 'a']`.

**Compréhensions de listes:**

* Permettent de créer des listes de manière concise.
* **Exemple:**

```python
# Créer une liste des nombres pairs de 1 à 10
nombres_pairs = [i for i in range(1, 11) if i % 2 == 0]

# Créer une liste des noms des personnes majeures
personnes = [{"nom": "Pierre", "age": 25}, {"nom": "Julie", "age": 18}, {"nom": "Paul", "age": 32}]
noms_majeurs = [personne["nom"] for personne in personnes if personne["age"] >= 18]
```

## Tuples

**Similaires aux listes mais non modifiables.**

* `tuple(...)`: Convertir un objet en tuple. **Exemple:** `tuple("abc") = ('a', 'b', 'c')`.
* `mon_tuple = (1, 2, 3)`
* `liste = list(mon_tuple)`  # Convertir un tuple en liste
* `[1, 2, 3]`
* `mon_tuple = tuple(liste)`  # Convertir une liste en tuple
* `(1, 2, 3)`

## Fonctions basiques

* `input("Votre nombre :")`: Demander à l'utilisateur de saisir une valeur.
* `dir(random)`: Afficher la liste des fonctions du module `random`.
* `help(random.randint)`: Afficher l'aide de la fonction `randint` du module `random`.
* `callable(pprint)`: Vérifier si un objet est une fonction callable.

**Fonctions de chaînes de caractères:**

* `.islower()`: Vérifier si la chaîne est en minuscules.
* `.istitle()`: Vérifier si la chaîne est en majuscule au début de chaque mot.
* `.isdigit()`: Vérifier si la chaîne est composée uniquement de chiffres.

**Fonctions utiles:**

* `len()`: Obtenir la longueur d'une liste ou d'une chaîne.
* `round()`: Arrondir un nombre.
* `min()`: Obtenir la valeur minimale d'une liste.
* `max()`: Obtenir la valeur maximale d'une liste.
* `sum()`: Obtenir la somme des éléments d'une liste.
* `range()`: Créer une liste de nombres.

**Fonctions logiques:**

* `any([False, False, True, False])`: True si au moins un élément de la liste est True.
* `all([False, False, True, False])`: False si tous les éléments de la liste sont False.

## Arguments des scripts

* Permettent de passer des arguments au script lors de son exécution.
* **Exemple:**

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

print(f"Tâche : {args.task}")
print(f"Langage : {args.language}")
```

## Dictionnaires

**Accès et manipulation:**

* `d[clé]`: Accéder à la valeur associée à la clé. **Exemple:** `d[0]["prenom"] = "Paul"`.
* `d.get(clé, valeur_par_défaut)`: Obtenir la valeur associée à la clé, sinon renvoyer la valeur par défaut. **Exemple:** `d.get("age", "La clé n'existe pas !")`.
* `d.keys()`: Obtenir un ensemble des clés du dictionnaire. **Exemple:** `d.keys() = {0, 1}`.
* `d.values()`: Obtenir une liste des valeurs du dictionnaire. **Exemple:** `d.values() = ["Paul", "Pierre"]`.
* `d.items()`: Obtenir une liste des couples clé-valeur du dictionnaire. **Exemple:** `d.items() = {(0, {"prenom": "Paul", "age": 22}), (1, {"prenom": "Pierre", "age": 32})}`.
* `del d[clé]`: Supprimer la clé et sa valeur associée du dictionnaire. **Exemple:** `del d["age"]`.

**Exemple:**

```python
films = {
    "Le Seigneur des Anneaux": 12,
    "Harry Potter": 9,
    "Blade Runner": 7.5,
}

prix = sum(films.values())  # Calculer le prix total des films
print(prix)  # Affiche 28.5
```

## Dates et heures

**Module `datetime`:**

* Permet de manipuler des dates et des heures.
* **Fonctions de base:**
    * `date(year, month, day)`: Créer un objet date.
    * `time(hour, minute, second)`: Créer un objet heure.
    * `datetime(year, month, day, hour, minute, second)`: Créer un objet date et heure.
    * `datetime.today()`: Obtenir la date et l'heure courantes.
* **Attributs d'un objet `datetime`:**
    * `.year`: Année.
    * `.month`: Mois.
    * `.day`: Jour.
    * `.hour`: Heure.
    * `.minute`: Minute.
    * `.second`: Seconde.
* **Méthodes d'un objet `datetime`:**
    * `.replace(year=..., month=..., day=..., ...)`: Modifier la date et/ou l'heure.
    * `+ timedelta(days=...)`: Ajouter un nombre de jours à la date.

**Modules `dateutil.parser` et `dateparser` (externes):**

* Permettent de parser des chaînes de caractères en objets `datetime`.
* **Exemple:**

```python
from datetime import datetime

ma_date = datetime.strptime("22 Oct 2021", "%d %b %Y")
print(ma_date)  # Affiche 2021-10-22 00:00:00
```

**Fuseaux horaires:**

* Module `zoneinfo`: Gérer les fuseaux horaires.
* **Exemple:**

```python
from zoneinfo import ZoneInfo

now_in_montreal = datetime.now(tz=ZoneInfo("America/Vancouver"))
now_in_paris = now_in_montreal.astimezone(ZoneInfo("Europe/Paris"))

print(now_in_montreal)  # Affiche la date et l'heure à Montréal
print(now_in_paris)    # Affiche la date et l'heure à Paris (convertie)
```

## Sérialisation en Python

La sérialisation est le processus de conversion d'un objet Python en un format qui peut être facilement stocké ou transmis, comme JSON. Cela est particulièrement utile lors de l'envoi de données à travers des API ou le stockage d'objets dans des bases de données.

### Fonction `serialize_object`

```python
def serialize_object(obj: Any) -> Any:
    """Convertit les objets non sérialisables en JSON en types sérialisables."""
    if isinstance(obj, ObjectId):
        return str(obj)  # Convertir ObjectId en string
    elif isinstance(obj, dict):
        return {k: serialize_object(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_object(i) for i in obj]
    return obj  # Retourner l'objet tel quel s'il est sérialisable
```

#### Description

- **ObjectId** : Si l'objet est de type `ObjectId`, il est converti en chaîne de caractères pour être compatible avec JSON.
- **Dictionnaires** : Si l'objet est un dictionnaire, chaque clé et valeur est sérialisée récursivement.
- **Listes** : Si l'objet est une liste, chaque élément est également sérialisé récursivement.
- **Autres types** : Les objets déjà sérialisables (comme les chaînes, nombres, booléens) sont retournés sans modification.

### Utilisation

Cette fonction peut être utilisée dans des applications web pour préparer des réponses JSON à partir d'objets complexes, garantissant que tous les types de données sont correctement formatés pour la transmission.


## Fichiers

**Ouverture et fermeture:**

* `f = open(chemin, "mode")`: Ouvrir un fichier en mode lecture ("r"), écriture ("w") ou ajout ("a").
* `f.close()`: Fermer le fichier.

**Exemple:**

```python
with open("fichier.txt", "r", encoding="utf-8") as f:
    contenu = f.read()

print(contenu)
```

**Manipulation du contenu:**

* `f.read()`: Lire le contenu du fichier en une seule fois.
* `f.readlines()`: Lire le contenu du fichier ligne par ligne et retourner une liste de chaînes.
* `f.splitlines()`: Lire le contenu du fichier ligne par ligne et retourner une liste de chaînes sans les caractères de fin de ligne.
* `f.seek(0)`: Repositionner le curseur au début du fichier.

**Ecriture dans un fichier:**

* `f.write(texte)`: Écrire du texte dans le fichier.
* **Exemple:**

```python
with open("fichier.txt", "w", encoding="utf-8") as f:
    f.write("Bonjour, monde !")
```

**Encodage et décodage:**

* Les fichiers sont stockés en octets.
* L'encodage permet de convertir du texte en octets et vice versa.
* L'encodage par défaut est "utf-8".
* **Exemple:**

```python
with open("fichier.txt", "w", encoding="utf-8") as f:
    f.write("Bonjour, monde !")

with open("fichier.txt", "r", encoding="utf-8") as f:
    contenu = f.read()

print(contenu)  # Affiche "Bonjour, monde !"
```

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

### pathlib

**Module `pathlib`:**

* Fournit des classes et des fonctions pour manipuler des chemins de fichiers de manière simple et intuitive.

**Principales fonctionnalités:**

* **Création de chemins:**
    * `Path.cwd()`: Obtenir le chemin du répertoire courant.
    * `Path("leChemin")`: Créer un objet `Path` à partir d'une chaîne de caractères.
    * `Path.home()`: Obtenir le chemin du répertoire personnel.
    * Opérateurs ` / ` et `.joinpath()` pour construire des chemins relatifs.
    * Exemples :
        * `p / "Documents" / "main.py"`
        * `p.joinpath("Documents", "main.py")`
* **Accesseurs d'attributs:**
    * `p.name`: Obtenir le nom du fichier ou du dossier.
    * `p.parent`: Obtenir le répertoire parent.
    * `p.stem`: Obtenir le nom du fichier sans l'extension.
    * `p.suffix`: Obtenir l'extension du fichier.
    * `p.suffixes`: Obtenir une liste des extensions du fichier (pour les liens symboliques).
    * `p.parts`: Obtenir une liste des composants du chemin.
* **Vérifications:**
    * `p.exists()`: Vérifier si le chemin existe.
    * `p.is_dir()`: Vérifier si le chemin est un dossier.
    * `p.is_file()`: Vérifier si le chemin est un fichier.
* **Création et suppression:**
    * `p.mkdir(exist_ok=True, parents=True)`: Créer un dossier (s'il n'existe pas déjà).
    * `p.touch()`: Créer un fichier vide (s'il n'existe pas déjà).
    * `p.unlink()`: Supprimer un fichier.
    * `p.rmdir()`: Supprimer un dossier vide.
* **Lecture et écriture de fichiers:**
    * `p.write_text("contenu")`: Écrire du texte dans un fichier.
    * `p.read_text()`: Lire le contenu d'un fichier.
* **Parcours de répertoire:**
    * `p.iterdir()`: Obtenir un itérateur sur les éléments du dossier (fichiers et sous-dossiers).
    * `p.glob("*.png")`: Obtenir un itérateur sur les fichiers correspondant au motif (ici, tous les fichiers ".png").
    * `p.rglob("*.png")`: Obtenir un itérateur sur tous les fichiers correspondant au motif dans le dossier et ses sous-dossiers.

**Suppression de dossiers non vides:**

* Le module `pathlib` ne permet pas directement de supprimer des dossiers non vides.
* Utiliser le module `shutil` pour cette tâche.
* `import shutil`
* `shutil.rmtree(p)` : Supprime le dossier et son contenu de manière récursive.

### Faker

**Bibliothèque `Faker` (installation requise: `pip install faker`)**

* Génère des données fictives réalistes pour les tests et les prototypes.

**Exemple d'utilisation:**

```python
from faker import Faker

# Initialiser Faker en français
fake = Faker(locale="fr_FR")

# Générer du texte
print(fake.text())

# Générer des noms uniques
print(fake.unique.name())

# Générer des professions
print(fake.job())

# Générer un prénom et un nom
print(f"{fake.first_name()} {fake.last_name()}")

# Générer un chemin de fichier fictif
print(fake.file_path(depth=5, category="video"))

# Générer un numéro de carte bancaire, sa date d'expiration et son code de sécurité
print(fake.credit_card_number(), fake.credit_card_expire(), fake.credit_card_security_code())

# Générer une couleur RGB
print(fake.rgb_color())

# Générer une chaîne de caractères avec des formats personnalisés
print(fake.numerify(text="%%%-#-%%%%-%%%%-%%%-##"))
print(fake.bothify(text="Product Number: ????-#########"))
```

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
* Améliorent la lisbilité du code et permettent de détecter des erreurs de typage à la compilation.

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

## Logging

1. **Importer le module `logging` :**

```python
import logging
```

2. **Configurer le logger de base (facultatif) :**

```python
logging.basicConfig(
    level=logging.INFO,  # Niveau de logging
    filename="app.log",  # Nom du fichier de logs
    filemode="a",       # Mode d'ouverture du fichier (ajout)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format des messages
)
```

**Créer un logger**

```python
logger = logging.getLogger(__name__)  # Nom du logger (souvent le nom du module)
```

**Utiliser les niveaux de logging**

- **DEBUG :** Informations détaillées pour le débogage.
- **INFO :** Informations générales sur le fonctionnement de l'application.
- **WARNING :** Avertissements sur des problèmes potentiels.
- **ERROR :** Erreurs qui empêchent le bon fonctionnement de l'application.
- **CRITICAL :** Erreurs critiques qui interrompent l'application.

```python
logger.debug("Message à afficher en mode DEBUG")
logger.info("Message d'information")
logger.warning("Attention, un problème potentiel a été détecté")
logger.error("Une erreur est survenue")
logger.critical("Erreur critique : l'application doit s'arrêter")
```

## Scrapping avec Beautiful Soup 4

**1. Installation:** `pip install beautifulsoup4`

**2. Importation:**

```python
import requests
from bs4 import BeautifulSoup
```

**3. Récupération et analyse:**

```python
url = "https://www.example.com/"  # Remplacez par l'URL cible
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
```

**4. Recherche d'éléments:**

* Utilisez `find_all` ou `find` avec des noms de balises, des classes ou des ID.

**5. Extraction de données:**

* Utilisez `.text.strip()` pour le contenu, `.get("attr")` pour les attributs.

**6. Boucle et itération:**

```python
for element in elements:
    # Extraire les données de chaque élément
    print(f"Données : {données_extraites}")
```

**Exemple:**

```python
# Récupérer les titres d'actualités

url = "https://www.lemonde.fr/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

titres = soup.find_all("h2", class_="article__title")

for titre in titres:
    # Extraire et afficher le texte du titre
    print(titre.text.strip())
```

## POO (Programmation Orientée Objet) : Classe vs DataClasse

Le texte présenté compare deux approches pour définir des structures de données en Python : les classes et les dataclasses (introduites en Python 3.7).

### Classe :

La définition d'une classe utilise le mot-clé `class`. Voici un exemple :

```python
class Voiture:
  voitures_crees = 0  # Attribut de classe pour compter les instances créées

  def __init__(self, marque, couleur):  # Méthode d'initialisation (constructeur)
    Voiture.voitures_crees += 1
    self.marque = marque  # Attribut d'instance
    self.couleur = couleur

# Création d'une instance (objet)
voiture_01 = Voiture("Lambo")
print(Voiture.voitures_crees)  # Affiche 1
```

**Points clés :**

* La classe **Voiture** définit un modèle pour créer des objets représentant des voitures.
* L'attribut de classe `voitures_crees` stocke le nombre d'instances créées.
* La méthode d'initialisation `__init__` est appelée lors de la création d'une instance.
* Les attributs d'instance (`marque` et `couleur`) stockent les données spécifiques à chaque objet.

### DataClasse (Python >= 3.7) :

Les dataclasses sont des classes simplifiées qui exploitent le décorateur `@dataclass` pour générer automatiquement certaines méthodes standard. Voici un exemple :

```python
from dataclasses import dataclass
from typing import ClassVar

@dataclass
class User:
  first_name: str  # Attribut d'instance obligatoire
  last_name: str = ""  # Attribut d'instance optionnel
  c: ClassVar[int] = 10  # Attribut de classe

  def __post_init__(self):
    self.full_name = f"{self.first_name} {self.last_name}"

# Création d'une instance
john = User(first_name="John")
print(repr(john.first_name))  # Affiche 'John'
print(User.__dict__)  # Affiche les attributs de la classe
print(john.full_name)  # Affiche 'John'
```

**Points clés :**

* Le décorateur `@dataclass` simplifie la définition de la classe.
* Les types d'attributs peuvent être indiqués pour améliorer la lisibilité et la vérification de type.
* Les dataclasses génèrent automatiquement des méthodes comme `__repr__` et `__eq__`.
* La méthode `__post_init__` permet d'exécuter du code supplémentaire après l'initialisation.

## Héritage en Python :

**Exemple 1 : Héritage de la classe `list`**

```python
class Liste(list):  # Héritage de la classe list
  def __init__(self, nom):
    self.nom = nom

# Création d'une instance
ma_liste = Liste("Liste de courses")
ma_liste.append("Pommes")
ma_liste.append("Lait")

print(ma_liste.nom)  # Affiche "Liste de courses"
print(ma_liste)  # Affiche ['Pommes', 'Lait']
```

**Points clés :**

* La classe `Liste` hérite de toutes les fonctionnalités de la classe `list`.
* La méthode `__init__` de la classe `Liste` est ajoutée pour personnaliser l'initialisation.
* L'instance `ma_liste` possède les attributs et les méthodes de la classe `list` et de la classe `Liste`.

**Exemple 2 : Héritage d'une classe personnalisée**

```python
projets = ["pr_GameOfThrones", "HarryPotter", "pr_Avengers"]

class Utilisateur:
  def __init__(self, nom, prenom):
    self.nom = nom
    self.prenom = prenom

  def __str__(self):
    return f"Utilisateur {self.nom} {self.prenom}"

  def afficher_projets(self):
    for projet in projets:
      print(projet)

class Junior(Utilisateur):  # Héritage de la classe Utilisateur
  def __init__(self, nom, prenom):
    super().__init__(nom, prenom)  # Appel à la méthode __init__ de la classe Utilisateur

# Création d'une instance
paul = Junior("Paul", "Durand")

paul.afficher_projets()
print(paul)  # Affiche "Utilisateur Paul Durand"
```

**Points clés :**

* La classe `Junior` hérite de la classe `Utilisateur`.
* La méthode `__init__` de la classe `Junior` utilise `super().__init__` pour initialiser les attributs de la classe `Utilisateur`.
* La méthode `afficher_projets` est définie dans la classe `Utilisateur` et est accessible par l'instance `paul`.

**Fonction `super()` :**

La fonction `super()` permet d'accéder aux méthodes et attributs de la classe parent.

**Exemple :**

```python
class Utilisateur:
  def afficher_projets(self):
    print("Liste des projets :")
    for projet in projets:
      print(projet)

class Junior(Utilisateur):
  def afficher_projets(self):
    super().afficher_projets()  # Appel à la méthode afficher_projets de la classe Utilisateur
    print("Ajout de projets spécifiques aux Juniors")

paul = Junior("Paul", "Durand")
paul.afficher_projets()
```

**Points clés :**

* La méthode `afficher_projets` de la classe `Junior` appelle la méthode du même nom dans la classe `Utilisateur`.
* Cela permet de réutiliser le code de la classe parent et d'ajouter des fonctionnalités spécifiques à la classe `Junior`.

L'héritage est un concept fondamental de la POO qui permet de réutiliser du code et de créer des structures de classes plus complexes.

## Classes abstraites en Python

**Définition:**

Une classe abstraite est une classe qui ne peut pas être instanciée directement. Elle sert de modèle pour définir des classes concrètes qui héritent de ses attributs et méthodes.

**Création d'une classe abstraite:**

Pour créer une classe abstraite en Python, il faut :

* Importer le module `abc`
* Hériter de la classe `ABC` du module `abc`
* Définir des méthodes abstraites (sans corps)

**Exemple de classe abstraite:**

```python
from abc import ABC, abstractmethod

class Shape(ABC):

  @abstractmethod
  def area(self):
    """Retourne l'aire de la forme."""
    pass

  @abstractmethod
  def perimeter(self):
    """Retourne le périmètre de la forme."""
    pass
```

**Méthodes abstraites:**

* Les méthodes abstraites n'ont pas de corps (pas d'implémentation).
* Elles définissent une signature que les classes filles doivent respecter.

**Classes concrètes:**

* Les classes concrètes héritent de la classe abstraite et implémentent les méthodes abstraites.

**Exemple de classe concrète:**

```python
from abc import ABC, abstractmethod

class Circle(Shape):

  def __init__(self, radius):
    self.radius = radius

  def area(self):
    return math.pi * self.radius ** 2

  def perimeter(self):
    return 2 * math.pi * self.radius
```

**Avantages des classes abstraites:**

* Permettent de définir des interfaces communes pour les classes concrètes.
* Favorisent la réutilisation du code et la cohérence entre les classes.
* Améliorent la lisbilité et la maintenabilité du code.

**Exemple d'utilisation:**

```python
from abc import ABC, abstractmethod

class Animal(ABC):

  @abstractmethod
  def speak(self):
    pass

class Dog(Animal):

  def speak(self):
    return "Woof!"

class Cat(Animal):

  def speak(self):
    return "Meow!"

def make_sound(animal):
  animal.speak()

chien = Dog()
chat = Cat()

make_sound(chien) # Affiche "Woof!"
make_sound(chat) # Affiche "Meow!"
```

## Polymorphisme en Python

**Exemple :**

```python
class Vehicule:
  def avance(self):
    print("Le véhicule démarre")

class Voiture(Vehicule):
  def avance(self):
    super().avance()
    print("La voiture roule")

class Avion(Vehicule):
  def avance(self):
    super().avance()
    print("L'avion vole")

v = Voiture()
a = Avion()

v.avance()
# Affiche :
# Le véhicule démarre
# La voiture roule

a.avance()
# Affiche :
# Le véhicule démarre
# L'avion vole
```

**Points clés :**

* Le polymorphisme permet d'utiliser une même fonction ou méthode avec des objets de types différents.
* La méthode `avance()` est définie dans la classe `Vehicule` et redéfinie dans les classes `Voiture` et `Avion`.
* La fonction `super().avance()` appelle la méthode `avance()` de la classe parente.
* L'appel à la méthode `avance()` sur les instances `v` et `a` déclenche l'exécution du code spécifique à chaque type d'objet.

## Méthodes de classe et méthodes statiques en Python

**Méthodes de classe:**

* Les méthodes de classe sont des méthodes qui s'utilisent sur la classe elle-même et non sur une instance de la classe.
* Elles sont définies avec le décorateur `@classmethod`.
* Elles peuvent retourner une instance de la classe.

**Exemple:**

```python
class Voiture:
  def __init__(self, marque, vitesse, prix):
    self.marque = marque
    self.vitesse = vitesse
    self.prix = prix

  @classmethod
  def lamborghini(cls):
    # Retourne une instance
    return cls(marque="Lamborghini", vitesse=250, prix=200000)

  @classmethod
  def porsche(cls):
    return cls(marque="Porsche", vitesse=200, prix=180000)

lambo = Voiture.lamborghini()
porsche = Voiture.porsche()

print(lambo.marque)  # Affiche "Lamborghini"
print(porsche.vitesse)  # Affiche 200
```

**Méthodes statiques:**

* Les méthodes statiques sont des méthodes qui n'ont pas besoin d'une instance de la classe pour être appelées.
* Elles sont définies avec le décorateur `@staticmethod`.
* Elles ne peuvent pas modifier les attributs de la classe.

**Exemple:**

```python
class Voiture:
  voiture_crees = 0
  def __init__(self, marque, vitesse, prix):
    Voiture.voiture_crees += 1
    self.marque = marque
    self.vitesse = vitesse
    self.prix = prix

  @classmethod
  def lamborghini(cls):
    return cls(marque="Lamborghini", vitesse=250, prix=200000)

  @classmethod
  def porsche(cls):
    return cls(marque="Porsche", vitesse=200, prix=180000)

  @staticmethod
  def afficher_nombre_voitures():
    print(f"Vous avez {Voiture.voiture_crees} voitures dans votre garage.")

lambo = Voiture.lamborghini()
porsche = Voiture.porsche()
Voiture.afficher_nombre_voitures()  # Affiche "Vous avez 2 voitures dans votre garage."
```

## Méthode __str__ en Python (équivalent à toString en Java)

**Exemple:**

```python
class Voiture:
  def __init__(self, marque, vitesse):
    self.marque = marque
    self.vitesse = vitesse

  def __str__(self):
    return f"Voiture de marque {self.marque} avec vitesse maximale de {self.vitesse}."

porsche = Voiture("Porsche", 200)
affichage = str(porsche)
print(affichage)
```

**Points clés:**

* La méthode `__str__` est une méthode spéciale qui est appelée lorsqu'on utilise la fonction `str()` sur une instance de la classe.
* La méthode `__str__` doit retourner une chaîne de caractères qui représente l'état de l'objet.

## BDD

### SQLite

**1. Importation de la bibliothèque SQLite**

```python
import sqlite3
```

**2. Établissement d'une connexion à la base de données**

```python
conn = sqlite3.connect("database.db")
```

- `sqlite3.connect()` crée une connexion à la base de données.
- Si la base de données "database.db" n'existe pas, elle sera créée.

**3. Création d'un curseur**

```python
c = conn.cursor()
```

- Le curseur permet d'exécuter des requêtes SQL et de récupérer les résultats.

**4. Création d'une table (si elle n'existe pas)**

```python
c.execute("""
CREATE TABLE IF NOT EXISTS employees(
  prenom text,
  nom text,
  salaire int
)
""")
```

- `CREATE TABLE` crée une table nommée "employees" avec trois colonnes : "prenom", "nom" et "salaire".
- `IF NOT EXISTS` évite la création de la table si elle existe déjà.

**5. Insertion de données dans la table**

```python
d = {"salaire":10000, "prenom":"Jean", "nom":"Dupond"}
c.execute("INSERT INTO employees VALUES (:prenom, :nom, :salaire)", d)
```

- `INSERT INTO` insère une nouvelle ligne dans la table "employees".
- Les valeurs à insérer sont stockées dans le dictionnaire `d`.
- Les placeholders `:prenom`, `:nom` et `:salaire` sont remplacés par les valeurs du dictionnaire.

**6. Sélection de données**

```python
d = {"a": "Paul"}
c.execute("SELECT * FROM employees WHERE prenom=:a", d)
premier = c.fetchone()
print(premier)  # Affiche le premier résultat
deuxieme = c.fetchone()
print(deuxieme)  # Affiche le deuxième résultat (None s'il n'y en a pas)

c.execute("SELECT * FROM employees WHERE prenom=:a", d)
donnees = c.fetchall()
print(donnees)  # Affiche tous les résultats
```

- `SELECT *` sélectionne toutes les colonnes de la table "employees".
- `WHERE prenom=:a` filtre les résultats en fonction de la valeur du prénom.
- `fetchone()` récupère un résultat à la fois.
- `fetchall()` récupère tous les résultats restants.

**7. Mise à jour de données**

```python
c.execute("""UPDATE employees SET salaire=:salaire WHERE prenom=:prenom AND nom=:nom""", d)
```

- `UPDATE` modifie des données existantes dans la table.
- `SET salaire=:salaire` met à jour la valeur du salaire.
- `WHERE prenom=:prenom AND nom=:nom` spécifie les lignes à mettre à jour.

**8. Suppression de données**

```python
c.execute("""DELETE FROM employees WHERE prenom='Jean'""")
```

- `DELETE` supprime des lignes de la table.

**9. Validation des modifications et fermeture de la connexion**

```python
conn.commit()  # Valide les modifications
conn.close()  # Ferme la connexion à la base de données
```

### TinyDB

**TinyDB est une bibliothèque Python légère et simple pour la gestion de bases de données NoSQL.**

**1. Installation et importation**

```python
# python -m pip install tinydb
from tinydb import Query, TinyDB, where
from tinydb.storages import MemoryStorage
```

**2. Création d'une base de données**

```python
# En mémoire
db = TinyDB(storage=MemoryStorage)

# Fichier JSON (optionnel)
db = TinyDB('data.json', indent=4)
```

**3. Insertion de données**

```python
# Exemples de dictionnaires (décommenter pour insérer)
dico1 = {"name": "Jean", "score": 1000}
dico2 = {"name": "Nico", "score": 2000}
dico3 = {"name": "Alexis", "score": 3000}

# Insertion d'un élément
# db.insert(dico1)

# Insertion de plusieurs éléments
# db.insert_multiple([dico2, dico3])
```

**4. Sélection de données**

```python
# Classe Query pour construire des requêtes
User = Query()

# Rechercher tous les éléments où "name" est "Alexis"
alexis = db.search(User.name == "Alexis")

# Récupérer un unique élément
alexis_unique = db.get(User.name == "Alexis")

# Rechercher tous les éléments où "score" est supérieur à 0
scores = db.search(where("score") > 0)

# Afficher les résultats
print(alexis)
print(scores)
```

**5. Mise à jour de données**

```python
# Mettre à jour tous les éléments en ajoutant un champ "roles" avec la valeur "Junior"
db.update({"roles": ["Junior"]})

# Mettre à jour uniquement l'élément où "name" est "Alexis" et changer "roles" à "Senior"
db.update({"roles": ["Senior"]}, where('name') == 'Alexis')
```

**6. Insertion ou mise à jour**

```python
# Si l'élément "Pierre" n'existe pas, l'insérer. Sinon, le mettre à jour.
db.upsert({"name": "Pierre", "score": 0, "roles": ["Senior"]}, where('name') == 'Pierre')
```

**7. Suppression de données**

```python
# Supprimer tous les éléments où "score" est égal à 0
db.remove(where('score') == 0)
```

**8. Vider la base de données**

```python
# Supprime tous les éléments de la base de données
db.truncate()
```

**9. Plusieurs tables (pas de jointure)**

TinyDB ne supporte pas les jointures entre tables, mais vous pouvez créer plusieurs tables distinctes pour organiser vos données.

```python
# Créer des tables nommées "Users" et "Roles"
users = db.table("Users")
roles = db.table("Roles")

# Insérer des éléments vides dans les tables (exemple)
users.insert({})
roles.insert({})
```

**Points à retenir:**

* TinyDB est une bibliothèque simple et pratique pour des projets légers.
* Elle ne supporte pas les jointures complexes.
* Elle est idéale pour stocker des données en mémoire ou en JSON.


### Typer

**Installation:**

```bash
python -m pip install typer
```

**Description:**

Typer est une bibliothèque Python permettant de créer des interfaces de ligne de commande (CLI).

**PySide2**

**Installation:**

```bash
python -m pip install PySide2
```

**Description:**

PySide2 est un framework Python permettant de développer des interfaces graphiques riches et multi-plateformes. Il s'agit d'une alternative à PyQt et offre des fonctionnalités similaires, telles que :

* Création de fenêtres, de widgets et de mises en page
* Gestion des événements et des interactions utilisateurs
* Support de divers styles et thèmes

**CurrencyConverter**

**Installation:**

```bash
python -m pip install currencyconverter
```

**Description:**

CurrencyConverter est une bibliothèque Python permettant de convertir des montants entre différentes devises.

**Interfaces Graphiques (GUI)**

L'installation de PySide2 permet de créer des interfaces graphiques (GUI) en Python.

## Python to EXE

Le processus de conversion d'un script Python en un fichier exécutable (.exe) permet de le distribuer plus facilement sur des systèmes Windows. La commande fournie utilise l'outil `auto-py-to-exe`. Voici un résumé des étapes :

1. **Installation:**
    ```bash
    pip3.9 install auto-py-to-exe
    ```
2. **Exécution de `auto-py-to-exe`:**
    ```bash
    auto-py-to-exe
    ```
3. **Instructions pas à pas:**
    * Sélectionnez le **fichier principal** de votre application.
    * Choisissez la version **Windows Based**.
    * Optez pour l'option **One Directory** (One File peut poser des problèmes).
    * Dans **Additional Files**, ajoutez le dossier **data** de votre projet.
    * Lancez la conversion en cliquant sur **Convert**.

## Tests Unitaires

**Structure des tests:**

* Créez un dossier `tests` à la racine de votre projet.
* Nommez vos fichiers de test en commençant par `test_`.

**Frameworks de test:**

* **unittest:** Bibliothèque standard de Python, simple et efficace.
* **doctest:** Permet de tester des fonctions et des classes à partir de leur documentation.
* **pytest:** Framework de test moderne et populaire, compatible avec unittest.

**Couverture de test:**

* **pytest:**
    * Installation: `pip install pytest`
    * Lancement: `python -m pytest .\test.py --verbose`
    * Rapport HTML:
        * Installation: `pip install pytest-html`
        * Lancement: `python -m pytest .\test.py -v --html=index.html`
* **coverage:**
    * Installation: `pip install coverage`
    * Lancement: `python -m coverage run -m unittest .\test_calc.py`

**Exemple avec unittest:**

```python
# test_somme.py

import unittest

class TestSomme(unittest.TestCase):

    def test_somme_deux_nombres(self):
        self.assertEqual(somme(1, 2), 3)

    def test_somme_liste_nombres(self):
        self.assertEqual(somme([1, 2, 3]), 6)

if __name__ == '__main__':
    unittest.main()
```

**Exemple avec doctest:**

```python
def somme(a, b):
    """
    Fonction qui additionne deux nombres.

    >>> somme(1, 2)
    3
    >>> somme([1, 2, 3])
    6
    """
    return a + b

print(somme(1, 2))
```

## Django : Guide de démarrage rapide

**Installation:**

1. **Installer Django:**
    ```bash
    pip install django==3.1.6
    ```
2. **Vérifier l'installation:**
    ```bash
    ./.env/Scripts/django-admin --help  # Remplacez ".env/Scripts" par le chemin correct sur votre système
    ```

**Création d'un projet:**

1. Créez un répertoire vide pour votre projet.
2. Ouvrez un terminal dans ce répertoire et lancez la commande suivante :
    ```bash
    ./.env/Scripts/django-admin startproject DocBlog  # Remplacez "DocBlog" par le nom souhaité pour votre projet
    ```

**Démarrage du serveur de développement:**

1. Accédez au répertoire de votre projet :
    ```bash
    cd DocBlog
    ```
2. Lancez le serveur de développement :
    ```bash
    python manage.py runserver
    ```
    Ouvrez ensuite votre navigateur web à l'adresse http://127.0.0.1:8000/ pour voir la page d'accueil par défaut de Django.

**Migrations de base de données:**

1. Django utilise des migrations pour gérer les modifications apportées au schéma de votre base de données.
2. Créez les tables initiales de la base de données en exécutant la commande suivante :
    ```bash
    python manage.py migrate
    ```

**Intégration de Chart.js:**

1. Chart.js est une bibliothèque JavaScript populaire pour créer des graphiques interactifs.
2. Vous pouvez intégrer Chart.js dans vos templates Django pour afficher des visualisations de données.
3. Référez-vous à la documentation de Chart.js ([https://www.chartjs.org/docs/latest/getting-started/](https://www.chartjs.org/docs/latest/getting-started/)) pour apprendre à créer différents types de graphiques et les intégrer dans vos pages.

**Hébergement gratuit Python:**

1. Plusieurs options d'hébergement gratuit existent pour déployer votre site web Django :
    * **PythonAnywhere:** [https://www.pythonanywhere.com/](https://www.pythonanywhere.com/)
    * **Heroku:** [https://www.heroku.com/students](https://www.heroku.com/students)
    * **Digital Ocean:** [https://www.digitalocean.com/](https://www.digitalocean.com/)
2. Chaque plateforme possède sa documentation et ses instructions spécifiques pour déployer une application Django.

**Ressources supplémentaires:**

* Documentation officielle de Django : [https://docs.djangoproject.com/en/5.0/](https://docs.djangoproject.com/en/5.0/)

