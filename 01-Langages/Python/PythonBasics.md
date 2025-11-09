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
* `.capitalize()`: Met la première lettre en majuscule et le reste en minuscules
* `.title()`: Met la première lettre de chaque mot en majuscule
* `"Le jour bonjour".count("jour")`: Compte le nombre d'occurrences de "jour"
* `"Le jour bonjour".index("jour")`: Renvoie l'index de la première occurrence de "jour"
* `"Le jour bonjour".find("jour")`: Idem que `index()` mais renvoie -1 si non trouvé
* `"Le jour bonjour".rfind("jour")`: Renvoie l'index de la dernière occurrence de "jour"
* `"Le jour bonjour".lfind("jour")`: Renvoie l'index de la première occurrence de "jour" en partant de la gauche
* `"bonjour".replace("jour", "soir")`: Remplace "jour" par "soir"
* `"image.png".endswith(".png") => True`: Vérifie si la chaîne se termine par ".png"
* `"image.png".startswith(".png") => False`: Vérifie si la chaîne commence par ".png"
* `" bon jour ".strip() => "bon jour"`: Supprime les espaces blancs en début et fin de chaîne
* `" bonjour ".strip(" ujor") => "bon"`: Supprime les caractères "u", "j", "o" et "r" en début et fin de chaîne
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
* `liste.index("valeur")`: Récupérer l'index de la première occurrence de la valeur. **Exemple:** `liste.index("a")`.
* `liste.count("valeur")`: Récupérer le nombre d'occurrences de la valeur. **Exemple:** `liste.count("a")`.
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
