# Python - Avancé

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

### self vs cls

- Utilisez toujours self comme premier argument des méthodes d'instance.

- Utilisez toujours cls comme premier argument des méthodes de classe.

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
* Améliorent la lisibilité et la maintenabilité du code.

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
