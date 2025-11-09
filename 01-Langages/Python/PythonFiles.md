# Python - Fichiers et Données

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
