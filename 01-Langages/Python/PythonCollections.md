# Python - Collections

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
