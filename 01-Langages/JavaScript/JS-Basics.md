# JavaScript - Bases

## Variables
- **`var`** : Variable globale.
- **`let`** : Variable locale (valide dans la fonction ou le bloc).
- **`const`** : Variable constante (non modifiable).

## Fonctions JS
- `prompt()`, `confirm()`, `alert()`
- `finally` : Exécute toujours un bloc de code après un `try/catch`.
- `isNaN(variable)` : Vérifie si la variable n'est pas un nombre.

## Tableaux
- **Tableau 1D** :
  ```js
  let tableau = ['a', 'b', 'c'];
  tableau.indexOf('b');
  tableau.splice(1, 0, "yo");
  ```

- **Parcourir un tableau** :
  - Avec `for` :
    ```js
    for (const fruit of panier) {
      console.log(fruit);
    }
    ```

  - Avec fonction fléchée :
    ```js
    listeDePays.forEach(pays => console.log(pays));
    ```

- **Décomposition d'un tableau** :
  ```js
  let [pseudo, age, sexe] = ['superSayen', '25', 'homme'];
  ```

- **Tableau 2D** :
  ```js
  let tab2D = [['a', 'b'], ['c', 'd']];
  tab2D.push(['e', 'f']);
  ```

- **Tableaux associatifs** :
  ```js
  let utilisateur = { prenom: 'Mark', nom: 'Zuckerberg' };
  utilisateur['pays'] = 'USA';
  delete utilisateur.nom;
  ```

## Fonctions et Closures
- **Exemple de closure** :
  ```js
  function timer() {
    let secondes = 0;
    return () => ++secondes;
  }
  let monTimer = timer();
  ```

## Objets JS
- **Objet littéral** :
  ```js
  let chien = {
    race: 'Shiba',
    aboyer() { console.log('ouaf'); }
  };
  chien.aboyer();
  ```

- **`Set` (Collection de valeurs uniques)** :
  ```js
  let monSet = new Set([1, 2, 3]);
  monSet.add(4);
  monSet.delete(2);
  ```

- **`Map` (Collection de paires clé-valeur)** :
  ```js
  let monMap = new Map();
  monMap.set('clé', 'valeur');
  console.log(monMap.get('clé'));
  ```

## Constructeurs et Héritage
- **Constructeur d'objets** :
  ```js
  function Utilisateur(prenom, nom) {
    this.prenom = prenom;
    this.nom = nom;
  }
  let utilisateur1 = new Utilisateur('John', 'Doe');
  ```

- **Héritage avec `call()` et `apply()`** :
  ```js
  function Animal(pattes) {
    this.pattes = pattes;
  }
  function Chien() {
    Animal.call(this, 4);  // Héritage
  }
  ```

## Classes JS
- **Classes avec héritage** :
  ```js
  class Animal {
    constructor(pattes) {
      this.pattes = pattes;
    }
    parler() {
      console.log('Je fais du bruit');
    }
  }

  class Chien extends Animal {
    parler() {
      console.log('Je aboie');
    }
  }
  ```

## Getter et Setter
- **Getter et Setter** :
  ```js
  class Utilisateur {
    constructor(prenom, nom) {
      this.prenom = prenom;
      this.nom = nom;
    }

    get nomComplet() {
      return this.prenom + ' ' + this.nom;
    }

    set nomComplet(valeur) {
      [this.prenom, this.nom] = valeur.split(' ');
    }
  }
  ```

## Mémo Objets et Tableaux
- **Manipulations courantes** : `push()`, `pop()`, `shift()`, `unshift()`, `splice()`, `slice()`, `indexOf()`, `join()`
