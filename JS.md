# Dico JS - Guide de Survie pour Débutants

## Sélectionner des éléments DOM
- **Sélectionner par balise** :  
  `document.getElementsByTagName('header');`
- **Sélectionner par ID** :  
  `document.getElementById('logo');`
- **Sélectionner par classe** :  
  `document.getElementsByClassName('container');`

- **Sélectionner avec `querySelector`** (le premier élément correspondant) :
  - `document.querySelector('h1');`  — sélectionne une balise `h1`
  - `document.querySelector('#id');` — sélectionne un élément par ID
  - `document.querySelector('.class');` — sélectionne un élément par classe

- **Sélectionner plusieurs éléments** avec `querySelectorAll` :
  `document.querySelectorAll('p');` — sélectionne tous les éléments `p`

## Manipulation DOM
- **Modifier le contenu texte** :
  `header.textContent = "le message";`
  
- **Modifier le HTML** :
  `header.innerHTML = "<div> Hello world!</div>";`

- **Créer un élément** :
  `let newDiv = document.createElement('div');`

- **Ajouter un élément** :
  - `h1.append('test');` — ajoute à la fin de la balise
  - `document.querySelector('.container').prepend(newDiv);` — ajoute avant un élément

- **Supprimer un élément** :
  `element.remove();`

## BOM vs DOM
- **BOM (Browser Object Model)** : Tous les objets utilisés par le navigateur.
- **DOM (Document Object Model)** : Représentation de la structure HTML de la page.

### Accéder aux éléments DOM
- `getElementsByTagName()`, `getElementById()`, `getElementsByClassName()`, `querySelector()`, `querySelectorAll()`

### Modifier le DOM
- `textContent` : Modifie le texte
- `innerHTML` : Modifie le HTML d'un élément

### Ajouter / Supprimer des éléments
- `createElement()`, `prepend()`, `append()`, `appendChild()`, `insertBefore()`, `remove()`

### Modifier le style
- `style.propriété` : `style.color = "red";`
- `className` : Modifie les classes CSS

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
- **Example de closure** :
  ```js
  function timer() {
    let secondes = 0;
    return () => ++secondes;
  }
  let monTimer = timer();
  ```

## Événements
- **Écouteurs d'événements classiques** : `onfocus`, `onclick`, `onchange`, `ondblclick`, etc.
- **`addEventListener()`** : Permet de gérer des événements comme `click`, `mouseover`, `paste`, etc.
  ```js
  element.addEventListener('click', (e) => {
    alert('Clic détecté');
    e.stopPropagation();
  });
  ```

## Timeouts et Intervalles
- **setTimeout** : Exécute une fonction après un délai.
  ```js
  let timer = setTimeout(() => alert('Bonjour'), 3000);
  clearTimeout(timer);  // Annule le timer
  ```

- **setInterval** : Exécute une fonction à intervalles réguliers.
  ```js
  let interval = setInterval(() => alert('Bonjour'), 5000);
  clearInterval(interval);  // Annule l'intervalle
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


Voici la version détaillée de toutes les fonctionnalités que tu souhaites ajouter dans ta documentation :

## **Dates**

#### Exemple d'utilisation :
```javascript
// Récupération de la date actuelle avec la norme anglo-saxonne
let dateActuelle = Date();
console.log(dateActuelle);

// Récupérer la date en secondes (depuis le 1er janvier 1970)
let dateEnSecondes = Date.now();
console.log(dateEnSecondes);

// Créer une nouvelle instance de la date actuelle
let dateActuelle = new Date();
let dateEnSecondes = new Date(Date.now());
// Créer une date précise (année, mois, jour, heure, minute, seconde, milliseconde)
let datePrecise = new Date(2019, 11, 09, 22, 25);
console.log(dateActuelle);
console.log(dateEnSecondes);
console.log(datePrecise);
```

#### **Date Getter et Setter** :
```javascript
let dateActuelle = new Date();
console.log(dateActuelle.getDay()); // jour de la semaine (0 pour dimanche, 6 pour samedi)
console.log(dateActuelle.getFullYear()); // année
console.log(dateActuelle.getDate()); // jour du mois
console.log(dateActuelle.getUTCDay()); // jour de la semaine en UTC
dateActuelle.setFullYear(2750); // Modification de l'année
console.log(dateActuelle.getFullYear());
```

#### **Date Locale** :
```javascript
let dateActuelle = new Date();

// Affichage de la date en fonction des options locales
let dateLocale = dateActuelle.toLocaleString(navigator.language, {
  weekday: 'long', // court, étroit, long
  year: 'numeric', // numérique
  month: 'long', // court, étroit, long, numérique
  day: 'numeric', // numérique, 2 chiffres
  hour: 'numeric',
  minute: 'numeric',
  second: 'numeric'
});

console.log(dateLocale);
```

---

## **Templates String**

Les template strings permettent d'insérer des variables ou expressions dans des chaînes de caractères à l'aide de backticks (`` ` ``).

#### Exemple avant :
```javascript
let prenom = "John";
let bonjour = "Bonjour " + prenom;
console.log(bonjour);
```

#### Exemple après avec template string :
```javascript
let prenom = "John";
let bonjour = `Bonjour ${prenom}`;
console.log(bonjour);
```

#### Exemple avec les dates :
```javascript
let date = new Date().getFullYear();
let copyright = `${date} © Moi`;
console.log(copyright);
```

#### Exemple avec des calculs :
```javascript
let aliments = { fruits: 5, legumes: 1, biscuits: 75 };
let panier = `Dans votre panier, vous avez ${aliments.fruits + aliments.legumes + aliments.biscuits} articles.`;
console.log(panier);
```

---

## **API**

#### Exemple d'API pour récupérer le prix du Bitcoin (via `XMLHttpRequest`) :
```javascript
const url = 'https://blockchain.info/ticker';

function recupererPrix() {
  let requete = new XMLHttpRequest();
  requete.open('GET', url);
  requete.responseType = 'json';
  requete.send();

  requete.onload = function() {
    if (requete.readyState === XMLHttpRequest.DONE) {
      if (requete.status === 200) {
        let reponse = requete.response;
        let prixEnEuros = reponse.EUR.last;
        console.log(prixEnEuros);
      }
      else {
        alert('Un problème est intervenu, merci de revenir plus tard.');
      }
    }
  }
  console.log('Prix actualisé');
}

setInterval(recupererPrix, 1000);
```

#### Exemple d'API POST :
```javascript
const url = 'https://lesoublisdelinfo.com/api.php';

let requete = new XMLHttpRequest();
requete.open('POST', url);
requete.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
requete.responseType = 'json';
requete.send('prenom=John');

requete.onload = function() {
  if (requete.readyState === XMLHttpRequest.DONE) {
    if (requete.status === 200) {
      let reponse = requete.response;
      console.log(reponse);
    }
    else {
      alert('Un problème est intervenu, merci de revenir plus tard.');
    }
  }
}
```

---

## **Fetch**

#### Exemple de récupération de données avec `fetch()` :
```javascript
const url = 'https://blockchain.info/ticker';

async function recupererPrix() {
  const requete = await fetch(url, { method: 'GET' });
  
  if(!requete.ok) {
    alert('Un problème est survenu.');
  } else {
    let donnees = await requete.json();
    document.querySelector('span').textContent = donnees.EUR.last;
  }
}

setInterval(recupererPrix, 1000);
```

#### Exemple POST avec `fetch()` :
```javascript
const url = 'https://lesoublisdelinfo.com/api.php';

async function envoyerPrenom(prenom) {
  const requete = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ prenom })
  });

  if(!requete.ok) {
    alert('Un problème est survenu.');
  } else {
    const donnees = await requete.json();
    console.log(donnees);
  }
}

envoyerPrenom('Elon');
```

---

## **Axios**

#### Exemple de récupération avec `Axios` :
```javascript
const url = 'https://blockchain.info/ticker';

async function recupererPrix() {
  axios.get(url)
    .then(function(donnees) {
      document.querySelector('span').textContent = donnees.data.EUR.last;
    })
    .catch(function(erreur) {
      alert('Un problème est survenu');
    })
    .then(function () {
      console.log('mise à jour effectuée');
    });
}

setInterval(recupererPrix, 1000);
```

#### Exemple POST avec `Axios` :
```javascript
const url = 'https://lesoublisdelinfo.com/api.php';

axios.post(url, new URLSearchParams({ prenom: 'Steve' }))
  .then(function(donnees) {
    console.log(donnees.data);
  })
  .catch(function(erreur) {
    console.log(erreur);
  });
```

---

## **Asynchronisme**

#### Promesses :
```javascript
// Exemple avec promesses
function chargerScript(script) {
  return new Promise((resolve, reject) => {
    let element = document.createElement('script');
    element.src = script;
    document.head.append(element);
    element.onload = () => resolve('Fichier ' + script + ' chargé');
    element.onerror = () => reject(new Error('Operation impossible pour le script ' + script));
  });
}

chargerScript('test.js')
  .then(result => console.log(result))
  .catch(error => console.log(error));
```

#### Async et Await :
```javascript
async function chargerScript(script) {
  try {
    const scriptA = await chargerScript('test.js');
    console.log(scriptA);
    const scriptB = await chargerScript('autre.js');
    console.log(scriptB);
  } catch (error) {
    console.log(error);
  }
}

chargerScript();
```

---

## **Cookies**

#### Exemple de création et gestion des cookies :
```javascript
// Créer un cookie
document.cookie = 'prenom=John';

// Modifier un cookie
document.cookie = 'prenom=Mark';

// Supprimer un cookie
document.cookie = 'prenom=';

// Créer un cookie avec expiration
let date = new Date(Date.now() + 31536000000);
date = date.toUTCString();
document.cookie = 'prenom=John; expires=' + date;

// Utilisation de max-age
document.cookie = 'prenom=John; max-age=31536000000';
```

---

## **LocalStorage / SessionStorage**

#### Exemple d'utilisation :
```javascript
// LocalStorage
if(localStorage.getItem('prenom')) {
  document.body.append('Bonjour ' + localStorage.getItem('prenom'));
} else {
  let prenom = prompt('Quel est votre prénom ?');
  localStorage.setItem('prenom', prenom);
  document.body.append('Bonjour ' + prenom);
}
```

---

## **jQuery**

#### Exemple de modification de contenu :
```javascript
// Avec JavaScript
document.querySelector('h1').textContent = 'Bonjour avec JavaScript';

// Avec jQuery
$('h1').text('Bonjour avec jQuery');
```

#### Exemple de modification de style :
```javascript
// Avec JavaScript
document.querySelector('h1').style.color = 'orange';

// Avec jQuery
$('h1').css('color', 'orange');
```

---

## **Géolocalisation**

#### Exemple de géolocalisation avec `navigator.geolocation` :
```javascript
if ('geolocation' in navigator) {
  navigator.geolocation.getCurrentPosition((position) => {
    console.log(position.coords.latitude);
    console.log(position.coords.longitude);
  }, error, options);
} else {
  alert('Le navigateur ne supporte pas la géolocalisation');
}
```

---

## **Modules (Import/Export)**

#### Importation d'un module :
```javascript
import { addition } from './math.js';
console.log(addition(5, 3));
```

#### Exportation d'un module :
```javascript
export function addition(a, b) {
  return a + b;
}
```

---

## **WebSockets**

#### Exemple de connexion à un WebSocket :
```javascript
let socket = new WebSocket('ws://exemple.com/socket');

socket.onopen = function(event) {
  console.log('Connecté au serveur');
  socket.send('Message initial');
};

socket.onmessage = function(event) {
  console.log('Message du serveur:', event.data);
};
```
