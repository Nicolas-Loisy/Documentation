# JavaScript - DOM et Événements

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
