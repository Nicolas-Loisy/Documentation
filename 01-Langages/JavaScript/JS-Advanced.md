# JavaScript - Avancé

## Dates

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

## Templates String

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

## Asynchronisme

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
