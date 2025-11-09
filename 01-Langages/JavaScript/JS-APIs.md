# JavaScript - APIs et Stockage

## API

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

## Fetch

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

## Axios

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

## Cookies

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

## LocalStorage / SessionStorage

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

## Géolocalisation

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
