# JavaScript - Modules et Outils

## jQuery

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

## Modules (Import/Export)

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

## WebSockets

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
