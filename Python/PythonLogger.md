# Logger

## Flask

- **`app.logger`** : Flask fournit un logger intégré basé sur la librairie standard `logging` de Python.  
- Par défaut, le niveau de log est `DEBUG` en mode développement (`debug=True`) et `WARNING` en mode production.  


### Configuration avec un fichier `logging.ini`

Vous pouvez personnaliser le logger Flask en utilisant un fichier `logging.ini` pour centraliser la gestion des logs.

#### Exemple : Configuration de Flask avec `logging.ini`

```python
import logging.config
from flask import Flask

def create_app():
    # Charger la configuration de logging
    logging.config.fileConfig('logging.ini')

    # Créer l'application Flask
    app = Flask(__name__)

    # Utiliser le logger intégré de Flask
    logger = app.logger
    logger.debug("Logger de Flask initialisé avec succès.")

    return app
```

---

## Bibliothèque avec Logging

Si vous développez une bibliothèque utilisée dans une application Flask, vous pouvez utiliser `logging.config.fileConfig` pour assurer que la bibliothèque partage la même configuration de logging.

### Exemple : Configuration dans la bibliothèque

```python
import logging.config

logging_ini_path='logging.ini'

# Charger la configuration de logging depuis le fichier
logging.config.fileConfig(logging_ini_path)
logger = logging.getLogger('bibliotheque')
logger.debug("Logger de la bibliothèque initialisé avec succès.")
```

Cela garantit que les messages de la bibliothèque suivent les mêmes règles que ceux de Flask.

---

## Exemple de `logging.ini` pour Flask et la Bibliothèque

Voici un exemple de fichier `logging.ini` pour configurer les loggers de Flask et de votre bibliothèque de manière unifiée :

```ini
[loggers]
keys=root,flask,bibliotheque

[handlers]
keys=consoleHandler,fileHandler

[formatters]
keys=simpleFormatter,detailedFormatter

[logger_root]
level=WARNING
handlers=consoleHandler

[logger_flask]
level=DEBUG
handlers=consoleHandler,fileHandler
qualname=flask.app
propagate=0

[logger_bibliotheque]
level=DEBUG
handlers=consoleHandler,fileHandler
qualname=bibliotheque
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=DEBUG
formatter=detailedFormatter
args=('application.log', 'a')

[formatter_simpleFormatter]
format=%(levelname)s - %(message)s

[formatter_detailedFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=%Y-%m-%d %H:%M:%S
```

## Bonnes pratiques

1. **Chargement de la configuration** : Chargez `logging.ini` avant toute utilisation des loggers pour garantir une configuration cohérente.
2. **Utilisation de noms explicites** : Identifiez chaque logger avec un `qualname` distinct (`flask.app`, `bibliotheque`).
