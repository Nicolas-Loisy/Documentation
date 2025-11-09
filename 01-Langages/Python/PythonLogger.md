# Logger

## Introduction aux Logs en Python

Le module `logging` de Python permet de générer des logs.

### Niveaux de Logs

Les logs sont classés par niveaux en fonction de leur importance :
- **DEBUG** : Informations détaillées pour le débogage.
- **INFO** : Informations générales sur le fonctionnement du programme.
- **WARNING** : Indications de problèmes.
- **ERROR** : Erreurs empêchant une partie du programme de fonctionner.
- **CRITICAL** : Erreurs graves.

### Exemple simple de Logging

```python
import logging

# Configurer le niveau de log
logging.basicConfig(level=logging.DEBUG)

# Générer des messages de log
logging.debug("Message de debug")
logging.info("Message d'information")
logging.warning("Message d'avertissement")
logging.error("Message d'erreur")
logging.critical("Message critique")
```

### Configuration des Logs avec un Fichier `.ini`

Le fichier `logging.ini` permet de centraliser la configuration des logs.

Les points importants :
1. **Loggers** : Identifient les sources de messages (ex. `root`, `flask.app`, etc.).
    - Chaque logger est nommé avec un `qualname`.
    - Le `qualname` correspond généralement au nom du module ou du projet.
2. **Handlers** : Définissent où les logs sont envoyés (console, fichier, etc.).
3. **Formatters** : Contrôlent le format des messages (date, niveau, nom du logger, etc.).

#### Structure d'un fichier `logging.ini`

```ini
[loggers]
keys=root,exampleLogger

[handlers]
keys=consoleHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=WARNING
handlers=consoleHandler

[logger_exampleLogger]
level=DEBUG
handlers=consoleHandler
qualname=exampleLogger

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=simpleFormatter
args=(sys.stdout,)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=%Y-%m-%d %H:%M:%S
```

Ce fichier configure un logger nommé `exampleLogger` pour afficher des messages de niveau `DEBUG` ou supérieur, avec un format clair, vers la console.

---
## Logger avec Flask

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


## Note importante :

Même si un logger (ex : `app_logger`) est configuré sur `DEBUG` dans le `.ini`, les messages `DEBUG` ne s'afficheront pas si ses handlers ont un niveau plus élevé (ex. `INFO`). Assurez-vous que **les niveaux du logger et des handlers soient alignés** pour afficher tous les messages souhaités.
