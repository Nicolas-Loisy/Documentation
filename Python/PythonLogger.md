# Logger

## Flask

L'initialisation de Flask donne un app qui instancie un logger python (librairie standard logging), flask vient juste determiner 

- `app.logger` : L'initialisation de Flask donne un app qui instancie un logger python (librairie standard logging).
- Il est pré-configuré pour enregistrer les messages selon le niveau défini (par défaut DEBUG en mode debug=True et WARNING en mode production).


Il est possible de configurer le logger avec une config `logging.ini` :

``` Python
import logging.config
from flask import Flask

def create_app():
    # Charger la configuration de logging
    logging.config.fileConfig('logging.ini')

    # Créer l'application Flask
    app = Flask(__name__)

    # Utiliser le logger ( intégré de Flask
    logger = app.logger
    logger.debug("Logger de Flask initialisé avec succès.")

```