# Piper

Pour configurer Piper comme assistant vocal sur votre Raspberry Pi 5 avec Docker et Portainer, voici les étapes à suivre :

### Étape 1 : Préparer Portainer

1. **Accédez à Portainer** : Ouvrez votre navigateur et accédez à l'interface Portainer.

2. **Créer un nouveau Stack** :
   - Allez dans l'onglet "Stacks" sur le côté gauche.
   - Cliquez sur "Add stack".

### Étape 2 : Configurer le Stack pour Piper

1. **Nom du Stack** : Donnez un nom à votre stack, par exemple `piper-stack`.

2. **Configuration du Stack** :
   - Dans le champ "Web editor", collez la configuration suivante :

     ```yaml
     version: "2.1"
     services:
       piper:
         image: lscr.io/linuxserver/piper:latest
         container_name: piper
         environment:
           - PUID=1000
           - PGID=1000
           - TZ=Europe/Paris
           - PIPER_VOICE=fr_FR-medium
           - PIPER_LENGTH=1.0 #optional
           - PIPER_NOISE=0.667 #optional
           - PIPER_NOISEW=0.333 #optional
           - PIPER_SPEAKER=0 #optional
           - PIPER_PROCS=1 #optional
         volumes:
           - /home/pi5-markio/piper_models:/config
         ports:
           - 10200:10200
         restart: unless-stopped
     ```

   - Remplacez `/home/pi5-markio/piper_models` par le chemin réel sur votre Raspberry Pi où vous souhaitez stocker les fichiers de configuration de Piper.

3. **Déployer le Stack** :
   - Cliquez sur "Deploy the stack".

### Étape 3 : Configurer Home Assistant

1. **Accédez à Home Assistant** : Ouvrez l'interface de Home Assistant.

2. **Ajouter l'intégration Wyoming** :
   - Allez dans "Configuration" > "Intégrations".
   - Cliquez sur "Ajouter une intégration".
   - Recherchez "Wyoming" et suivez les instructions pour ajouter l'intégration.
   - Fournissez l'adresse IP et le port (10200) de votre Raspberry Pi où Piper est en cours d'exécution.

### Étape 4 : Vérifier le Fonctionnement

1. **Vérifiez les Logs** :
   - Dans Portainer, allez dans "Containers" et trouvez le conteneur `piper`.
   - Vérifiez les logs pour vous assurer qu'il n'y a pas d'erreurs.

2. **Testez l'Assistant Vocal** :
   - Dans Home Assistant, configurez une automatisation ou une commande vocale pour tester Piper.

### Remarques

- Assurez-vous que les ports ne sont pas bloqués par un pare-feu.
- Vous pouvez ajuster les paramètres de voix (`PIPER_VOICE`, `PIPER_LENGTH`, etc.) selon vos préférences.

### TTS GlaDos Voice

```shell
wget https://raw.githubusercontent.com/TazzerMAN/piper-voice-glados-fr/main/models/fr_FR-glados-medium.tar.gz -O /home/pi5-markio/piper_models/fr_FR-glados-medium.tar.gz
```
```shell
tar -xzvf /home/pi5-markio/piper_models/fr_FR-glados-medium.tar.gz -C /home/pi5-markio/piper_models/
```
Pour les modèles customs, il faut les nommer avec juste le nom de la voix : 
- ~~fr_FR-glados-medium.onnx~~ -> glados.onnx
- ~~fr_FR-glados-medium.onnx.json~~ -> glados.onnx.json