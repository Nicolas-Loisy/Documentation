Voici un guide détaillé, étape par étape, pour configurer votre Raspberry Pi 5 avec tous les outils et projets nécessaires.

### Étape 1 : Préparation du Raspberry Pi

1. **Installation du Système d'Exploitation** :
   - Téléchargez **Raspberry Pi OS Lite** depuis le site officiel de Raspberry Pi.
   - Utilisez **Raspberry Pi Imager** pour flasher l'image sur une carte microSD.
   - Insérez la carte microSD dans votre Raspberry Pi et démarrez-le.

2. **Configuration Initiale** :
   - Connectez-vous au Raspberry Pi via SSH. Vous pouvez utiliser un outil comme PuTTY sur Windows ou le terminal sur macOS/Linux.
     ```bash
     ssh pi@<IP_DE_VOTRE_PI>
     ```
   - Mettez à jour le système :
     ```bash
     sudo apt update
     sudo apt upgrade
     ```
3. **Optimisation du Pi 5** :
    Voici comment activer ces optimisations sur votre Raspberry Pi 5 :
    
    Mettez à jour le firmware :
    ```bash
    sudo rpi-update
    ```
    Éditez la configuration du bootloader :
    
    ```bash
    sudo rpi-eeprom-config -e
    ```

    Puis ajoutez la ligne suivante pour le Pi 5 :
    ```bash    
    SDRAM_BANKLOW=1
    ```

    Enfin, redémarrez votre système pour appliquer les changements.

### Étape 2 : Installation de Docker

1. **Installer Docker** :
   - Téléchargez et exécutez le script d'installation de Docker :
     ```bash
     curl -fsSL https://get.docker.com -o get-docker.sh
     sudo sh get-docker.sh
     ```

2. **Ajouter votre utilisateur au groupe Docker** :
   - Cela permet d'exécuter des commandes Docker sans `sudo`.
     ```bash
     sudo usermod -aG docker $USER
     ```
   - Redémarrez votre session ou le Raspberry Pi pour appliquer les changements.

### Étape 3 : Installation de Portainer

1. **Créer un Volume Docker pour Portainer** :
   - Exécutez la commande suivante pour créer un volume :
     ```bash
     docker volume create portainer_data
     ```

2. **Lancer Portainer** :
   - Utilisez la commande suivante pour démarrer Portainer :
     ```bash
     docker run -d -p 9000:9000 --name=portainer --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer-ce
     ```
    - Par mesure de sécurité, il faut :
    ```bash
    docker stop portainer
    docker start portainer
    ```

3. **Accéder à Portainer** :
   - Ouvrez votre navigateur et accédez à `http://<IP_DE_VOTRE_PI>:9000`.
   - Suivez les instructions pour configurer Portainer.


### Étape 4 : Configuration de Tailscale

Tailscale peut fonctionner sur les cartes Raspberry Pi exécutant Raspbian. Les packages sont disponibles en versions 32 et 64 bits.

1. **Installer le `apt-transport-https` plugin :**
```bash
sudo apt-get install apt-transport-https
```

2. **Ajoutez la clé de signature du package et le référentiel de Tailscale :**
```bash
curl -fsSL https://pkgs.tailscale.com/stable/raspbian/bullseye.noarmor.gpg | sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg > /dev/null
curl -fsSL https://pkgs.tailscale.com/stable/raspbian/bullseye.tailscale-keyring.list | sudo tee /etc/apt/sources.list.d/tailscale.list
```

3. **Installer Tailscale :**
```bash
sudo apt-get update
sudo apt-get install tailscale
```

4. **Connectez votre machine à votre réseau Tailscale et authentifiez-vous dans votre navigateur :**
```bash
sudo tailscale up
```
5. **Vous pouvez trouver votre adresse IPv4 Tailscale en exécutant :**
```bash
tailscale ip -4
```
Connectez-vous à votre compte Tailscale pour authentifier votre Raspberry Pi.

### Étape 5 : Déploiement des Projets avec Docker

1. **Créer des Conteneurs pour Chaque Projet** :
   - Utilisez Portainer pour créer des conteneurs pour chaque projet.
   - Par exemple, pour **Home Assistant** :
     - Allez dans "Containers" puis "Add container".
     - Utilisez l'image `homeassistant/raspberrypi4-homeassistant:stable`.
     - Configurez les ports (par exemple, 8123 pour l'interface web).
     - Ajoutez des volumes pour persister les données (par exemple, `/path/to/config:/config`).

2. **Exemple de Configuration pour Home Assistant** :
   - Dans Portainer, configurez les volumes et les ports comme suit :
     - **Volumes** : `/home/pi/homeassistant:/config`
     - **Ports** : `8123:8123`

### Étape 6 : Configuration de Nginx Proxy Manager

1. **Créer un Répertoire pour Nginx Proxy Manager** :
   - Créez un répertoire pour stocker les données de Nginx Proxy Manager :
     ```bash
     mkdir -p /home/pi/nginx-proxy-manager/data
     mkdir -p /home/pi/nginx-proxy-manager/letsencrypt
     ```

2. **Créer un Fichier `docker-compose.yml`** :
   - Dans le répertoire `/home/pi/nginx-proxy-manager`, créez un fichier `docker-compose.yml` avec le contenu suivant :
     ```yaml
     version: '3'
     services:
       app:
         image: 'jc21/nginx-proxy-manager:latest'
         restart: unless-stopped
         ports:
           - '80:80'
           - '81:81'
           - '443:443'
         volumes:
           - ./data:/data
           - ./letsencrypt:/etc/letsencrypt
     ```

3. **Lancer Nginx Proxy Manager** :
   - Exécutez la commande suivante dans le répertoire contenant le fichier `docker-compose.yml` :
     ```bash
     docker-compose up -d
     ```

4. **Accéder à Nginx Proxy Manager** :
   - Ouvrez votre navigateur et accédez à `http://<IP_DE_VOTRE_PI>:81`.
   - Configurez les proxys pour chaque service.

### Étape 7 : Surveillance et Maintenance

1. **Surveillance** :
   - Utilisez Portainer pour surveiller l'état et les performances des conteneurs.
   - Configurez des alertes si nécessaire.

2. **Sauvegarde** :
   - Sauvegardez régulièrement vos fichiers de configuration et vos données importantes.

### Conclusion

En suivant ces étapes détaillées, vous pouvez configurer votre Raspberry Pi 5 pour exécuter vos projets de manière efficace et sécurisée. Cette configuration utilise Docker pour la conteneurisation, Portainer pour la gestion des conteneurs, Tailscale pour l'accès à distance, et Nginx Proxy Manager pour gérer les accès via un reverse proxy. Si vous avez besoin d'aide pour une étape spécifique, n'hésitez pas à demander !