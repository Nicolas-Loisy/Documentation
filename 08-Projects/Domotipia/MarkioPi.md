# Configuration du Raspberry Pi 5 - MarkioPi

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

1. **Installer le `lsb-release & curl` plugin :**
```bash
sudo apt install lsb-release curl
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


### Étape 5 : Cockpit

Installation de Cockpit sur Raspberry Pi

1. **Mise à jour du système**
```bash
sudo apt update
sudo apt upgrade -y
```

2. **Configuration réseau**

Il est recommandé d'utiliser une **IP statique** (idéalement via réservation DHCP dans le routeur).
Sinon, configurez manuellement l'IP fixe sur le Raspberry Pi.


3. **Ajout du dépôt Debian Backports**

Récupérer le nom de code du système :

```bash
. /etc/os-release
```

Ajouter le dépôt :

```bash
echo "deb http://deb.debian.org/debian ${VERSION_CODENAME}-backports main" | \
sudo tee /etc/apt/sources.list.d/backports.list
```

Mettre à jour la liste des paquets :

```bash
sudo apt update
```

4. **Installation de Cockpit**

Installer Cockpit depuis le dépôt backports :

```bash
sudo apt install -t ${VERSION_CODENAME}-backports cockpit
```

5. **Accès à l'interface web**

Récupérer l'adresse IP du Raspberry Pi :

```bash
hostname -I
```

Accéder depuis un navigateur à :

```
https://<IPADDRESS>:9090
```

> Note : Vous verrez un avertissement dû au certificat auto-signé.


### Étape 6 : Installer File Browser sur Raspberry Pi avec Portainer

#### 1. Préparer le Raspberry Pi

Mettre à jour le système :

```bash
sudo apt update && sudo apt upgrade -y
```

Installer **Docker** et **Portainer** :

```bash
# Installer Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Installer Portainer
docker volume create portainer_data
docker run -d -p 9000:9000 -p 8000:8000 \
  --name=portainer \
  --restart=always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce
```

Accéder à Portainer depuis le navigateur :

```
http://<IP_RASPBERRY>:9000
```


#### 2. Préparer les fichiers pour File Browser

Créer les dossiers et fichiers nécessaires :

```bash
sudo mkdir -p /opt/stacks/filebrowser/{data,config}
sudo touch /opt/stacks/filebrowser/data/database.db
sudo chown -R $USER:$USER /opt/stacks/filebrowser
```

Créer le fichier de configuration `/opt/stacks/filebrowser/config/filebrowser.json` :

```json
{
  "port": 80,
  "baseURL": "",
  "address": "",
  "log": "stdout",
  "database": "/database.db",
  "root": "/srv"
}
```

* **baseURL** : laisser vide pour accès direct
* **address** : laisser vide pour écouter toutes les interfaces


#### 3. Déployer File Browser via Portainer

#### Option A : via Stack (recommandé)

1. Dans Portainer : **Stacks → Add Stack**
2. Nom : `filebrowser`
3. Colle ce contenu `docker-compose.yaml` :

```yaml
services:
  filebrowser:
    image: filebrowser/filebrowser
    container_name: filebrowser
    user: "1000:1000"  # UID:GID du user pi
    ports:
      - 8080:80         # Host:Container → 8080 sur le Pi redirige vers 80 du container
    volumes:
      - /:/srv
      - /opt/stacks/filebrowser/data/database.db:/database.db
      - /opt/stacks/filebrowser/config/filebrowser.json:/.filebrowser.json
    restart: always
```

4. Clique **Deploy the stack**.

#### Option B : via Container

1. **Containers → Add container**
2. Nom : `filebrowser`
3. Image : `filebrowser/filebrowser`
4. Ports : `8080 → 80`
5. Volumes :

   * `/ → /srv`
   * `/opt/stacks/filebrowser/data/database.db → /database.db`
   * `/opt/stacks/filebrowser/config/filebrowser.json → /.filebrowser.json`
6. User : `1000:1000`
7. Restart policy : `Always`
8. Déployer le container.

#### 4. Accéder à File Browser

* URL : `http://<IP_RASPBERRY>:8080`
* Identifiant : `admin`
* Mot de passe : généré lors du premier lancement (visible dans les logs du container Portainer).

#### 5. Sécuriser

* Change le mot de passe par défaut dans **Settings → Change Password**.
* Si exposé à Internet, utiliser **reverse proxy + HTTPS**.

#### 6. Mettre à jour File Browser

1. Dans Portainer : **Stacks → filebrowser → Recreate → Pull latest image**
2. Docker téléchargera la dernière version et relancera le container.



### Étape 7 : Déploiement des Projets avec Docker

**Créer des Conteneurs pour Chaque Projet** :
   - Utilisez Portainer pour créer des conteneurs pour chaque projet.

#### Étape 7.1 : Home Assistant

Lancer Home Assistant en Docker.

Dans **Portainer**, crée un nouveau **conteneur** avec ces paramètres :

- **Nom** : `homeassistant`
- **Image** : `ghcr.io/home-assistant/home-assistant:stable`
- **Network** : `host` (très important pour détecter les appareils)
- **Ports** : `8123:8123`
- **Volumes** :
  - `/home/pi/homeassistant:/config`
  - `/etc/localtime:/etc/localtime:ro`
  - `/run/dbus:/run/dbus:ro`
- **Redémarrage** : `Always`
- **Devices** : `/dev/snd:/dev/snd`

Ou en ligne de commande :
```bash
docker run -d \
  --name homeassistant \
  --restart always \
  -v /home/pi/homeassistant:/config \
  -v /etc/localtime:/etc/localtime:ro \
  --network=host \
  ghcr.io/home-assistant/home-assistant:stable
```

Accède à Home Assistant via :
```
http://<IP_DE_TON_PI>:8123
```

### Étape 8 : Configuration de Nginx Proxy Manager

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

### Étape 9 : Surveillance et Maintenance

1. **Surveillance** :
   - Utilisez Portainer pour surveiller l'état et les performances des conteneurs.
   - Configurez des alertes si nécessaire.

2. **Sauvegarde** :
   - Sauvegardez régulièrement vos fichiers de configuration et vos données importantes.

### Conclusion

En suivant ces étapes détaillées, vous pouvez configurer votre Raspberry Pi 5 pour exécuter vos projets de manière efficace et sécurisée. Cette configuration utilise Docker pour la conteneurisation, Portainer pour la gestion des conteneurs, Tailscale pour l'accès à distance, et Nginx Proxy Manager pour gérer les accès via un reverse proxy. Si vous avez besoin d'aide pour une étape spécifique, n'hésitez pas à demander !
