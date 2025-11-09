# SSH

## Connexion SSH de base

**Se connecter à un serveur distant :**
```bash
ssh utilisateur@adresse_ip
ssh utilisateur@nom_de_domaine
ssh utilisateur@192.168.1.100
```

**Se connecter avec un port spécifique :**
```bash
ssh -p 2222 utilisateur@adresse_ip
```

**Se connecter avec une clé privée spécifique :**
```bash
ssh -i /chemin/vers/cle_privee utilisateur@adresse_ip
```

## Gestion des clés SSH

**Générer une paire de clés SSH (RSA) :**
```bash
ssh-keygen -t rsa -b 4096 -C "votre_email@exemple.com"
```

**Générer une paire de clés SSH (Ed25519 - recommandé) :**
```bash
ssh-keygen -t ed25519 -C "votre_email@exemple.com"
```

**Copier la clé publique vers un serveur distant :**
```bash
ssh-copy-id utilisateur@adresse_ip
ssh-copy-id -i ~/.ssh/id_rsa.pub utilisateur@adresse_ip
```

**Afficher la clé publique :**
```bash
cat ~/.ssh/id_rsa.pub
cat ~/.ssh/id_ed25519.pub
```

**Lister les clés SSH :**
```bash
ls -la ~/.ssh
```

**Modifier les permissions des clés (sécurité) :**
```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub
chmod 600 ~/.ssh/config
```

## Configuration SSH

**Fichier de configuration SSH :**
```bash
nano ~/.ssh/config
```

**Exemple de configuration SSH :**
```
Host github
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519

Host serveur-perso
    HostName 192.168.1.100
    User admin
    Port 2222
    IdentityFile ~/.ssh/cle_serveur
```

**Utilisation avec la configuration :**
```bash
ssh github
ssh serveur-perso
```

## Commandes utiles

**Tester la connexion SSH à GitHub :**
```bash
ssh -T git@github.com
```

**Sortie attendue :**
```
Hi username! You've successfully authenticated, but GitHub does not provide shell access.
```

**Tester la connexion SSH à GitLab :**
```bash
ssh -T git@gitlab.com
```

**Mode verbose (debug) :**
```bash
ssh -v utilisateur@adresse_ip     # Verbose
ssh -vv utilisateur@adresse_ip    # Plus de détails
ssh -vvv utilisateur@adresse_ip   # Maximum de détails
```

**Exécuter une commande à distance sans ouvrir de session :**
```bash
ssh utilisateur@adresse_ip "ls -la"
ssh utilisateur@adresse_ip "df -h"
ssh utilisateur@adresse_ip "uptime"
```

**Copier des fichiers via SSH (SCP) :**
```bash
# Local vers distant
scp fichier.txt utilisateur@adresse_ip:/chemin/destination/

# Distant vers local
scp utilisateur@adresse_ip:/chemin/fichier.txt /chemin/local/

# Copier un dossier entier
scp -r dossier/ utilisateur@adresse_ip:/chemin/destination/
```

**Copier des fichiers via SSH (RSYNC - recommandé) :**
```bash
# Local vers distant
rsync -avz fichier.txt utilisateur@adresse_ip:/chemin/destination/

# Distant vers local
rsync -avz utilisateur@adresse_ip:/chemin/fichier.txt /chemin/local/

# Synchroniser un dossier entier
rsync -avz --delete dossier/ utilisateur@adresse_ip:/chemin/destination/
```

**Créer un tunnel SSH (port forwarding) :**
```bash
# Local port forwarding
ssh -L 8080:localhost:80 utilisateur@adresse_ip

# Remote port forwarding
ssh -R 8080:localhost:80 utilisateur@adresse_ip

# Dynamic port forwarding (SOCKS proxy)
ssh -D 8080 utilisateur@adresse_ip
```

**Maintenir une connexion SSH active :**
```bash
ssh -o ServerAliveInterval=60 utilisateur@adresse_ip
```

**Se connecter sans vérification de l'hôte (attention : non sécurisé) :**
```bash
ssh -o StrictHostKeyChecking=no utilisateur@adresse_ip
```

## SSH Agent

**Démarrer l'agent SSH :**
```bash
eval "$(ssh-agent -s)"
```

**Ajouter une clé à l'agent SSH :**
```bash
ssh-add ~/.ssh/id_rsa
ssh-add ~/.ssh/id_ed25519
```

**Lister les clés chargées dans l'agent :**
```bash
ssh-add -l
```

**Supprimer toutes les clés de l'agent :**
```bash
ssh-add -D
```

## Sécurité SSH

**Désactiver l'authentification par mot de passe (fichier `/etc/ssh/sshd_config` sur le serveur) :**
```
PasswordAuthentication no
PubkeyAuthentication yes
```

**Changer le port SSH par défaut (fichier `/etc/ssh/sshd_config`) :**
```
Port 2222
```

**Désactiver la connexion root (fichier `/etc/ssh/sshd_config`) :**
```
PermitRootLogin no
```

**Redémarrer le service SSH après modification :**
```bash
sudo systemctl restart sshd
sudo service ssh restart
```

## Problèmes courants

**Erreur "Permission denied (publickey)" :**
- Vérifier que la clé publique est bien dans `~/.ssh/authorized_keys` sur le serveur
- Vérifier les permissions des fichiers et dossiers SSH
- Utiliser `ssh -v` pour voir les détails de l'erreur

**Erreur "Host key verification failed" :**
```bash
# Supprimer l'ancienne clé d'hôte
ssh-keygen -R adresse_ip
ssh-keygen -R nom_de_domaine
```

**Connexion trop lente :**
- Désactiver la recherche DNS inverse dans `/etc/ssh/sshd_config` :
```
UseDNS no
```

## Ressources

- Documentation officielle OpenSSH : https://www.openssh.com/
- Guide de sécurité SSH : https://www.ssh.com/academy/ssh/security
