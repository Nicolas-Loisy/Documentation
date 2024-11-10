# Node.js

---

## 1. Installation de Node.js

### A. Installer `curl`
Si `curl` n'est pas déjà installé, vous pouvez l'ajouter avec la commande suivante :
```bash
apt-get install curl
```

### B. Télécharger et installer Node.js

1. Récupérer le script d'installation pour la version souhaitée de Node.js (ici version 4.x) et l'exécuter :
   ```bash
   curl -sL https://deb.nodesource.com/setup_4.x | bash
   ```

2. Installer Node.js :
   ```bash
   apt-get install nodejs
   ```

### C. Vérification de l'installation

- Vérifiez que Node.js est installé avec succès :
  ```bash
  node -v
  ```

- Vérifiez la version de `npm` (le gestionnaire de paquets Node.js) :
  ```bash
  npm -v
  ```

---

Cette documentation couvre l'installation de Node.js sur une machine utilisant `apt` (gestionnaire de paquets Ubuntu/Debian).