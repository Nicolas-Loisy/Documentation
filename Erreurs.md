# Erreurs Classées par Technologie

### **Erreurs Localhost (Configuration Apache)**

- **Problème** : Mauvaise configuration des logs dans `docker/config/projetAdministration_vhost.conf`.
- **Erreur** : L'erreur de log `ErrorLog` et `CustomLog` pointe vers des fichiers Apache incorrects.
- **Solution** : Modifier les lignes dans le fichier de configuration pour correspondre à la configuration Apache de la machine Docker.
    - Remplacer :
      ```bash
      ErrorLog /var/log/httpd/projetAdministration_error.log
      CustomLog /var/log/httpd/projetAdministration.log combined
      ```
    - Par :
      ```bash
      ErrorLog /var/log/apache2/projetAdministration_error.log
      CustomLog /var/log/apache2/projetAdministration.log combined
      ```

---

### **Erreur Page 500**

- **Problème** : Toutes les pages du site affichent une erreur 500.
- **Cause** : Un espace supplémentaire dans le fichier `.env.local` après une clé d’environnement, par exemple :
    ```env
    LA_KEY= blablabla
    ```
- **Solution** : Supprimer l’espace supplémentaire après le signe égal dans le fichier `.env.local`.
    - Exemple corrigé :
      ```env
      LA_KEY=blablabla
      ```

---

### **Erreur "Class Not Found"**

- **Problème** : L'erreur indique que la classe `App\Command\...` n’a pas été trouvée dans le fichier `/src/Command/...`.
- **Cause** : Mauvaise correspondance entre le nom de la classe, le nom du fichier ou le namespace.
- **Solution** : Vérifier les points suivants :
    - Le nom de la classe et le nom du fichier doivent correspondre.
    - Vérifier le namespace de la classe.
    - Assurer qu’il n’y a pas de virgule (`,`) en trop à la fin des paramètres d’une fonction ou méthode.

---

### **Erreur Docker - Makefile**

- **Problème** : Erreur dans le fichier `Makefile` :
    ```
    Makefile:4: *** missing separator.  Stop.
    ```
- **Cause** : Mauvaise indentation dans le fichier `Makefile`.
- **Solution** : Assurer que les indentations dans le `Makefile` sont effectuées avec des tabulations, et non des espaces. 
    - Dans VS Code : En bas à droite, cliquez sur "Space:4" et sélectionnez "Convert indent to Tabs".

---

### **Problème de Base de Données MySQL - Symfony 4 (Symf4)**

- **Problème** : Erreur de connexion à la base de données dans le container Docker :
    ```
    ERROR 2002 (HY000): Can't connect to local MySQL server through socket '/tmp/mysql.sock' (111)
    ```
- **Cause** : Il existe un problème de droits d’accès sur les fichiers de sauvegarde MySQL.
- **Solution** : Modifier les permissions des fichiers dans le container Docker.
    1. Accéder au répertoire où sont stockés les fichiers MySQL :
        ```bash
        cd /var/lib/mysql
        ```
    2. Modifier les permissions des fichiers :
        ```bash
        chown -R root:root ib_logfile0
        chown -R root:root ib_logfile1
        chown -R root:root ibdata1
        chown -R root:root nom_save_database_1/
        chown -R root:root nom_save_database_2/
        chown -R root:root performance_schema/
        chown -R root:root test/
        ```
    3. Revenir au répertoire principal et donner les bonnes permissions au répertoire MySQL :
        ```bash
        cd /var/lib
        chown -R mysql:root mysql
        ```
    4. Redémarrer MySQL :
        ```bash
        service mysqld start
        ```