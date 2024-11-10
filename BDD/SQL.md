# SQL / MySQL

## 1. Importer un CSV dans MySQL

### Étapes :

#### A. Depuis le terminal Ubuntu (hôte) :
```bash
docker cp /mnt/c/Users/Nicolas\ L/Desktop/fichier_CSV.csv mysql:/fichier_CSV.csv
docker exec -ti mysql /bin/bash
```

#### B. Dans le container MySQL :
1. Connectez-vous à MySQL :
   ```bash
   mysql --local-infile=1 -h localhost -u root -p
   ```
   (Mot de passe : `root`)

2. Activez l'importation locale des fichiers CSV :
   ```sql
   SET GLOBAL local_infile=1;
   SHOW GLOBAL VARIABLES LIKE 'local_infile';
   ```

3. Sélectionnez la base de données :
   ```sql
   USE nom_database;
   ```

4. Importez le CSV dans la table :
   ```sql
   LOAD DATA LOCAL INFILE '/fichier_CSV.csv' 
   INTO TABLE testtable 
   FIELDS TERMINATED BY ',' 
   LINES TERMINATED BY '\n' 
   (nomCol1, nomCol2, nomCol3, nomCol4, nomCol5, nomCol6, ...);
   ```

---

## 2. Création de Base de Données et Utilisateurs

### A. Créer un utilisateur et lui accorder des privilèges :

1. Créer un utilisateur local :
   ```sql
   CREATE USER 'nomDataBase'@'localhost' IDENTIFIED BY 'nomUserDataBase';
   ```

2. Créer un utilisateur accessible depuis n'importe où :
   ```sql
   CREATE USER 'nomDataBase'@'%' IDENTIFIED BY 'nomUserDataBase';
   ```

3. Accorder les privilèges :
   ```sql
   GRANT ALL PRIVILEGES ON *.* TO 'nomDataBase'@'%' WITH GRANT OPTION;
   GRANT ALL PRIVILEGES ON *.* TO 'nomDataBase'@'localhost' WITH GRANT OPTION;
   ```

4. Vérifier les utilisateurs existants :
   ```sql
   USE mysql;
   SELECT Host, user FROM user;
   ```

5. Créer une base de données et l'associer à un utilisateur :
   ```sql
   CREATE DATABASE nomDatabase;
   GRANT ALL PRIVILEGES ON nomDatabase.* TO 'user'@'localhost';
   GRANT ALL PRIVILEGES ON nomDatabase.* TO 'user'@'%';
   ```

6. Vérifier les privilèges de l'utilisateur :
   ```sql
   SHOW GRANTS;
   ```

---

## 3. Gestion des Bases de Données

### A. Afficher les bases de données et tables

- Lister toutes les bases de données :
  ```sql
  SHOW DATABASES;
  ```

- Sélectionner une base de données :
  ```sql
  USE databasename;
  ```

- Lister les tables de la base sélectionnée :
  ```sql
  SHOW TABLES;
  ```

- Afficher la structure d'une table :
  ```sql
  DESC tablename;
  ```

---

## 4. Connexion à la Base de Données depuis Symfony

### A. Connexion dans le container MySQL :
```bash
docker exec -ti mysql /bin/bash
mysql -u root -p
```
(Mot de passe : `root`)

### B. Connexion depuis le Backend :
```bash
mysql -h mysql nomDataBase -p
```
(Mot de passe : `root`)

---

## 5. Exporter et Importer des Bases de Données

### A. Exporter une base de données avec `mysqldump` :

1. Dans le container Backend (après `docker-compose up`) :
   ```bash
   mysqldump -h database -u nomUser -p nomDataBase > dump.sql
   ```

2. Si cela ne fonctionne pas, utilisez `root` comme utilisateur :
   ```bash
   mysqldump -h database -u root -p nomDataBase > dump.sql
   ```

### B. Importer un dump SQL dans MySQL :

1. Copier le dump dans le container MySQL :
   ```bash
   docker cp /mnt/c/Users/Nicolas\ L/Desktop/dump.sql mysql:/dump.sql
   ```

2. Accéder au container MySQL :
   ```bash
   docker exec -ti mysql /bin/bash
   ```

3. Importer le dump :
   ```bash
   mysql -p nomDataBase < /dump.sql
   ```
   (Mot de passe : `root`)

4. Accorder des privilèges après l'import :
   ```sql
   GRANT ALL PRIVILEGES ON nomDataBase.* TO 'nomUser'@'localhost';
   ```