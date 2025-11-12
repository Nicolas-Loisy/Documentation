# Documentation Docker

[Docker Cheat Sheet](https://docs.docker.com/get-started/docker_cheatsheet.pdf)

## 1. Commandes de Base

### Démarrer un Container
```bash
docker-compose up -d
```

### Accéder à un Container
```bash
docker exec -ti nom_du_container_docker /bin/bash
```

### Arrêter les Containers
```bash
docker-compose down
```

---

## 2. Lister les Containers et Images

### Lister les Containers
```bash
docker container ls
```

### Lister les Images
```bash
docker images
```

---

## 3. Exporter une Image Docker

### Sauvegarder une Image
```bash
docker save -o NOMCOPIEIMG REPOSITORY:TAG
```
Exemple :
```bash
docker save -o mysql8 mysql:8
```

---

## 4. Différence entre `docker run` et `docker-compose`
- **`docker run`** : Lance un seul container.
- **`docker-compose`** : Gère plusieurs containers définis dans `docker-compose.yml`.

---

## 5. Créer une Image Docker

### Générer une Image
```bash
docker build -t nom-image:latest .
```


---

## Erreurs

### Erreur  "network not found"

```
Error response from daemon: failed to set up container networking:
network <id> not found
```

1. Nettoyer l’environnement

```bash
docker compose -f etc/docker/docker-compose.yml down --remove-orphans
docker network prune -f
```

2. Recréer les conteneurs

```bash
make docker-start
# ou
docker compose -f etc/docker/docker-compose.yml up -d --build
```

3. Vérifier le statut

```bash
docker ps
```