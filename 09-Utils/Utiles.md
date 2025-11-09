# Pratique : Commandes Utiles

### Générer un fichier avec une taille précise

La commande suivante permet de créer un fichier de taille spécifique (par exemple, 100 Mo) :

```bash
fsutil file createnew c:fichier100Mo.txt 104857600
```

**Explication** :
- `fichier100Mo.txt` : Nom du fichier à créer.
- `104857600` : Taille en octets (100 Mo).

(Note : Cette commande est spécifique à Windows).
