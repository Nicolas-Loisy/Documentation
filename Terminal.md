# Documentation des Terminaux

## Make

`make` est un outil pour automatiser la compilation de projets. Voici comment l'installer sous Windows :

1. **Téléchargez** `make` depuis [ezwinports](https://sourceforge.net/projects/ezwinports/files/) en choisissant `make-4.1-2-without-guile-w32-bin.zip` (version sans `guile`).

2. **Extrayez** l'archive ZIP.

3. **Copiez** le contenu extrait dans `Git\mingw64\` en fusionnant les dossiers sans écraser les fichiers existants.

Assurez-vous que les outils UNIX standard sont installés et configurés dans votre PATH pour éviter les erreurs.

### Gestion des erreurs dans un Makefile

Dans un Makefile, chaque commande est exécutée dans un sous-shell indépendant, ce qui signifie que l'échec d'une commande entraîne l'arrêt immédiat de l'exécution du Makefile, sauf si une gestion explicite des erreurs est mise en place.

#### Solution
Ajouter `|| true` après chaque commande pour ignorer les erreurs et continuer l'exécution.

#### Exemple corrigé :
```make
run-tests:
	@echo "***** $@"
	@{
		source .venv/bin/activate;
		cd project_directory;
		npx command --config ./config-file-1.yaml || true;
		npx command --config ./config-file-2.yaml || true;
		npx command --config ./config-file-3.yaml || true;
		npx command --config ./config-file-4.yaml || true;
	}
```

#### Alternative : Stockage des erreurs
Rediriger les erreurs vers un fichier pour analyse :
```make
npx command --config ./config-file-1.yaml || echo "Erreur sur config-file-1" >> errors.log
```

#### Méthodes à éviter
- `-npx` : Provoque une erreur de commande invalide.
- `set -e` : Stoppe l'exécution au premier échec.


