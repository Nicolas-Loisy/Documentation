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


___

# Cheat Sheet `screen` - Ubuntu

## Installer screen
```bash
sudo apt update
sudo apt install screen
````

## Créer une session

```bash
screen -S nom_session
```

## Détacher la session

```text
Ctrl+a, d          # raccourci clavier
Ctrl+a : detach    # commande interne
```

## Lister les sessions

```bash
screen -ls
```

## Reprendre une session

```bash
screen -r nom_session
screen -r ID_session
```

## Commandes utiles dans screen

| Raccourci       | Action                        |
| --------------- | ----------------------------- |
| Ctrl+a c        | Nouvelle fenêtre              |
| Ctrl+a n        | Fenêtre suivante              |
| Ctrl+a p        | Fenêtre précédente            |
| Ctrl+a d        | Détacher session              |
| Ctrl+a : detach | Détacher via commande interne |

> ⚠️ “No other window” signifie qu’il n’y a qu’une seule fenêtre. Normal.

## Comparaison rapide

| Méthode       | Survie déconnexion | Visualisation sortie |
| ------------- | ------------------ | -------------------- |
| bg + disown   | Oui                | Non (fichier)        |
| nohup         | Oui                | Non (fichier)        |
| screen / tmux | Oui                | Oui                  |

## Exemple complet

```bash
screen -S ma_session
./ma_commande
Ctrl+A D
screen -r ma_session
```

## Alternative : bg + disown / nohup

### La commande est déjà lancée

bg + disown (commande déjà lancée)
#### Suspendre la commande en cours
Ctrl+Z

#### La relancer en arrière-plan
bg

#### Détacher de la session SSH
disown

### La commande n'est pas encore start
nohup (commande à lancer)
nohup ./ma_commande > sortie.log 2>&1 &


nohup empêche la commande d’être tuée à la fermeture de SSH

> sortie.log 2>&1 redirige la sortie standard et les erreurs vers un fichier

& lance la commande en arrière-plan

