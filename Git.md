# Aide-mémoire Git

## Étapes

**1. Récupérer les dernières modifications**

Sur la branche locale "master":

```
git pull origin master
```

**2. Créer une nouvelle branche**

Depuis la branche "master":

```
git checkout -b laNewBranch
```

**3. Développer sur la nouvelle branche**

Ajouter les fichiers modifiés:

```
git add --all
```

Committer les modifications:

```
git commit -m "Description du commit"
```

Répéter les étapes `git add` et `git commit` pour chaque étape de développement.

**4. Pousser les modifications**

Pousser la nouvelle branche vers le dépôt distant:

```
git push origin laNewBranch
```

Si nécessaire, créer une pull request pour soumettre vos modifications.

**Retourner à l'étape 3**

Continuer le développement sur la nouvelle branche en répétant les étapes 3 et 4.

**Si nouvelle tâche:**

**Retourner à l'étape 1**

## Dico

* `git branch`: Lister les branches
* `git add --all`: Ajouter tous les fichiers modifiés (y compris les suppressions)
* `git branch -m newBranchName`: Renommer la branche actuelle
* `git branch -d nomDeLaBranche`: Supprimer la branche indiquée
* `git checkout -b develop origin/develop`: Récupérer une branche distante (ex: develop)

**Mises à jour**

* `git fetch --all`: Mettre à jour les branches de suivi à distance

**Fusionner des branches**

* `git pull origin labranch`: Mettre à jour la branche locale avec les commits distants (résoudre les conflits si nécessaire)
* `git merge labranch`: Vérifier si la branche est compatible avec la branche actuelle

**Gérer les modifications non commit**

* `git stash`: Sauvegarder les modifications non commit
* `git stash pop`: Restaurer les modifications sauvegardées

**Intégrer un commit spécifique**

* `git cherry-pick NUMduCOMMIT`: Intégrer un commit particulier dans la branche actuelle

**Historique des commits**

* `git log`: Afficher l'historique des commits (q pour quitter)

**Résoudre des conflits**

Si des modifications ont été faites dans la branche master (remote) avant de push:

1. `git checkout master`
2. `git pull origin master`
3. `git checkout maBranche`
4. `git merge master` (résoudre les conflits)
5. Commiter les résolutions et push

**Corriger un commit**

Si le dernier commit en local est erroné:

* `git commit --amend --no-edit`: "Écraser" le dernier commit
* `git commit -f --amend --no-edit`: "Écraser" le dernier commit si déjà push (attention à l'historique)
* `git commit --amend -m "le new message"`: "Écraser" le dernier commit avec un nouveau message

**Modifier la date d'un commit**

* `GIT_COMMITTER_DATE="$(date)" git commit --amend --no-edit --date "$(date)"`: Remplacer la date du dernier commit par la date du jour

**Problème Zone.Identifier**

* `git config core.protectNTFS false`

**Informations complémentaires**

* Pour plus d'informations, consultez la documentation officielle Git: [https://git-scm.com/book/en/v2](https://git-scm.com/book/en/v2)
