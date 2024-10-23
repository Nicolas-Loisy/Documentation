# Python

## Ressources et outils Python

**Cours Udemy:** [https://www.udemy.com/course/the-complete-python-course/](https://www.udemy.com/course/the-complete-python-course/)

**Tutoriel DocstringFr Python:** [https://github.com/DocstringFr/la-formation-complete-python](https://github.com/DocstringFr/la-formation-complete-python)

**PyPI - Index des packages Python:** [https://pypi.org/](https://pypi.org/)

**IDE:** PyCharm Community (téléchargement depuis le site officiel)

**Gestionnaire d'environnements:** pyenv (instructions d'installation ci-dessous)

**Base de données vectorielles hébergée:** pinecone.io

**Comprendre `__init__.py`:**

- Ce fichier sert de marqueur pour indiquer qu'un répertoire contient un package ou un module Python.
- Lorsque vous importez un package ou un module, Python charge automatiquement ce fichier.

## Commandes essentielles dans le terminal:

- **Lister les versions Python installées:** `py --list` (si applicable)
- **Vérifier la version Python actuelle:** `python --version` ou `python3 --version`
- **Créer un environnement virtuel:** `python -m venv <nom_environnement>`
- **Activer un environnement virtuel:** `source <nom_environnement>/bin/activate` (Windows : `<nom_environnement>\Scripts\activate`)
- **Désactiver un environnement virtuel:** `deactivate`
- **Installer des packages:** `pip install <nom_paquet>`
- **Mettre à jour des packages:** `pip install --upgrade <nom_paquet>`
- **Enregistrer les packages installés dans un fichier requirements:** `pip freeze > requirements.txt`
- **Installer des packages à partir d'un fichier requirements:** `pip install -r requirements.txt`
- **Vérifier l'installation d'un package:** `python -c "import <nom_paquet>; print(<nom_paquet>)"`

## Installer Python 3.11.3 avec pyenv dans WSL:

**Prérequis:**

- Assurez-vous d'avoir WSL 2 (version 2) ou une version ultérieure installée et configurée.

**Étapes:**

1. **Installer les dépendances requises:**
    ```bash
    sudo apt-get install gcc
    ```
    
    ```bash
    sudo apt update; sudo apt install build-essential libssl-dev zlib1g-dev \libbz2-dev libreadline-dev libsqlite3-dev curl \libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
    ```

2. **Installer et configurer pyenv:**

   ```bash
   curl -sL https://pyenv.run | bash
   ```

   **Important:** Lisez attentivement la sortie du script et n'acceptez les modifications que si vous les comprenez parfaitement.
   Copiez et collez les lignes suivantes à la fin de votre fichier `~/.bashrc` (remplacez `~` par le chemin de votre répertoire personnel si nécessaire) :

   ```bash
   export PYENV_ROOT="$HOME/.pyenv"
    [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
   ```

3. **Installer Python 3.11.3:**

   ```bash
   pyenv install 3.11.3
   ```

4. **Définir Python 3.11.3 comme version locale:**

   ```bash
   pyenv local 3.11.3
   ```

5. **Vérifier l'installation:**

   ```bash
   python --version
   ```

Cela devrait afficher `Python 3.11.3` (ou similaire) si l'installation a réussi.

**Dépannage:**

- Si vous rencontrez des erreurs de permission, utilisez `sudo` avant les commandes nécessitant des privilèges élevés.
- Pour un dépannage plus avancé, envisagez de rechercher de l'aide sur des forums en ligne ou dans des communautés.

**N'oubliez pas d'activer l'environnement virtuel approprié avant de travailler sur des projets spécifiques pour éviter les conflits.**
