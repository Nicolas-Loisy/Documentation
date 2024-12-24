# **Guide d'installation de PyTorch avec CUDA 12.4**

### **Prérequis**
1. **GPU NVIDIA** compatible avec CUDA.
2. **Pilotes NVIDIA** installés et à jour.
3. **CUDA Toolkit 12.4** installé.
4. **Python 3.8 ou supérieur** (Python 3.11 recommandé).
5. **pip** à jour :
   ```bash
   python -m pip install --upgrade pip
   ```

---

### **Étapes d'installation**

#### **1. Désinstaller les anciennes versions de CUDA**
Avant d'installer CUDA 12.4, supprimez toutes les versions précédentes de CUDA et leurs composants associés.

##### Sous Windows :
1. Ouvrez **Panneau de configuration** > **Programmes et fonctionnalités**.
2. Désinstallez :
   - NVIDIA CUDA Toolkit
   - NVIDIA Graphics Driver (facultatif si vous souhaitez mettre à jour également).
   - Tout autre composant CUDA (Nsight, CUDA Samples, etc.).

---

#### **2. Installer CUDA Toolkit 12.4**
1. Téléchargez CUDA 12.4 depuis le site officiel des [archives CUDA de NVIDIA pour Windows](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local) ou depuis [les archives CUDA de NVIDIA](https://developer.nvidia.com/cuda-toolkit-archive).
2. Installez le toolkit en suivant les instructions pour votre système d'exploitation.

##### Vérifiez l'installation :
- **Chemin d'accès à CUDA** :
  Assurez-vous que le chemin de CUDA est ajouté à la variable d'environnement `PATH`.
  - Sous Windows : `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin`.
  - Sous Linux : Ajoutez `/usr/local/cuda-12.4/bin` à votre `PATH`.

- **Tester CUDA** :
  ```bash
  nvcc --version
  ```
  Vous devriez voir :
  ```
  Cuda compilation tools, release 12.4, V12.4.x
  ```

---

#### **3. Installer PyTorch avec CUDA 12.4**
1. Installez PyTorch à l'aide de la commande suivante :
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

2. Vérifiez que PyTorch a été correctement installé :

   Dans le terminal : ```python```
   ```python
   import torch
   print("PyTorch version:", torch.__version__)
   print("CUDA available:", torch.cuda.is_available())
   print("Current GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
   ```

---

### **Problèmes courants et solutions**

#### **1. CUDA non détecté (`CUDA available: False`)**
- **Cause probable :**
  Les pilotes NVIDIA ou CUDA ne sont pas correctement installés.
- **Solution :**
  - Vérifiez que les pilotes NVIDIA sont installés :
    ```bash
    nvidia-smi
    ```
    Cette commande doit afficher les informations sur votre GPU.
  - Assurez-vous que le chemin CUDA est dans votre `PATH` et redémarrez votre machine.

---

#### **2. PyTorch utilise le CPU (`PyTorch version: 2.x+cpu`)**
- **Cause probable :**
  Une version CPU de PyTorch a été installée par erreur.
- **Solution :**
  Réinstallez PyTorch avec le support CUDA 12.4 :
  ```bash
  pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```

---

#### **3. `nvcc` fonctionne mais PyTorch ne détecte pas CUDA**
- **Cause probable :**
  CUDA est installé, mais la version de PyTorch ne correspond pas.
- **Solution :**
  - Assurez-vous que CUDA 12.4 est installé.
  - Vérifiez la compatibilité des versions PyTorch et CUDA sur la [page officielle de PyTorch](https://pytorch.org/get-started/previous-versions/).

---

#### **4. Erreur `No GPU detected`**
- **Cause probable :**
  Le GPU est désactivé ou indisponible.
- **Solution :**
  - Activez votre GPU dans le panneau de configuration NVIDIA.
  - Sur les ordinateurs portables, assurez-vous que le GPU NVIDIA est utilisé au lieu du GPU intégré.

---

### **Exemple de sortie attendue**

Lorsque tout est correctement configuré, vous devriez voir :
```python
PyTorch version: 2.0.1+cu124
CUDA available: True
Current GPU: NVIDIA GeForce RTX 3080
```

---

### **Résumé des commandes clés**
```bash
# Vérifier les pilotes NVIDIA
nvidia-smi

# Vérifier la version CUDA
nvcc --version

# Installer PyTorch avec CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Tester PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

Avec ce guide, vous devriez être en mesure d’installer et de configurer PyTorch avec CUDA 12.4 sans problème. Si vous rencontrez d'autres difficultés, n'hésitez pas à demander de l'aide !

--- 

Le lien pour télécharger CUDA 12.4 pour Windows a bien été ajouté dans la section d'installation de CUDA Toolkit.


