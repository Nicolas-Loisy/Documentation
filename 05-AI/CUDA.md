# CUDA

Pour savoir si votre PC peut utiliser CUDA, il faut vérifier si votre carte graphique (GPU) est compatible avec CUDA. Voici les étapes pour le savoir :


### 1. **Vérifiez le fabricant de votre GPU**
CUDA est une technologie développée par NVIDIA. Si votre GPU est fabriqué par une autre marque (comme AMD ou Intel), CUDA ne sera pas disponible.

---

### 2. **Identifiez votre GPU**
- **Sous Windows** :
  1. Faites un clic droit sur le bureau et sélectionnez **Paramètres d'affichage**.
  2. Cliquez sur **Paramètres graphiques avancés** pour voir le nom de votre GPU.
  3. Ou ouvrez le **Gestionnaire de périphériques**, développez la section **Cartes graphiques**.

- **Sous Linux** :
  1. Ouvrez un terminal et tapez :
     ```bash
     lspci | grep -i nvidia
     ```
  2. Ou utilisez `nvidia-smi` si le pilote NVIDIA est installé.

---

### 3. **Vérifiez la compatibilité CUDA**
Une fois que vous connaissez le modèle de votre GPU :
1. Rendez-vous sur la page officielle de [compatibilité CUDA de NVIDIA](https://developer.nvidia.com/cuda-gpus).
2. Recherchez votre modèle de GPU dans la liste. Si votre GPU figure dans la liste, il est compatible avec CUDA.

---

### 4. **Installez les pilotes et CUDA Toolkit**
- Téléchargez et installez les derniers pilotes NVIDIA pour votre GPU depuis le site officiel : [Pilotes NVIDIA](https://www.nvidia.com/Download/index.aspx?lang=fr).
- Téléchargez le **CUDA Toolkit** correspondant à votre système d'exploitation : [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

---

### 5. **Testez la compatibilité**
Après installation, vous pouvez tester si CUDA est fonctionnel :
1. **Sous Windows/Linux**, ouvrez un terminal ou une invite de commande.
2. Tapez :
   ```bash
   nvcc --version
   ```
   Si la commande retourne une version de CUDA, votre PC est prêt à utiliser CUDA.
