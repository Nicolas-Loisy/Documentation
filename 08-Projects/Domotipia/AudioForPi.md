# Configuration Audio pour Raspberry Pi

## Solution MPD/MPC

Sur le Pi, faire un serveur audio MPD/MPC.

### Configuration MPD

Dans le fichier de config : `/etc/mpd.conf`

Commenter :
- `music_directory`
- `db_file`
- `user`

Ajouter :

```
audio_output {
    type            "alsa"
    name            "My ALSA Device"
    device          "hw:0,0"  # This specifies the first ALSA device
    mixer_type      "software"
}
```

### Intégration avec Home Assistant

Dans HA, ajouter l'intégration MPD (localhost, 6600), un nouveau mediaplayer apparaîtra.

---

## Rhasspy - Assistant Vocal

### Configurer le conteneur Rhasspy

Dans l'interface de création du conteneur, configurez comme suit :

**Name** : `rhasspy`

**Image** : `rhasspy/rhasspy`

**Ports** : Mappez le port 12101 pour l'interface web de Rhasspy (si vous utilisez ce port par défaut).

12101 sur l'hôte → 12101 dans le conteneur.

Pour permettre l'accès au microphone du Raspberry Pi, assurez-vous que le périphérique audio est mappé correctement dans le conteneur. Ajoutez une section Devices dans la configuration du conteneur :

`/dev/snd : /dev/snd` (Cela permet à Rhasspy d'utiliser le microphone de votre Raspberry Pi).

**Commande, override** : `--profile fr`

### Démarrer le conteneur Rhasspy

Une fois le conteneur configuré, cliquez sur Deploy the container pour lancer Rhasspy dans Docker via Portainer.

### Accéder à Rhasspy

Vous pouvez maintenant accéder à l'interface web de Rhasspy en vous rendant sur `http://<votre-ip>:12101` dans votre navigateur.
