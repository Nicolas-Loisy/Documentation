Solution :

Sur le pi, faire un serveur audio MPD/MPC.

Dans le fichier de config : /etc/mpd.conf
Commenter :
- music_directory 
- db_file
- user

Ajouter :

```
audio_output {
    type            "alsa"
    name            "My ALSA Device"
    device          "hw:0,0"  # This specifies the first ALSA device
    mixer_type      "software"
}
```


Dans HA, ajouter l'integration MPD (localhost, 6600), un new mediaplayer apparaitra.