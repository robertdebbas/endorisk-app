# Mettre EndoRisk en ligne (Streamlit Community Cloud, gratuit)

Ce dossier est autonome : il contient l'interface (`app.py`), le code de prédiction
(`endorisk_model.py`), le modèle entraîné (`models/arms_model.joblib`) et la liste
des bibliothèques (`requirements.txt`). Rien d'autre n'est nécessaire.

## Étapes (environ 20 minutes, une seule fois)

1. **Créer un dépôt GitHub.** Sur github.com, « New repository », nom par exemple
   `endorisk-app`, visibilité **Private** (le code et le modèle restent à vous ;
   Streamlit Cloud sait lire les dépôts privés). Ne pas cocher « Add a README ».
2. **Envoyer les fichiers.** Deux façons :
   - sans git : sur la page du dépôt vide, « uploading an existing file », glisser
     tout le contenu de ce dossier (y compris le sous-dossier `models` et le dossier
     caché `.streamlit`), puis « Commit changes » ;
   - avec git, depuis ce dossier :
     ```
     git remote add origin https://github.com/<votre-compte>/endorisk-app.git
     git push -u origin main
     ```
     (le dépôt local est déjà initialisé et le premier commit est fait.)
3. **Créer l'application sur Streamlit.** Aller sur share.streamlit.io, se connecter
   avec le compte GitHub, « Create app » → « Deploy a public app from GitHub »,
   choisir le dépôt, branche `main`, fichier `app.py`. Laisser le reste par défaut.
4. **Attendre 2 à 3 minutes.** L'application se construit, puis une adresse du type
   `https://endorisk-app-xxxx.streamlit.app` apparaît. C'est ce lien qu'on envoie
   à Antoine, par WhatsApp ou autrement.

## Protéger l'accès (optionnel)

Par défaut le lien est public mais introuvable sans l'adresse exacte. Pour
restreindre aux seuls invités : dans Streamlit Cloud, « Settings » → « Sharing »,
passer en « Only specific people can view this app » et saisir les adresses e-mail
d'Antoine et la vôtre. Ils devront se connecter avec cette adresse.

## Mettre à jour

Remplacer les fichiers modifiés dans le dépôt (upload ou `git push`). Streamlit
redéploie tout seul en une minute. Le modèle se met à jour en remplaçant
`models/arms_model.joblib`.

## Si la construction échoue

Le cas le plus fréquent est une incompatibilité de version de scikit-learn avec le
modèle sauvegardé. Les versions dans `requirements.txt` sont exactement celles qui
ont servi à l'entraîner ; ne pas les changer. Les journaux de construction sont
visibles dans Streamlit Cloud via « Manage app » en bas à droite de la page.
