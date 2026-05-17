# Données

Ce dossier est intentionnellement vide dans le repo. Les fichiers de données ne sont pas versionnés (taille + droits Rakuten Challenge).

## Pour reproduire le projet

1. S'inscrire au [Rakuten France Multimodal Product Data Classification Challenge](https://challengedata.ens.fr/challenges/35)
2. Télécharger `X_train_update.csv` et `Y_train_CVw08PX.csv`
3. Placer les fichiers dans ce dossier `data/`
4. Lancer les notebooks dans l'ordre (`01_eda.ipynb`, `02_baseline.ipynb`, `03_embeddings.ipynb`)

Les fichiers intermédiaires (`data/processed/*.parquet`, `data/processed/embeddings.npy`) seront générés automatiquement.

### Note sur les embeddings (notebook 03)

Le notebook 03 nécessite l'encodage de ~85k textes via `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. En CPU, cela peut prendre plusieurs heures. Sur GPU (ex. Google Colab gratuit, T4), comptez ~3-4 minutes.