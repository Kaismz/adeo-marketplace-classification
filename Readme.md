# ADEO Marketplace — Classification automatique de produits

Projet personnel réalisé en préparation d'une candidature à l'alternance Data Scientist chez ADEO (M2, rentrée 2026).

L'idée : prendre un problème central d'un Data Scientist Marketplace — la classification automatique des produits dans la taxonomie catalogue — et le traiter sur un dataset proxy public, en allant aussi loin que possible avec des moyens raisonnables.

---

## Résultats

| Modèle | F1-macro (test) | Accuracy (test) | Temps train | Inférence |
|---|:---:|:---:|:---:|:---:|
| **Baseline TF-IDF + LogReg** | **0.796** | **0.806** | 40 s | < 1 ms / produit |
| Embeddings MiniLM multilingue | 0.634 | 0.670 | 140 s | < 1 ms / produit |

Le résultat principal n'est pas le score baseline (correct mais sans plus). C'est l'écart : **les embeddings perdent 16 points face au TF-IDF**, et de manière uniforme sur les 27 classes. Le pourquoi est ce qui m'a le plus appris sur ce dataset, et c'est ce que je détaille plus bas.

---

## Le dataset

Données : [Rakuten France Multimodal Product Data Classification](https://challengedata.ens.fr/challenges/35) (~85k produits, 27 catégories, marketplace française multi-vendeurs, multilingue de fait).

C'est ce qui se rapproche le plus d'un catalogue marketplace à vendeurs tiers comme ADEO, en accès public. La structure du problème — taxonomie marchande, saisie hétérogène entre vendeurs, descriptions parfois absentes, vocabulaire produit spécifique — est suffisamment proche pour que les conclusions soient transposables.

Splits : 70 / 15 / 15, stratifiés sur la classe, seed 42.

---

## Ce que l'EDA a révélé (et qui oriente tout le reste)

L'exploration (notebook 01) a fait remonter quatre choses qui ont pesé sur les choix de modélisation :

**1. Le déséquilibre des classes est modéré mais réel** (ratio 13× entre classe majoritaire et minoritaire). Ça suffit à rendre l'accuracy seule trompeuse. J'ai donc pris **F1-macro comme métrique principale** (moyenne non pondérée des F1 par classe), et `class_weight='balanced'` côté modèle.

**2. La description est absente dans 35 % des produits, mais cette absence est très hétérogène par classe.** Certaines catégories (presse, livres anciens, jeux vidéo) sont sans description à 89-97 %. D'autres (puériculture, électronique grand public) sont presque toujours renseignées. Conséquence directe : le fait que la description soit vide est en soi un signal discriminant, et le modèle doit être robuste à cette hétérogénéité.

**3. Le texte est bruité** : 28 % des descriptions contiennent du HTML brut, 64 % des entités HTML non décodées, et plusieurs templates vendeurs reviennent à l'identique avec des encodages cassés différents (`Caractéristique`, `Caracteristique`, `Caract?istique`). J'ai ajouté à ma pipeline une normalisation Unicode (NFKD + suppression des diacritiques) pour rendre le modèle robuste à ces variations, au prix de la perte de quelques nuances orthographiques.

**4. Le vocabulaire est très discriminant entre classes.** En regardant les top tokens de 6 catégories représentatives, je peux les identifier sans hésiter (`piscine, filtration, m3` → piscines ; `lot, tomes, partitions, revues` → livres en lots). C'est un signal fort que TF-IDF va bien fonctionner — et a posteriori c'est ce qui a permis d'anticiper plusieurs confusions du modèle dès l'EDA.

---

## La baseline (notebook 02)

TF-IDF avec unigrammes et bigrammes (50 000 features, `min_df=5`, `sublinear_tf=True`), puis régression logistique avec `liblinear` et `class_weight='balanced'`. Choix volontairement simple : avant d'aller chercher de la complexité, je voulais un point de comparaison crédible.

**Test : F1-macro 0.796, accuracy 0.806, écart val ↔ test de 0.16 pt.** L'écart insignifiant entre validation et test confirme que les hyperparamètres n'ont pas été ajustés par tâtonnement sur la validation.

Quelques observations qui me semblent utiles :

- La performance varie énormément selon la classe (F1 de 0.55 à 0.97). Et l'allure suit ce qu'on avait vu en EDA : les classes au texte riche et au vocabulaire spécifique sont très bien prédites (piscines à 0.96), celles au texte court avec description souvent vide sont les plus dures (livres d'occasion à 0.55).
- La classe "livres d'occasion" se comporte comme un aimant à faux positifs : sa précision est de 0.46 parce que 6 autres classes lui envoient une partie de leurs erreurs. Quand le modèle hésite face à un titre court et un texte vide à connotation livresque, il choisit cette classe par défaut.
- Les deux confusions que j'avais anticipées en EDA (livres en lots ↔ livres d'occasion, puériculture ↔ jouets) se sont confirmées partiellement : la première à 10 %, la seconde à seulement 3-4 % (le vocabulaire technique de la puériculture — `poussette, siege, securite` — sépare mieux que je ne l'avais pensé). C'est la partie où j'ai appris qu'une intuition d'EDA, même appuyée, peut être sur-estimée ; il faut la confronter aux chiffres.

### Trois typologies d'erreurs distinctes

En regardant des exemples concrets de produits mal classés sur les paires confuses, j'ai identifié trois types d'erreurs très différents, qui n'appellent pas les mêmes solutions :

| Type | Exemple | Ce qui résoudrait |
|---|---|---|
| Ambiguïté ontologique | Jeu d'échecs en bois pour enfant : 1280 ou 1281 ? Un humain hésiterait aussi. | Image, ou refonte de la taxonomie source. |
| Description vide, signal latent dans le titre | "Venus Wa Katamoi - Tome 1 à 5" classé comme livre individuel au lieu de lot. Le modèle ne capte pas le pattern "Tome X à Y". | Features regex de détection de lot, ou bigrammes plus longs. Le moins coûteux. |
| Description vide, aucun signal disponible | "Le Roi Et L'architecte" — impossible de distinguer un livre neuf d'un livre d'occasion sur ce seul titre. | Ne se résoudra pas par du texte seul. Il faut du multimodal (prix, image, vendeur). |

C'est la partie de l'analyse que je trouve la plus utile pour penser une V2. Tous les modèles ne sont pas équipés pour le même type d'erreurs.

---

## L'expérience embeddings (notebook 03)

L'hypothèse était : un modèle pré-entraîné sur du texte multilingue (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions, encodé sur GPU T4 via Colab en 3,4 min) devrait au moins égaler le TF-IDF, et probablement faire mieux sur les classes où le vocabulaire est partagé.

Résultat : **0.634 F1-macro en test, soit 16 points en dessous de la baseline. Et la perte est uniforme : la baseline gagne sur les 27 classes**, sans exception.

Quelques hypothèses, par ordre décroissant de plausibilité d'après moi :

D'abord, **j'ai vraisemblablement saboté MiniLM par le preprocessing.** Pour TF-IDF, il est utile d'enlever la ponctuation, de mettre en minuscules, de normaliser les accents. Mais MiniLM est pré-entraîné sur du texte naturel avec sa ponctuation et sa casse. En lui donnant le même `text_clean` qu'à TF-IDF, je l'ai privé d'une partie du signal contextuel sur lequel il sait s'appuyer. Une V2 logique : ré-encoder à partir de `text` brut (titre + description sans nettoyage). C'est probablement la première chose à tester.

Ensuite, **le dataset joue contre les embeddings.** Sur 85k textes, environ un tiers est juste un titre court de 8-12 mots, souvent un patchwork de mots-clés ("Star Wars 1p/ Jean Pierre Michael Ris..."). Ce n'est pas du texte rédactionnel, c'est le terrain naturel de TF-IDF. À l'inverse, les transformers brillent sur des phrases naturelles, des questions-réponses, des comparaisons sémantiques.

Enfin, **TF-IDF a une vraie supériorité structurelle ici**. Avec 50 000 features sparses, il peut capter des tokens très spécifiques (`piscine`, `partitions`, `m3`) avec une précision parfaite. MiniLM doit compresser tout le sens dans 384 dimensions denses — un goulot d'étranglement quand le signal discriminant est purement lexical.

Conclusion pratique : sur ce problème, le surcoût d'un modèle pré-entraîné n'est pas justifié. La baseline est 130× plus légère en mémoire (10 Mo vs 0,08 Mo) et fait mieux. C'est exactement le genre d'arbitrage qu'un DS marketplace doit savoir trancher : ne pas mettre du BERT là où une régression logistique suffit.

---

## Ce qui serait à creuser ensuite

- **Ré-encoder MiniLM sans nettoyage agressif.** Le test le plus rentable et le plus rapide.
- **Ajouter des features manuelles** sur les patterns détectables dans les titres (`Tome \d+ à \d+`, présence de `+` séparateurs, etc.). Ça corrigerait sans doute une fraction notable de la confusion 2403 → 10.
- **Approche multimodale.** Le dataset contient un `imageid` que je n'ai pas utilisé. Pour les classes où le texte est intrinsèquement insuffisant (livre neuf vs occasion), c'est probablement la seule voie.
- **Fine-tuner un CamemBERT ou XLM-RoBERTa** sur les données. C'est ce qui permettrait probablement de dépasser 0.85 F1-macro, mais c'est aussi un projet d'une autre échelle (compute, temps d'entraînement, complexité de mise en prod).

---

## Lien avec le contexte ADEO

Plusieurs constats du projet me semblent transférables à une marketplace comme celle de Leroy Merlin :

- La marketplace est multilingue de fait dès qu'on opère dans plusieurs pays, ce qui contraint le choix du modèle dès le départ.
- Les vendeurs tiers ne remplissent pas tous les champs de la même manière. Une part significative du catalogue arrivera toujours avec une description minimale ou absente, et le modèle doit le gérer.
- Sur certaines sous-catégories proches (typiquement les déclinaisons fines d'un même type de produit), la classification texte-only atteindra un plafond. À ce moment-là, ajouter de l'image, du prix ou de la métadonnée vendeur est probablement plus rentable que d'aller chercher un modèle texte plus gros.
- Une baseline simple peut être étonnamment difficile à battre quand le vocabulaire métier est riche et discriminant. C'est un argument pour ne pas surinvestir dans la complexité avant d'avoir établi une vraie référence.

---

## Reproduction

```bash
git clone https://github.com/Kaismz/adeo-marketplace-classification.git
cd adeo-marketplace-classification
conda create -n projet_ML python=3.11 -y
conda activate projet_ML
pip install -r requirements.txt
```

Les données du Rakuten Data Challenge ne sont pas redistribuées (droits ENS). Procédure dans `Data/Readme.md`.

Ensuite, lancer les notebooks dans l'ordre 01 → 02 → 03. Le notebook 03 demande un GPU pour l'encodage (~3 min sur Colab gratuit, plusieurs heures en CPU).

---

## Structure

```
.
├── Notebooks/
│   ├── 01_eda.ipynb              # EDA orientée business
│   ├── 02_baseline.ipynb         # TF-IDF + LogReg
│   └── 03_embeddings.ipynb       # MiniLM + LogReg
├── models/
│   ├── *_metrics.json            # Métriques détaillées (versionnées)
│   ├── *_predictions.json        # Prédictions val + test (versionnées)
│   └── *.joblib                  # Modèles entraînés (locaux uniquement)
├── Data/
│   └── Readme.md                 # Procédure de récupération
├── requirements.txt
└── .gitignore
```

---

Kaïs Mazari — M1 Mathématiques & Applications, parcours Ingénierie Statistique et Numérique, Université de Lille.

[LinkedIn](https://linkedin.com/in/kaïs-mazari-915b59384) · [kais.mazari.etu@univ-lille.fr](mailto:kais.mazari.etu@univ-lille.fr)
