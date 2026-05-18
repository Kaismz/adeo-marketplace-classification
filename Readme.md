# ADEO Marketplace — Classification automatique de produits

Projet personnel réalisé en préparation d'une candidature à l'alternance Data Scientist chez ADEO (M2, rentrée 2026).

## Le problème

Une marketplace comme celle de Leroy Merlin (filiale ADEO) hébèrge des millions de produits déposés par des milliers de vendeurs tiers. Chaque produit doit être rangé dans la bonne catégorie de l'arborescence catalogue ("Outillage > Électroportatif > Perceuses", par exemple), sinon il devient invisible dans la navigation et le moteur de recherche.

Demander aux vendeurs de classer eux-mêmes leurs produits dans une taxonomie de plusieurs milliers de catégories ne marche pas en pratique :

- Les vendeurs ne connaissent pas l'arborescence interne de la plateforme
- Ils choisissent la catégorie la plus large pour ne pas se tromper
- Ils saisissent peu ou pas de description
- Ils utilisent un vocabulaire hétérogène d'un vendeur à l'autre

D'où l'enjeu : **classer automatiquement les produits dans la bonne catégorie, à partir des données texte fournies par le vendeur (titre + description).**

C'est l'un des sujets ML récurrents d'un Data Scientist Marketplace, au même titre que la détection de doublons, la détection d'anomalies de prix, ou le ranking de la recherche.

## Pourquoi c'est non trivial

Trois difficultés caractéristiques apparaissent dès qu'on regarde un vrai catalogue marketplace :

1. **Texte court et hétérogène.** Un titre fait souvent 8-12 mots, parfois une simple liste de mots-clés. Les descriptions, quand elles existent, vont de quelques lignes à des notices complètes.
2. **Données manquantes massives.** Une part significative des produits (souvent un tiers ou plus) arrive sans description.
3. **Catégories proches.** Beaucoup de sous-catégories partagent un vocabulaire commun (livres en lots vs livres d'occasion, jouets pour bébés vs puériculture, vis bois vs vis métal). Distinguer ces voisines est la vraie difficulté.

N'ayant pas accès au catalogue ADEO, ce projet attaque le problème sur le dataset public le plus proche en termes de structure : **Rakuten France Multimodal Product Data Classification** (~85 000 produits, 27 catégories, marketplace française multilingue de fait). Les trois difficultés ci-dessus s'y retrouvent à l'identique.

---

## Résultats

[... la suite reste identique]

## Résultats

| Modèle | F1-macro (test) | Accuracy (test) | Temps train | Inférence |
|---|:---:|:---:|:---:|:---:|
| **Baseline TF-IDF + LogReg** | **0.796** | **0.806** | 40 s | < 1 ms / produit |
| Embeddings MiniLM multilingue | 0.634 | 0.670 | 140 s | < 1 ms / produit |

Le résultat principal n'est pas le score baseline (correct mais sans plus). C'est l'écart : **les embeddings perdent 16 points face au TF-IDF**, uniformément sur les 27 classes. Les sections suivantes détaillent pourquoi.

---

## Le dataset

Données : [Rakuten France Multimodal Product Data Classification](https://challengedata.ens.fr/challenges/35).

- ~85 000 produits, 27 catégories
- Marketplace française avec vendeurs tiers
- Multilingue de fait (FR + DE + EN)
- Features texte : `designation` (titre court) + `description` (souvent vide)

C'est le dataset public qui se rapproche le plus d'un catalogue marketplace à vendeurs tiers comme ADEO. La structure du problème — taxonomie marchande, saisie hétérogène entre vendeurs, vocabulaire produit spécifique — rend les conclusions transposables.

Splits : 70 / 15 / 15, stratifiés sur la classe, seed 42.

---

## 01 — Exploratory Data Analysis

L'exploration a fait remonter quatre constats qui ont orienté tous les choix de modélisation.

**Déséquilibre modéré entre classes.** Ratio 13× entre la classe majoritaire (piscines, ~12 % du dataset) et la minoritaire (~0,9 %). Suffisant pour que l'accuracy seule soit trompeuse. Métrique principale retenue : **F1-macro** (moyenne non pondérée des F1 par classe).

**Description manquante : 35 % en moyenne, mais hétérogène par classe.** Certaines catégories (presse, livres anciens, jeux vidéo) sont sans description à 89-97 %. D'autres sont presque toujours renseignées. Le fait que la description soit vide constitue en soi un signal discriminant.

**Texte bruité.** 28 % des descriptions contiennent du HTML brut, 64 % des entités HTML non décodées, et plusieurs templates vendeurs reviennent à l'identique avec des encodages cassés différents. La pipeline de nettoyage inclut une normalisation Unicode (NFKD + suppression des diacritiques) pour rendre le modèle robuste à ces variations.

**Vocabulaire très discriminant entre classes.** Un coup d'œil sur les top tokens de 6 catégories représentatives permet de les identifier sans hésiter (`piscine, filtration, m3` → piscines ; `lot, tomes, partitions, revues` → livres en lots). Signal fort que TF-IDF est bien adapté au problème.

---

## 02 — Baseline TF-IDF + LogReg

### Configuration

- TF-IDF : unigrammes + bigrammes, 50 000 features, `min_df=5`, `sublinear_tf=True`
- LogReg : solveur `liblinear`, `class_weight='balanced'`, `C=1.0`
- Convergence en 12 itérations sur 300 autorisées

### Performance

**Test : F1-macro 0.796, accuracy 0.806.**

Écart val ↔ test de 0,16 pt : les hyperparamètres n'ont pas été ajustés par tâtonnement sur la validation, le score test est fiable.

### Performance par classe

Très variable selon la classe (F1 de 0,55 à 0,97). L'allure suit ce qu'avait montré l'EDA :

- Classes au texte riche et au vocabulaire spécifique → très bien prédites (piscines à 0,96, papeterie à 0,93)
- Classes au texte court avec description vide → les plus dures (livres d'occasion à 0,55)

### Confusions principales

Deux confusions avaient été anticipées en EDA via le chevauchement de vocabulaire entre classes. Une seule s'est vraiment confirmée :

- **Livres en lots ↔ livres d'occasion** : 10 % de confusion, validée
- **Puériculture ↔ jouets** : 3-4 % seulement, plus faible qu'attendu (le vocabulaire technique de la puériculture — `poussette, siege, securite` — sépare mieux que prévu)

La vraie confusion principale du modèle, non anticipée : **les sous-catégories de jouets entre elles** (17 % de confusion 1281 → 1280).

### La classe "livres d'occasion" comme aimant à faux positifs

Sa précision n'est que de 0,46. En cause : 6 autres classes lui envoient une partie de leurs erreurs. Face à un titre court et une description vide à connotation livresque, le modèle choisit cette classe par défaut.

### Trois typologies d'erreurs distinctes

L'analyse de produits mal classés a fait émerger trois types d'erreurs qui n'appellent pas les mêmes solutions :

| Type | Exemple | Ce qui résoudrait |
|---|---|---|
| **Ambiguïté ontologique** | Jeu d'échecs en bois pour enfant : 1280 ou 1281 ? Un humain hésiterait aussi. | Image, ou refonte de la taxonomie. |
| **Description vide, signal latent dans le titre** | "Venus Wa Katamoi - Tome 1 à 5" classé comme livre individuel. Le pattern "Tome X à Y" n'est pas capté. | Features regex de détection de lot. Solution la moins coûteuse. |
| **Description vide, aucun signal disponible** | "Le Roi Et L'architecte" — distinguer un livre neuf d'un livre d'occasion sur ce seul titre est impossible. | Modèle multimodal (prix, image, métadonnées vendeur). |

---

## 03 — Embeddings MiniLM

### Configuration

- Modèle : `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims, multilingue)
- Encodage : GPU T4 Colab, 3,4 min pour 85k textes
- Classifieur : LogReg identique à la baseline (comparaison équitable)

### Résultat

**Test : F1-macro 0,634, soit -16 points face à la baseline.**

Et la perte est uniforme : **la baseline gagne sur les 27 classes**, sans exception. Aucune catégorie où les embeddings apportent un gain.

### Trois explications, par ordre de plausibilité

**1. Preprocessing inadapté aux transformers.** Le nettoyage agressif (lowercase, suppression des accents et de la ponctuation) est optimal pour TF-IDF mais détruit le signal contextuel sur lequel MiniLM s'appuie. MiniLM a été pré-entraîné sur du texte naturel avec sa casse et sa ponctuation. Le réencoder à partir de `text` brut serait la première chose à tester.

**2. Dataset défavorable.** Un tiers du corpus est juste un titre court de 8-12 mots, souvent un patchwork de mots-clés ("Star Wars 1p/ Jean Pierre Michael Ris..."). Ce n'est pas du texte rédactionnel ; c'est le terrain naturel de TF-IDF, pas celui des transformers.

**3. Supériorité structurelle de TF-IDF ici.** Avec 50 000 features sparses, TF-IDF capte des tokens très spécifiques (`piscine`, `partitions`, `m3`) avec une précision parfaite. MiniLM doit compresser tout le sens dans 384 dimensions denses — goulot d'étranglement quand le signal discriminant est purement lexical.

### Implication

Sur ce problème, le surcoût d'un modèle pré-entraîné n'est pas justifié. La baseline est 130× plus légère en mémoire (10 Mo vs 0,08 Mo) pour le modèle, et fait mieux. Type d'arbitrage central pour un DS marketplace : ne pas mettre du BERT là où une régression logistique suffit.

---

## Pistes V3

- **Ré-encoder MiniLM à partir du texte brut** (sans nettoyage). Test le plus rentable et le plus rapide.
- **Features manuelles** sur les patterns détectables dans les titres (`Tome \d+ à \d+`, séparateurs `+`, énumérations). Corrigerait probablement une fraction notable de la confusion 2403 → 10.
- **Approche multimodale.** Le dataset contient un `imageid` non exploité ici. Pour les classes où le texte est intrinsèquement insuffisant (livre neuf vs occasion), c'est probablement la seule voie.
- **Fine-tuning d'un CamemBERT ou XLM-RoBERTa.** Vraisemblablement la voie vers > 0,85 F1-macro, mais autre échelle de projet (compute, mise en prod).

---

## Lien avec le contexte ADEO

Plusieurs constats du projet semblent transférables à une marketplace comme celle de Leroy Merlin :

- La marketplace est multilingue de fait dès qu'on opère dans plusieurs pays, ce qui contraint le choix du modèle dès le départ.
- Les vendeurs tiers ne remplissent pas tous les champs de la même manière. Une part significative du catalogue arrivera toujours avec une description minimale ou absente.
- Sur certaines sous-catégories proches, la classification texte-only atteindra un plafond. Ajouter de l'image, du prix ou de la métadonnée vendeur est alors plus rentable que de pousser un modèle texte plus gros.
- Une baseline simple peut être étonnamment difficile à battre quand le vocabulaire métier est riche et discriminant. Argument pour ne pas surinvestir dans la complexité avant d'avoir établi une référence.

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

Lancer les notebooks dans l'ordre : 01 → 02 → 03. Le notebook 03 demande un GPU pour l'encodage (3 min sur Colab gratuit, plusieurs heures en CPU).

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
