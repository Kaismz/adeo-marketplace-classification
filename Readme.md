# ADEO Marketplace — Classification automatique de produits

> Projet personnel — Étude de cas appliquée au métier de Data Scientist Marketplace chez ADEO.
> Réalisé en candidatant à l'alternance Data Scientist (M2, 2026-2027).

---

## TL;DR

| Modèle | F1-macro (test) | Accuracy (test) | Temps train | Inférence |
|---|:---:|:---:|:---:|:---:|
| **Baseline TF-IDF + LogReg** | **0.796** | **0.806** | 40 s | < 1 ms / produit |
| Embeddings multilingues (MiniLM) | 0.634 | 0.670 | 140 s | < 1 ms / produit |

**Conclusion principale** : sur ce dataset (textes courts, vocabulaire métier ultra-spécifique, 35 % de descriptions manquantes), la baseline TF-IDF surpasse une approche embeddings sémantiques de **+16 pts F1-macro**. Le surcoût d'un modèle dense pré-entraîné ne se justifie pas. Cette conclusion contre-intuitive est expliquée et documentée.

---

## Contexte et motivation

ADEO Marketplace, comme toute marketplace à vendeurs tiers, fait face à un problème central : **classer automatiquement les produits déposés par les vendeurs dans la bonne catégorie de l'arborescence catalogue**. Une mauvaise classification = produit invisible = CA perdu.

N'ayant pas accès aux données ADEO, j'ai pris comme proxy le dataset public le plus proche : **Rakuten France Multimodal Product Data Classification** (~85k produits, 27 catégories, multilingue FR/DE/EN, données réelles de marketplace française). La structure du problème — taxonomie marchande, hétérogénéité de saisie vendeur, multilingue — est quasi-identique à celle d'ADEO.

L'objectif n'est pas de battre l'état de l'art (les meilleurs modèles du challenge plafonnent à ~0.86 F1-macro avec BERT fine-tuné + multimodal). C'est de **démontrer comment un Data Scientist Marketplace attaquerait ce problème** : EDA orientée business, baseline solide et déployable, analyse d'erreurs structurée, et test honnête d'une hypothèse populaire (embeddings sémantiques).

---

## Approche

### Données
- **Source** : [Rakuten Data Challenge — ENS](https://challengedata.ens.fr/challenges/35)
- **Volume** : 84 916 produits, 27 catégories
- **Features texte** : `designation` (titre court) + `description` (longue, **35 % manquante**)
- **Splits** : 70 / 15 / 15 stratifiés sur la classe (RANDOM_STATE=42)

### Pipeline
1. **EDA** (`notebooks/01_eda.ipynb`) : exploration, identification des enjeux qualité, dégagement d'hypothèses business.
2. **Baseline** (`notebooks/02_baseline.ipynb`) : TF-IDF (1-2grammes, 50k features, sublinear_tf) + LogReg (`liblinear`, `class_weight='balanced'`).
3. **Comparaison embeddings** (`notebooks/03_embeddings.ipynb`) : `paraphrase-multilingual-MiniLM-L12-v2` (384 dims, encodé sur GPU T4 Colab en 3.4 min) + LogReg identique.

### Métrique principale
**F1-macro** (équipondère les 27 classes), motivée par le déséquilibre classes (ratio 13× entre max et min). Accuracy reportée en complément.

---

## Résultats clés

### Performance par classe (validation)

La baseline réussit très bien sur les classes au **vocabulaire ultra-discriminant** (piscines à 0.96, papeterie à 0.93, livres "neufs" à 0.97), et trébuche sur le **pôle "livres / presse / occasion"** où plusieurs catégories partagent un vocabulaire commun (livre, tome, edition).

### Confusions principales identifiées

Les confusions principales avaient été **anticipées dès l'EDA** (top tokens chevauchants entre classes) et confirmées par la matrice de confusion :

- **Pôle "livres"** : classes 10 (occasion), 2403 (lots), 2705 (essais), 2280 (presse) — confondues entre elles à 10-15 %. La classe 10 absorbe des faux positifs venant de 6 autres classes, ce qui plombe sa précision (0.46).
- **Pôle "jouets / puériculture"** : confusion 1281 → 1280 à 17 %, due à une ambiguïté ontologique réelle (jeux d'enfants en bois, jouets musicaux, etc. appartiennent objectivement aux deux sous-catégories).

### Analyse d'erreurs : 3 typologies distinctes

| Type | Exemple | Résolvable comment ? |
|---|---|---|
| Ambiguïté ontologique | 1281 ↔ 1280 (jouets / jouets) | Images, refonte taxonomie |
| Desc. vide + signal latent dans le titre | 2403 → 10 ("Tome 1 à 5", "+ X + Y + Z") | Features regex de détection de lot |
| Desc. vide sans signal disponible | 2705 → 10 (livres neufs vs occasion) | Modèle multimodal (titre + image + métadonnées vendeur) |

---

## L'hypothèse embeddings (et son infirmation)

Hypothèse classique : *"des embeddings sémantiques pré-entraînés (sentence-transformers) feront mieux qu'un TF-IDF sur ce problème de classification multilingue"*.

Résultat empirique : **F1-macro 0.634 vs 0.796 pour la baseline (-16 pts), perte uniforme sur 27/27 classes**.

**Diagnostic** :
- **Texte court et bruité** : 35 % de descriptions vides, titres souvent listes de mots-clés ("Star Wars 1p/ Jean Pierre Michael Ris..."). Pas le terrain de jeu naturel des transformers, pré-entraînés sur du texte rédactionnel.
- **Vocabulaire ultra-discriminant** : TF-IDF avec 50k features capture exactement ces tokens spécifiques. MiniLM doit compresser dans 384 dims — perte inévitable.
- **Préprocessing agressif désavantage MiniLM** : lowercase + suppression accents + suppression ponctuation, optimal pour TF-IDF, détruit le signal contextuel sur lequel MiniLM est entraîné.

**Implication business** : sur une marketplace réelle, la complexité supplémentaire d'un modèle dense doit se justifier par un gain mesurable. Ici, **non**. La baseline est 130× plus légère en mémoire et tout aussi rapide en inférence.

---

## Transposition au contexte ADEO

| Observation Rakuten | Implication ADEO |
|---|---|
| Multilingue (FR + DE + EN) | ADEO opère dans 13 pays → modèle multilingue indispensable |
| 35 % de descriptions vides, hétérogène par classe | Marketplace à vendeurs tiers : robustesse au texte manquant requise |
| Confusions sur catégories proches (sous-pôles) | Sur Leroy Merlin : confusion attendue entre catégories proches (ex. "Vis bois" vs "Vis métal") → revue de taxonomie + multimodal |
| HTML brut + encodages cassés dans 28-63 % des descriptions | Pipeline de nettoyage texte indispensable en prod |
| Modèle baseline déjà déployable (< 1 ms par produit) | Latence compatible avec l'ajout temps réel de produits par les vendeurs |

---

## Limites et pistes V3

- **Préprocessing destructif pour les embeddings** : re-encoder le texte brut (sans nettoyage) permettrait potentiellement de récupérer une partie du delta.
- **Modèle texte-only** : pour les classes type "livre neuf vs livre d'occasion", aucun gain possible sans features auxiliaires (prix, image, métadonnées vendeur).
- **Approche multimodale** : ajout des images produit (`imageid` du dataset, non utilisé ici) → probable gain sur les classes ambiguës sémantiquement.
- **Fine-tuning** d'un CamemBERT ou XLM-RoBERTa sur les données : approche lourde mais probablement la voie vers > 0.85 F1-macro.

---

## Reproduction

### Prérequis
- Python 3.11
- ~2 Go d'espace disque (modèle d'embeddings + données)

### Installation
```bash
git clone https://github.com/Kaismz/adeo-marketplace-classification.git
cd adeo-marketplace-classification
conda create -n projet_ML python=3.11 -y
conda activate projet_ML
pip install -r requirements.txt
```

### Données
Voir [`Data/Readme.md`](./Data/Readme.md). Les données du Rakuten Data Challenge ne sont pas redistribuées ici (droits ENS).

### Exécution
Lancer les notebooks dans l'ordre :
1. `Notebooks/01_eda.ipynb`
2. `Notebooks/02_baseline.ipynb`
3. `Notebooks/03_embeddings.ipynb` (encodage GPU recommandé via Colab — instructions dans le notebook)

---

## Structure du projet

```
.
├── Notebooks/
│   ├── 01_eda.ipynb              # EDA orientée business
│   ├── 02_baseline.ipynb         # TF-IDF + LogReg (F1-macro 0.796)
│   └── 03_embeddings.ipynb       # MiniLM + LogReg (comparaison)
├── models/
│   ├── *_metrics.json            # Métriques détaillées (versionnées)
│   ├── *_predictions.json        # Prédictions val + test (versionnées)
│   └── *.joblib                  # Modèles entraînés (locaux uniquement)
├── Data/
│   ├── Readme.md                 # Instructions de récupération
│   └── processed/                # Fichiers intermédiaires (locaux)
├── requirements.txt
└── .gitignore
```

---

## À propos

**Kaïs Mazari** — M1 Mathématiques & Applications (Université de Lille), parcours Ingénierie Statistique et Numérique (Data Science).

Projet réalisé dans le cadre d'une candidature à l'alternance Data Scientist chez ADEO (M2, septembre 2026).

- LinkedIn : [linkedin.com/in/kaïs-mazari-915b59384](https://linkedin.com/in/kaïs-mazari-915b59384)
- Email : kais.mazari.etu@univ-lille.fr
