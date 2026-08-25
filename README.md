# Mesure et correction des biais algorithmiques dans les systèmes de recommandation basés sur les GNN

Mémoire de master — Clarisse Gradelet & Capucine Lettier

## Sujet

Une approche expérimentale de la mesure et de la correction des biais algorithmiques
dans les systèmes de recommandation basés sur les Graph Neural Networks (GNN),
appliquée au contexte des réseaux sociaux.

**Mots-clés :** Mesure · Propagation · Correction

## Problématique

Les biais dans les systèmes de recommandation ne sont pas isolés : ils interagissent
et se renforcent mutuellement au sein des architectures GNN, où le mécanisme de
*message passing* propage — et amplifie — ces biais à travers le graphe social.

L'objectif de ce travail est de :
1. **Identifier et taxonomiser** les biais présents dans les données et le modèle
   (taxonomie des 7 biais de Chen et al., 2023)
2. **Mesurer conjointement** ces biais afin de quantifier leurs corrélations
3. **Corriger** via une méthode IPS (Inverse Propensity Scoring) unifiée, intégrée
   à la loss function
4. **Évaluer le trade-off** équité / précision qui en résulte

## Dataset

**Yelp** — avis de commerces + graphe social entre utilisateurs (relation "amis").
Dataset de référence en SocialRS (Sharma et al., 2024), compatible directement
avec LightGCN.

## Modèle

**LightGCN** — architecture GNN de référence pour les systèmes de recommandation,
utilisée comme base d'expérimentation pour la mesure et la correction des biais.

## Structure du repo

```
gnn-bias-recsys/
├── data/                   # Données brutes et prétraitées (non versionnées, voir .gitignore)
├── notebooks/
│   └── 01_setup_lightgcn_yelp.ipynb   # Chargement des données, setup du modèle
├── src/
│   ├── data_loader.py       # Chargement et préparation du dataset Yelp
│   ├── model.py              # Wrapper LightGCN
│   ├── bias_metrics.py       # Métriques de biais (Phase 1 — mesure)
│   └── ips_correction.py     # Correction IPS unifiée (Phase 2 — correction)
├── results/                 # Résultats d'expériences (métriques, figures)
├── requirements.txt
└── README.md
```

## État d'avancement

- [x] Choix du dataset : Yelp (remplace LastFM, choix initial)
- [x] État de l'art structuré (Chizari et al. 2025, Jin et al. 2023, Chen et al. 2023, Sharma et al. 2024)
- [x] Environnement configuré (LightGCN)
- [ ] Phase 1 — Mesure des biais (données / modèle / résultats)
- [ ] Phase 2 — Correction via IPS unifiée
- [ ] Phase 3 — Validation (correction isolée vs. jointe, trade-off accuracy/fairness)

**Question ouverte actuelle :** quelle part de l'amplification des biais est héritée
des données, et quelle part est fabriquée par l'architecture GNN elle-même ?

## Références principales

- Chizari, N.; Tajfar, K.; Moreno-García, M.N. (2025). *Measuring Inter-Bias Effects
  and Fairness–Accuracy Trade-Offs in GNN-Based Recommender Systems*. Future Internet, 17, 461.
- Jin, D. et al. (2023). *A Survey on Fairness-aware Recommender Systems*. Information Fusion, 100.
- Chen, J.; Dong, H.; Wang, X. et al. (2023). *Bias and Debias in Recommender System:
  A Survey and Future Directions*. ACM TOIS, 41(3).
- Sharma, K.; Lee, Y.-C.; Nambi, S. et al. (2024). *A Survey of Graph Neural Networks
  for Social Recommender Systems*. ACM Computing Surveys.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook notebooks/01_setup_lightgcn_yelp.ipynb
```
