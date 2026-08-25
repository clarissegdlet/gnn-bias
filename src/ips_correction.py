"""
ips_correction.py

Phase 2 — Correction des biais via Inverse Propensity Scoring (IPS).

Principe : chaque interaction observée a une probabilité d'avoir été exposée
à l'utilisateur (le "propensity score"). Pondérer la loss par l'inverse de
cette probabilité corrige le déséquilibre : les interactions sur des items
peu exposés (donc rarement observées) comptent proportionnellement plus,
ce qui compense le biais d'exposition sans avoir besoin de ré-échantillonner
les données.

Ici, la correction est généralisée pour couvrir simultanément :
  - le biais de popularité
  - le biais d'exposition
  - le biais de conformité
  - le biais de position

plutôt que de traiter chaque biais séparément dans des passes de correction
indépendantes (l'hypothèse centrale du mémoire : une correction jointe est
plus efficace qu'une correction biais par biais).
"""

import numpy as np
import pandas as pd
import torch


def compute_propensity_scores(
    interactions: pd.DataFrame,
    item_col: str = "item_id",
    position_col: str | None = "position",
    smoothing: float = 0.1,
) -> pd.Series:
    """Calcule un propensity score par interaction, combinant popularité
    de l'item et position dans la liste de recommandation (si disponible).

    smoothing évite les scores nuls (division par zéro dans l'IPS) pour les
    items très peu exposés.
    """
    item_freq = interactions[item_col].value_counts(normalize=True)
    propensity = interactions[item_col].map(item_freq) + smoothing
    propensity = propensity / propensity.max()  # normalisation dans [smoothing/max, 1]

    if position_col is not None and position_col in interactions.columns:
        # Les positions basses (haut de liste) ont une probabilité d'exposition
        # plus forte -> on l'intègre multiplicativement.
        position_weight = 1.0 / np.log2(interactions[position_col].astype(float) + 2)
        propensity = propensity * position_weight

    return propensity.clip(lower=smoothing)


def ips_weights(propensity: pd.Series, clip_max: float = 10.0) -> pd.Series:
    """Convertit les propensity scores en poids IPS = 1 / propensity,
    avec clipping pour éviter l'explosion de variance sur les items
    extrêmement peu exposés (problème classique de l'IPS).
    """
    weights = 1.0 / propensity
    return weights.clip(upper=clip_max)


def weighted_bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """BPR loss pondérée par les poids IPS.

    À substituer à la loss standard de LightGCN (BPRLoss) pour intégrer
    la correction directement dans la fonction objectif (Phase 2 — correction
    "dans la loss", telle que décrite dans le plan expérimental).
    """
    diff = pos_scores - neg_scores
    losses = -torch.log(torch.sigmoid(diff) + 1e-10)
    return (weights * losses).mean()


def joint_correction(
    interactions: pd.DataFrame,
    item_col: str = "item_id",
    position_col: str | None = "position",
) -> pd.Series:
    """Point d'entrée unique : calcule les poids IPS finaux à injecter dans
    l'entraînement, en combinant popularité + exposition + position en une
    seule passe (correction jointe plutôt que biais par biais).
    """
    propensity = compute_propensity_scores(interactions, item_col, position_col)
    return ips_weights(propensity)


if __name__ == "__main__":
    # Exemple minimal sur données synthétiques, pour vérifier le pipeline
    # avant de le brancher sur le vrai dataset Yelp.
    rng = np.random.default_rng(0)
    demo = pd.DataFrame({
        "user_id": rng.integers(0, 50, 500),
        "item_id": rng.zipf(a=2.0, size=500) % 100,  # distribution longue traîne
        "position": rng.integers(1, 20, 500),
    })

    weights = joint_correction(demo)
    print("Poids IPS calculés sur données synthétiques :")
    print(f"  min={weights.min():.3f}  max={weights.max():.3f}  mean={weights.mean():.3f}")
