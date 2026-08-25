"""
bias_metrics.py

Phase 1 — Mesure des biais.

Implémente des métriques de biais aux trois niveaux définis dans le plan
expérimental :
  - Niveau 1 : dans les données (avant entraînement)
  - Niveau 2 : dans le modèle (embeddings)
  - Niveau 3 : dans les résultats (top-k recommandé)

Chaque fonction retourne un scalaire ou une série comparable, de façon à
pouvoir ensuite calculer les corrélations entre biais (l'objectif central
de la Phase 1 : montrer qu'ils s'alimentent mutuellement).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Niveau 1 — Biais dans les données
# ---------------------------------------------------------------------------

def gini_coefficient(counts: pd.Series) -> float:
    """Coefficient de Gini sur la distribution des interactions par item.

    Mesure la concentration : proche de 0 = distribution équitable,
    proche de 1 = quelques items concentrent presque toutes les interactions.
    Sert de proxy direct au popularity bias.
    """
    values = np.sort(counts.values.astype(float))
    n = len(values)
    if n == 0:
        return 0.0
    cumulative = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n + 1)) * values) - (n + 1) * cumulative[-1]) / (n * cumulative[-1])


def catalog_coverage(recommended_items: set, catalog_items: set) -> float:
    """Proportion du catalogue effectivement recommandée au moins une fois.

    Une faible couverture indique un biais de popularité fort : le modèle
    ne recommande qu'une fraction restreinte des items disponibles.
    """
    if not catalog_items:
        return 0.0
    return len(recommended_items & catalog_items) / len(catalog_items)


def exposure_bias(interactions: pd.DataFrame, item_col: str = "item_id") -> pd.Series:
    """Distribution d'exposition par item = fréquence normalisée d'apparition.

    Base pour le calcul des propensity scores utilisés en Phase 2 (IPS).
    """
    counts = interactions[item_col].value_counts()
    return counts / counts.sum()


# ---------------------------------------------------------------------------
# Niveau 2 — Biais dans le modèle (embeddings)
# ---------------------------------------------------------------------------

def embedding_norm_bias(embeddings: np.ndarray, popularity: np.ndarray) -> float:
    """Corrélation entre la norme des embeddings appris et la popularité des items.

    Une corrélation forte suggère que le modèle encode la popularité dans la
    géométrie de l'espace latent plutôt que dans la pertinence réelle —
    un symptôme classique de popularity bias amplifié par le message passing.
    """
    norms = np.linalg.norm(embeddings, axis=1)
    return float(np.corrcoef(norms, popularity)[0, 1])


def sensitivity_test(model_predict_fn, base_input: dict, perturbed_attribute: str, values: list) -> pd.DataFrame:
    """Teste la sensibilité des prédictions à un attribut donné.

    On fait varier `perturbed_attribute` sur les valeurs fournies, en gardant
    tout le reste constant, et on observe la variation du score prédit.
    Une forte variation pour un attribut qui ne devrait pas influencer la
    pertinence (ex. l'ancienneté du compte utilisateur) révèle un biais
    encodé dans le modèle.
    """
    results = []
    for v in values:
        perturbed = dict(base_input)
        perturbed[perturbed_attribute] = v
        score = model_predict_fn(perturbed)
        results.append({"value": v, "score": score})
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Niveau 3 — Biais dans les résultats (top-k)
# ---------------------------------------------------------------------------

def topk_exposure_shift(topk_recommendations: pd.DataFrame, original_popularity: pd.Series, k: int = 10) -> float:
    """Compare la distribution de popularité des items recommandés en top-k
    à leur distribution de popularité d'origine dans les données.

    Un ratio > 1 indique une amplification du biais de popularité par le
    modèle : les items déjà populaires sont sur-représentés dans le top-k
    par rapport à leur poids initial dans les données.
    """
    topk_items = topk_recommendations.groupby("user_id").head(k)["item_id"]
    topk_pop_share = topk_items.map(original_popularity).mean()
    baseline_pop_share = original_popularity.mean()
    if baseline_pop_share == 0:
        return float("nan")
    return topk_pop_share / baseline_pop_share


# ---------------------------------------------------------------------------
# Corrélations entre biais — cœur de la contribution Phase 1
# ---------------------------------------------------------------------------

def bias_correlation_matrix(bias_scores: dict[str, pd.Series]) -> pd.DataFrame:
    """Construit la matrice de corrélation entre plusieurs séries de scores
    de biais (un par type de biais, alignés sur le même index d'items ou
    d'utilisateurs).

    C'est cette matrice qui permet de quantifier empiriquement l'intuition
    de départ : les biais ne sont pas isolés, ils se renforcent mutuellement.
    """
    df = pd.DataFrame(bias_scores)
    return df.corr(method="spearman")
