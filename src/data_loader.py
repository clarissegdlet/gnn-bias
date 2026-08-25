"""
data_loader.py

Chargement et préparation du dataset Yelp pour l'entraînement LightGCN.
Le dataset combine :
  - les interactions utilisateur-item (avis / ratings)
  - le graphe social entre utilisateurs (relation "amis")

Format attendu (convention RecBole) :
  yelp.inter  -> user_id, item_id, rating, timestamp
  yelp.user   -> user_id, friends (liste d'ids séparés par des virgules)
  yelp.item   -> item_id, categories, ...
"""

import pandas as pd
import numpy as np
from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "yelp"


def load_interactions(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Charge les interactions utilisateur-item (avis Yelp)."""
    path = data_dir / "yelp.inter"
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]  # nettoyage format RecBole
    return df


def load_social_graph(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Charge le graphe social (relations 'amis' entre utilisateurs).

    Retourne une arête par ligne : user_id, friend_id
    """
    path = data_dir / "yelp.user"
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]

    edges = []
    for _, row in df.iterrows():
        if pd.isna(row.get("friends")):
            continue
        friends = str(row["friends"]).split(",")
        for f in friends:
            f = f.strip()
            if f:
                edges.append((row["user_id"], f))

    return pd.DataFrame(edges, columns=["user_id", "friend_id"])


def compute_item_popularity(interactions: pd.DataFrame) -> pd.Series:
    """Popularité brute de chaque item = nombre d'interactions reçues.

    Sert de base à la mesure du popularity bias (Phase 1) et au calcul
    des propensity scores pour la correction IPS (Phase 2).
    """
    return interactions.groupby("item_id").size().sort_values(ascending=False)


def train_test_split(
    interactions: pd.DataFrame,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split simple par utilisateur (leave-ratio-out), stratifié par user_id
    pour garantir que chaque utilisateur est présent dans train et test.
    """
    rng = np.random.default_rng(seed)
    test_idx = []

    for _, group in interactions.groupby("user_id"):
        n_test = max(1, int(len(group) * test_ratio))
        idx = rng.choice(group.index, size=n_test, replace=False)
        test_idx.extend(idx)

    test_df = interactions.loc[test_idx]
    train_df = interactions.drop(index=test_idx)
    return train_df, test_df


if __name__ == "__main__":
    inter = load_interactions()
    pop = compute_item_popularity(inter)
    print(f"{len(inter)} interactions chargées")
    print(f"Item le plus populaire : {pop.index[0]} ({pop.iloc[0]} interactions)")
