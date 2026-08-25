"""
model.py

Wrapper autour de LightGCN (via RecBole) pour l'entraînement sur Yelp.

LightGCN simplifie le message passing des GNN classiques en retirant les
transformations non-linéaires et les poids appris à chaque couche : seule
l'agrégation par moyenne des voisins est conservée. C'est ce mécanisme de
propagation qu'on va instrumenter en Phase 1 pour observer la diffusion des
biais couche par couche.
"""

from dataclasses import dataclass, field


@dataclass
class LightGCNConfig:
    """Configuration d'entraînement pour LightGCN sur Yelp."""

    dataset: str = "yelp"
    embedding_size: int = 64
    n_layers: int = 3          # nombre de couches de message passing
    reg_weight: float = 1e-4
    learning_rate: float = 1e-3
    train_batch_size: int = 4096
    epochs: int = 100
    top_k: list = field(default_factory=lambda: [10, 20])

    def to_recbole_dict(self) -> dict:
        """Convertit la config au format attendu par RecBole."""
        return {
            "model": "LightGCN",
            "dataset": self.dataset,
            "embedding_size": self.embedding_size,
            "n_layers": self.n_layers,
            "reg_weight": self.reg_weight,
            "learning_rate": self.learning_rate,
            "train_batch_size": self.train_batch_size,
            "epochs": self.epochs,
            "topk": self.top_k,
            "metrics": ["Recall", "NDCG", "Precision", "Hit"],
            "valid_metric": "NDCG@10",
        }


def build_trainer(config: LightGCNConfig):
    """Instancie le modèle et le trainer RecBole à partir de la config.

    Nécessite RecBole installé (voir requirements.txt).
    Placeholder d'intégration — à connecter une fois le dataset Yelp
    au format RecBole finalisé dans data/yelp/.
    """
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.model.general_recommender import LightGCN
    from recbole.trainer import Trainer

    rb_config = Config(model="LightGCN", dataset=config.dataset, config_dict=config.to_recbole_dict())
    dataset = create_dataset(rb_config)
    train_data, valid_data, test_data = data_preparation(rb_config, dataset)

    model = LightGCN(rb_config, train_data.dataset)
    trainer = Trainer(rb_config, model)

    return trainer, train_data, valid_data, test_data


if __name__ == "__main__":
    cfg = LightGCNConfig()
    print("Configuration LightGCN par défaut :")
    for k, v in cfg.to_recbole_dict().items():
        print(f"  {k}: {v}")
