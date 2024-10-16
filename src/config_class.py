from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List
import tensorflow as tf


@dataclass
class ModelParams:
    dense_units: int = 512
    freeze_backbone: bool = True
    dropout_rate: float = 0.5
    saved_model_name = "model.keras"


@dataclass
class TrainingParams:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "binary_crossentropy"
    training_name: str = "training_name"
    max_class_imbalance_ratio: float = 2
    use_class_weight: bool = False
    augment_class: bool = True
    steps_per_epoch: int | None = None
    validation_steps: int | None = 50
    use_hyperparameter_tuning: bool = True
    max_tuning_trials: int = 15


@dataclass
class Callback:
    type: str
    args: Dict[str, any] = field(default_factory=dict)

    def to_dict(self):
        """Convert the Callback to a dictionary format."""
        return asdict(self)


@dataclass
class Paths:
    subsets_division_ds_path = (
        "C:\\Users\\Admin\\Downloads\\archive (3)\\" "list_eval_partition.csv"
    )
    labels_ds_path = (
        "C:\\Users\\Admin\\Downloads\\archive (3)\\" "list_attr_celeba.csv"
    )
    images_dir = (
        "C:\\Users\\Admin\\Downloads\\archive (3)\\"
        "img_align_celeba\\img_align_celeba"
    )


@dataclass
class BaldOrNotConfig:
    model_params: ModelParams = field(default_factory=lambda: ModelParams())
    training_params: TrainingParams = field(
        default_factory=lambda: TrainingParams()
    )
    callbacks: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            Callback(
                type="EarlyStopping",
                args={"monitor": "val_loss", "patience": 5},
            ).to_dict(),
            Callback(
                type="TensorBoard",
                args={"log_dir": "logs", "histogram_freq": 1},
            ).to_dict(),
        ]
    )
    metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1_score"]
    )
    paths: Paths = field(default_factory=lambda: Paths())

    def __post_init__(self):
        self.model_params = (
            ModelParams(**self.model_params)
            if isinstance(self.model_params, dict)
            else self.model_params
        )
        self.training_params = (
            TrainingParams(**self.training_params)
            if isinstance(self.training_params, dict)
            else self.training_params
        )
        self.callbacks = [
            Callback(**params).to_dict()
            if isinstance(params, dict)
            else params
            for params in self.callbacks
        ]
        for callback in self.callbacks:
            if "args" not in callback:
                callback["args"] = {}
        self.paths = (
            Paths(**self.paths) if isinstance(self.paths, dict) else self.paths
        )
