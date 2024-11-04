import os
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ModelParams:
    dense_units: int = 512
    freeze_backbone: bool = True
    dropout_rate: float = 0.6
    saved_model_name = "model.keras"


@dataclass
class TrainingParams:
    epochs: int = 30
    batch_size: int = 64
    loss_function: str = "binary_crossentropy"
    max_class_imbalance_ratio: int = 3
    steps_per_epoch: int | None = None
    validation_steps: int | None = None
    use_class_weight: bool = True
    augment_minority_class: bool = True
    learning_rate: float = 0.0001
    optimizer: str = "adam"
    training_name: str = "training_name"
    metrics_report_filename: str = "metrics_report.txt"


@dataclass
class TuningParams:
    epochs: int = 2
    batch_size: int = 128
    loss_function: str = "binary_crossentropy"
    max_class_imbalance_ratio: int = 1
    steps_per_epoch: int | None = 5
    validation_steps: int | None = 5
    use_class_weight: bool = False
    augment_minority_class: bool = False
    max_tuning_trials: int = 2
    objective: str = "val_f1_score"  # The metric to be optimized during tuning
    factor: int = 3
    hp_dense_units_values: list[int] = field(
        default_factory=lambda: [128, 256, 512]
    )
    hp_dropout_rate_min_value: float = 0.2
    hp_dropout_rate_max_value: float = 0.7
    hp_dropout_rate_step: float = 0.1
    hp_learning_rate_values: list[float] = field(
        default_factory=lambda: [1e-2, 1e-3, 1e-4]
    )


@dataclass
class AugmentationParams:
    brightness_max_delta: float = 0.1
    contrast_lower: float = 0.9
    contrast_upper: float = 1.1


@dataclass
class Callback:
    type: str
    args: dict[str, any] = field(default_factory=dict)

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
    train_csv_path = os.path.join("..", "src", "data", "train.csv")
    val_csv_path = os.path.join("..", "src", "data", "val.csv")
    test_csv_path = os.path.join("..", "src", "data", "test.csv")
    config_yaml_path = os.path.join("..", "config.yaml")
    results_dir = os.path.join("..", "results")
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
    tuning_params: TuningParams = field(default_factory=lambda: TuningParams())
    augmentation_params: AugmentationParams = field(
        default_factory=lambda: AugmentationParams()
    )
    callbacks: list[dict[str, Any]] = field(
        default_factory=lambda: [
            Callback(
                type="EarlyStopping",
                args={"monitor": "val_f1_score", "mode": "max", "patience": 5},
            ).to_dict(),
            Callback(
                type="TensorBoard",
                args={"log_dir": "logs", "histogram_freq": 1},
            ).to_dict(),
        ]
    )
    metrics: list[str] = field(
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
        self.tuning_params = (
            TuningParams(**self.tuning_params)
            if isinstance(self.tuning_params, dict)
            else self.tuning_params
        )
        self.augmentation_params = (
            AugmentationParams(**self.augmentation_params)
            if isinstance(self.augmentation_params, dict)
            else self.augmentation_params
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
