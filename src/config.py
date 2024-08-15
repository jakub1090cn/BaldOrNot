from dataclasses import dataclass, field, asdict
from typing import List, Dict

@dataclass
class ModelParams:
    dense_units: int = 512
    freeze_backbone: bool = True
    dropout_rate: float = 0.5


@dataclass
class TrainingParams:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "binary_crossentropy"


@dataclass
class Callback:
    type: str
    args: Dict[str, any] = field(default_factory=dict)

    def to_dict(self):
        """Convert the Callback to a dictionary format."""
        return asdict(self)


@dataclass
class Paths:
    train_path: str = ''
    val_path: str = ''
    images_dir: str = ''


@dataclass
class BoldOrNotConfig:
    model_params: ModelParams = field(default_factory=lambda: ModelParams())
    training_params: TrainingParams = field(default_factory=lambda: TrainingParams())
    callbacks: List[Dict[str, any]] = field(default_factory=lambda: [
        Callback(type="EarlyStopping",
                 args={
                        "monitor": "val_loss",
                        "patience": 5
                    }).to_dict()
    ])
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    paths: Paths = field(default_factory=lambda: Paths())

