import os
import yaml
from functools import partial

import keras_tuner as kt
from tensorflow import keras
from config_class import BaldOrNotConfig
from metrics import get_metrics
from src.model import BaldOrNotModel


def model_builder(hp, config):
    """
    Builds the model for hyperparameter tuning.

    Args:
        hp: Hyperparameter object for Keras Tuner.
        config: Configuration object containing the model parameters.

    Returns:
        A compiled Keras model.
    """
    params = config.tuning_params
    hp_dense_units = hp.Choice(
        "dense_units", values=params.hp_dense_units_values
    )
    hp_dropout_rate = hp.Float(
        "dropout_rate",
        min_value=params.hp_dropout_rate_min_value,
        max_value=params.hp_dropout_rate_max_value,
        step=params.hp_dropout_rate_step,
    )
    hp_learning_rate = hp.Choice(
        "learning_rate", values=params.hp_learning_rate_values
    )

    model = BaldOrNotModel(
        dense_units=hp_dense_units,
        dropout_rate=hp_dropout_rate,
        freeze_backbone=config.model_params.freeze_backbone,
    )

    optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=config.tuning_params.loss_function,
        metrics=get_metrics(config.metrics),
    )

    return model


def tune_model(train_dataset, val_dataset, config: BaldOrNotConfig):
    """
    Tunes the model's hyperparameters using Keras Tuner with Hyperband.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        config (BaldOrNotConfig): Configuration object containing the training parameters.

    Returns:
        The best hyperparameters found during tuning.
    """
    tuner = kt.Hyperband(
        partial(model_builder, config=config),
        objective=config.tuning_params.objective,
        max_epochs=config.tuning_params.epochs,  # Max epochs for Hyperband
        factor=config.tuning_params.factor,  # Factor controlling resource allocation reduction per round
        directory=os.path.join("..", "tuning_logs"),
        project_name=f"hyperband_tuning_{config.tuning_params.steps_per_epoch}",
    )

    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.tuning_params.epochs,
        steps_per_epoch=config.tuning_params.steps_per_epoch,
        validation_steps=config.tuning_params.validation_steps,
    )

    # Retrieve the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    return best_hps


def update_config_with_best_hps(best_hps, config_file_path):
    """
    Updates the YAML configuration file with the best hyperparameters found.

    Args:
        best_hps: Best hyperparameters found by the tuner.
        config_file_path (str): Path to the configuration file to save the updated values.
    """
    with open(config_file_path, "r") as file:
        config_data = yaml.safe_load(file)

    config_data["model_params"]["dropout_rate"] = best_hps.get("dropout_rate")
    config_data["model_params"]["dense_units"] = best_hps.get("dense_units")
    config_data["training_params"]["learning_rate"] = best_hps.get(
        "learning_rate"
    )

    with open(config_file_path, "w") as file:
        yaml.dump(config_data, file)
