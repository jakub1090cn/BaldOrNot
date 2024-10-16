import logging
import os
from datetime import datetime

import pandas as pd
import tensorflow as tf

from src.config_class import BaldOrNotConfig
from src.data import BaldDataset
from src.model import BaldOrNotModel
from src.utils import check_log_exists
from src.metrics import get_metrics
from src.constants import (
    BALD_LABEL,
    NOT_BALD_LABEL,
    NUMBER_OF_CLASSES,
    DEFAULT_IMG_SIZE,
    N_CHANNELS_RGB,
)
from src.tuning import tune_training_process


def get_classes_weights():
    df = pd.read_csv("..//src//data//train.csv")
    n_total = len(df)
    n_not_bald = df["label"].value_counts()[NOT_BALD_LABEL]
    n_bald = df["label"].value_counts()[BALD_LABEL]
    not_bald_weight = (1 / n_not_bald) * (n_total / NUMBER_OF_CLASSES)
    bald_weight = (1 / n_bald) * (n_total / NUMBER_OF_CLASSES)
    return {str(NOT_BALD_LABEL): not_bald_weight, str(BALD_LABEL): bald_weight}


@check_log_exists
def train_model(config: BaldOrNotConfig, output_dir_path: str):
    logging.info("Starting model training...")

    train_csv_path = os.path.join("..", "src", "data", "train.csv")
    train_df = pd.read_csv(train_csv_path)

    val_csv_path = os.path.join("..", "src", "data", "val.csv")
    val_df = pd.read_csv(val_csv_path)

    param_sources = {
        "model_params": ["dense_units", "dropout_rate"],
        "training_params": [
            "batch_size",
            "learning_rate",
            "max_class_imbalance_ratio",
            "use_class_weight",
            "augment_class",
        ],
    }

    if config.training_params.use_hyperparameter_tuning:
        logging.info("Starting hyperparameter tuning...")
        best_hyperparameters = tune_training_process(
            train_df, val_df, config=config, output_dir_path=output_dir_path
        )
        current_params = best_hyperparameters
    else:
        current_params = {
            param: getattr(getattr(config, source), param)
            for source, params in param_sources.items()
            for param in params
        }

    for param_name, param_value in current_params.items():
        globals()[param_name] = param_value

    balanced_train_df = BaldDataset.adjust_class_distribution(
        train_df,
        max_class_ratio=max_class_imbalance_ratio,
    )
    train_dataset = BaldDataset(
        balanced_train_df,
        batch_size=batch_size,
        dim=DEFAULT_IMG_SIZE,
        n_channels=N_CHANNELS_RGB,
        shuffle=True,
        augment_minority_class=augment_class,
    )
    logging.info(f"Training dataset initialized with batch size {batch_size}")

    val_dataset = BaldDataset(
        val_df,
        batch_size=batch_size,
        dim=DEFAULT_IMG_SIZE,
        n_channels=N_CHANNELS_RGB,
        shuffle=True,
        augment_minority_class=False,
    )
    logging.info(
        f"Validation dataset initialized with batch size {batch_size}"
    )

    logging.info("Building model with predefined hyperparameters...")
    model = BaldOrNotModel(
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        freeze_backbone=config.model_params.freeze_backbone,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=config.training_params.loss_function,
        metrics=get_metrics(config.metrics),
    )

    # Initialize callbacks
    tf_callbacks = []
    for callback_dict in config.callbacks:
        if callback_dict["type"] == "EarlyStopping":
            tf_callbacks.append(
                tf.keras.callbacks.EarlyStopping(**callback_dict["args"])
            )
            logging.info(
                f"EarlyStopping callback added with parameters: "
                f"{callback_dict['args']}"
            )
        elif callback_dict["type"] == "TensorBoard":
            tf_callbacks.append(
                tf.keras.callbacks.TensorBoard(**callback_dict["args"])
            )
            logging.info(
                f"TensorBoard callback added with parameters: "
                f"{callback_dict['args']}"
            )

    logging.info(
        f"Starting training for {config.training_params.epochs} epochs"
    )

    if use_class_weight:
        class_weight = get_classes_weights()
    else:
        class_weight = None

    history = model.fit(
        train_dataset,
        epochs=config.training_params.epochs,
        class_weight=class_weight,
        validation_data=val_dataset,
        callbacks=tf_callbacks,
        steps_per_epoch=config.training_params.steps_per_epoch,
        validation_steps=config.training_params.validation_steps,
    )

    logging.info("Model training completed")

    model_path = os.path.join(
        output_dir_path, config.model_params.saved_model_name
    )
    model.save(model_path)
    logging.info(f"Model saved at {model_path}")

    return history


def init_output_dir(training_name: str) -> str:
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_training = f"{training_name}{current_date}"
    output_dir_path = os.path.join(project_path, "trainings", current_training)
    os.makedirs(output_dir_path, exist_ok=True)
    return output_dir_path
