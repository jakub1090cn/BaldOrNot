import logging
import os
from dataclasses import asdict
from datetime import datetime

import pandas as pd
import tensorflow as tf

from src.config_class import BaldOrNotConfig
from src.data import BaldDataset
from src.model import BaldOrNotModel
from src.utils import check_log_exists
from src.metrics import get_metrics
from src.constants import (
    DEFAULT_IMG_SIZE,
    N_CHANNELS_RGB,
)


@check_log_exists
def train_model(config: BaldOrNotConfig, output_dir_path: str):
    """
    Trains the BaldOrNot model using the specified configuration, with optional
    hyperparameter tuning using Keras Tuner.
    """
    logging.info("Starting model training...")

    train_csv_path = os.path.join("..", "src", "data", "train.csv")
    train_df = pd.read_csv(train_csv_path)
    train_df = BaldDataset.adjust_class_distribution(
        train_df,
        max_class_ratio=config.training_params.max_class_imbalance_ratio,
    )
    train_dataset = BaldDataset(
        train_df,
        batch_size=config.training_params.batch_size,
        dim=DEFAULT_IMG_SIZE,
        n_channels=N_CHANNELS_RGB,
        shuffle=True,
        augment_minority_class=config.training_params.augment_minority_class,
    )

    val_csv_path = os.path.join("..", "src", "data", "val.csv")
    val_df = pd.read_csv(val_csv_path)
    val_dataset = BaldDataset(
        val_df,
        batch_size=config.training_params.batch_size,
        dim=DEFAULT_IMG_SIZE,
        n_channels=N_CHANNELS_RGB,
        shuffle=True,
        augment_minority_class=False,
    )

    if config.training_params.use_tuned_hyperparameters:
        pass

    model = BaldOrNotModel(**asdict(config.model_params))
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.training_params.learning_rate
    )
    model.compile(
        optimizer=optimizer,
        loss=config.training_params.loss_function,
        metrics=get_metrics(config.metrics),
    )

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

    if config.training_params.use_class_weight:
        class_weight = BaldDataset.get_classes_weights(train_df)
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
