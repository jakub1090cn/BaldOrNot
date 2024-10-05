import logging
import os
from dataclasses import asdict
from datetime import datetime

import pandas as pd
import tensorflow as tf

from constants import NOT_BALD_LABEL
from src.config_class import BaldOrNotConfig
from src.data import BaldDataset
from src.model import BaldOrNotModel
from src.utils import check_log_exists


@check_log_exists
def train_model(config: BaldOrNotConfig, output_dir_path: str):
    """
    Trains the BaldOrNot model using the specified configuration, with optional
    hyperparameter tuning using Keras Tuner.

    Args:
        config (BaldOrNotConfig): The configuration object containing model,
        training, and path parameters.
        output_dir_path (str): Directory where the trained model will be saved.

    Returns:
        history: Training history object.
    """

    logging.info("Starting model training...")

    # Load datasets
    train_csv_path = os.path.join("..", "src", "data", "train.csv")
    train_df_unbalanced = pd.read_csv(train_csv_path)
    train_df_still_unb = BaldDataset.undersample_classes(
        train_df_unbalanced,
        label_col="label",
        class_sample_sizes={0: 500, 1: 500},
    )
    train_dataset = BaldDataset(
        train_df_still_unb, batch_size=config.training_params.batch_size
    )
    logging.info(
        f"Training dataset initialized with batch size "
        f"{config.training_params.batch_size}"
    )

    val_csv_path = os.path.join("..", "src", "data", "val.csv")
    val_df_unbalanced = pd.read_csv(val_csv_path)
    val_df_still_unb = BaldDataset.undersample_classes(
        val_df_unbalanced, label_col="label", class_sample_sizes={0: 50, 1: 50}
    )
    val_dataset = BaldDataset(
        val_df_still_unb, batch_size=config.training_params.batch_size
    )
    logging.info(
        f"Validation dataset initialized with batch size "
        f"{config.training_params.batch_size}"
    )

    logging.info("Building model with predefined hyperparameters...")
    model = BaldOrNotModel(**asdict(config.model_params))
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.training_params.learning_rate
    )
    model.compile(
        optimizer=optimizer,
        loss=config.training_params.loss_function,
        metrics=config.metrics,
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

    # Train the best model
    logging.info(
        f"Starting training for {config.training_params.epochs} epochs"
    )
    history = model.fit(
        train_dataset,
        epochs=config.training_params.epochs,
        validation_data=val_dataset,
        callbacks=tf_callbacks,
    )
    logging.info("Model training completed")

    # Save the best model
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
