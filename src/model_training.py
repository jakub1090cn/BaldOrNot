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
from src.constants import BALD_LABEL, NOT_BALD_LABEL


def get_classes_weights():
    df = pd.read_csv("..//src//data//train.csv")
    n_total = len(df)
    n_not_bald = df["label"].value_counts()[NOT_BALD_LABEL]
    n_bald = df["label"].value_counts()[BALD_LABEL]
    not_bald_weight = n_total / n_not_bald
    bald_weight = n_total / n_bald
    return {str(NOT_BALD_LABEL): not_bald_weight, str(BALD_LABEL): bald_weight}


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
    train_df = pd.read_csv(train_csv_path)
    train_dataset = BaldDataset(
        train_df, batch_size=config.training_params.batch_size
    )
    logging.info(
        f"Training dataset initialized with batch size "
        f"{config.training_params.batch_size}"
    )

    val_csv_path = os.path.join("..", "src", "data", "val.csv")
    val_df = pd.read_csv(val_csv_path)
    val_dataset = BaldDataset(
        val_df, batch_size=config.training_params.batch_size
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
            # callback_dict['log_dir'] = os.path.join(output_dir_path,
            #                                         callback_dict['log_dir'])
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
        class_weight=get_classes_weights(),
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
