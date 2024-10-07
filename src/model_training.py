import json
import logging
import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from constants import BALD_LABEL, NOT_BALD_LABEL
from src.config_class import BaldOrNotConfig
from src.data import BaldDataset
from src.model import BaldOrNotModel
from src.utils import check_log_exists


def get_classes_weights(df):
    n_total = len(df)
    n_not_bald = df["label"].value_counts()[NOT_BALD_LABEL]
    n_bald = df["label"].value_counts()[BALD_LABEL]
    not_bald_weight = n_total / n_not_bald
    bald_weight = n_total / n_bald
    return {NOT_BALD_LABEL: not_bald_weight, BALD_LABEL: bald_weight}


@check_log_exists
def train_model(config: BaldOrNotConfig, output_dir_path: str):
    logging.info("Starting model training...")

    train_csv_path = os.path.join("..", "src", "data", "train.csv")
    train_df = pd.read_csv(train_csv_path)
    train_df_limited = BaldDataset.undersample_classes(
        train_df,
        label_col="label",
        class_sample_sizes={0: 5000, 1: 3000},
    )
    train_dataset = BaldDataset(
        train_df_limited, batch_size=config.training_params.batch_size
    )
    logging.info(
        f"Training dataset initialized with batch size "
        f"{config.training_params.batch_size}"
    )

    val_csv_path = os.path.join("..", "src", "data", "val.csv")
    val_df = pd.read_csv(val_csv_path)
    val_df_limited = BaldDataset.undersample_classes(
        val_df, label_col="label", class_sample_sizes={0: 500, 1: 300}
    )
    val_dataset = BaldDataset(
        val_df_limited, batch_size=config.training_params.batch_size
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

    if config.training_params.class_weights_path is None:
        class_weights = get_classes_weights(train_df_limited)
        best_class_weights = class_weights.copy()
        best_val_loss = float("inf")
    else:
        with open(config.training_params.class_weights_path, "r") as f:
            saved_data = json.load(f)
            best_class_weights = saved_data.get("best_class_weights")
            best_val_loss = saved_data.get("best_val_loss", float("inf"))

    output_file = os.path.join(output_dir_path, "best_class_weights.json")

    logging.info(
        f"Starting training for {config.training_params.epochs} epochs"
    )

    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    for epoch in range(config.training_params.epochs):
        logging.info(f"Epoch {epoch + 1}/{config.training_params.epochs}")

        epoch_history = model.fit(
            train_dataset,
            epochs=1,
            class_weight=best_class_weights,
            validation_data=val_dataset,
            callbacks=tf_callbacks,
        )

        for key in epoch_history.history.keys():
            if key in history:
                history[key].extend(epoch_history.history[key])
            else:
                history[key] = epoch_history.history[key]

        val_loss, *metrics = model.evaluate(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_class_weights = best_class_weights.copy()

        best_class_weights[NOT_BALD_LABEL] *= np.random.uniform(0.8, 1.2)
        best_class_weights[BALD_LABEL] *= np.random.uniform(0.8, 1.2)

        with open(output_file, "w") as f:
            json.dump(
                {
                    "best_class_weights": best_class_weights,
                    "best_val_loss": best_val_loss,
                },
                f,
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
