import tensorflow as tf
from keras_tuner import HyperParameters, RandomSearch
from itertools import product
import logging

from src.constants import N_CHANNELS_RGB, DEFAULT_IMG_SIZE
from src.model import BaldOrNotModel
from src.data import BaldDataset
from src.utils import check_log_exists


def build_model(hp, config):
    dense_units = hp.Int(
        "dense_units", min_value=128, max_value=1024, step=128
    )
    dropout_rate = hp.Float(
        "dropout_rate", min_value=0.2, max_value=0.7, step=0.1
    )

    model = BaldOrNotModel(
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        freeze_backbone=config.model_params.freeze_backbone,
    )

    learning_rate = hp.Float(
        "learning_rate", min_value=1e-5, max_value=1e-2, sampling="log"
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )

    return model


@check_log_exists
def tune_training_process(train_df, val_df, config, output_dir_path):
    batch_sizes = [32, 64, 128]
    max_class_imbalance_ratios = [1.0, 2.0, 3.0]
    use_class_weight_options = [True, False]
    augment_class_options = [True, False]

    best_val_loss = float("inf")
    best_params = {}

    param_combinations = list(
        product(
            batch_sizes,
            max_class_imbalance_ratios,
            use_class_weight_options,
            augment_class_options,
        )
    )

    logging.info(
        f"Number of parameter combinations: {len(param_combinations)}"
    )

    for (
        batch_size,
        max_class_imbalance_ratio,
        use_class_weight,
        augment_class,
    ) in param_combinations:
        logging.info(
            f"Training with batch_size={batch_size}, "
            f"max_class_imbalance_ratio={max_class_imbalance_ratio}, "
            f"use_class_weight={use_class_weight}, "
            f"augment_class={augment_class}..."
        )
        print(
            f"Training with batch_size={batch_size}, "
            f"max_class_imbalance_ratio={max_class_imbalance_ratio}, "
            f"use_class_weight={use_class_weight}, "
            f"augment_class={augment_class}..."
        )

        # Adjust the training data distribution
        adjusted_train_df = BaldDataset.adjust_class_distribution(
            train_df, max_class_ratio=max_class_imbalance_ratio
        )
        train_dataset = BaldDataset(
            adjusted_train_df,
            batch_size=batch_size,
            dim=DEFAULT_IMG_SIZE,
            n_channels=N_CHANNELS_RGB,
            shuffle=True,
            augment_minority_class=augment_class,
        )
        val_dataset = BaldDataset(
            val_df,
            batch_size=batch_size,
            dim=DEFAULT_IMG_SIZE,
            n_channels=N_CHANNELS_RGB,
            shuffle=False,
            augment_minority_class=False,
        )

        logging.info("Initializing tuner...")
        tuner = RandomSearch(
            lambda hp: build_model(hp, config),
            objective="val_loss",
            max_trials=1,
            executions_per_trial=1,
            directory=output_dir_path,
            project_name=f"training_process_tuning_{batch_size}_{max_class_imbalance_ratio}_{use_class_weight}_{augment_class}",
        )

        logging.info("Starting hyperparameter tuning...")
        tuner.search(
            train_dataset,
            epochs=5,
            validation_data=val_dataset,
            steps_per_epoch=20,
            validation_steps=20,
        )

        best_trial = tuner.get_best_hyperparameters(num_trials=1)[0]
        val_loss = best_trial.get("val_loss", float("inf"))
        logging.info(f"Best val_loss for current trial: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {
                "batch_size": batch_size,
                "max_class_imbalance_ratio": max_class_imbalance_ratio,
                "use_class_weight": use_class_weight,
                "augment_class": augment_class,
                "dense_units": best_trial.get("dense_units"),
                "dropout_rate": best_trial.get("dropout_rate"),
                "learning_rate": best_trial.get("learning_rate"),
            }
            logging.info(f"New best parameters found: {best_params}")

    return best_params
