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

    # Initialize a dictionary to keep track of the best values for each parameter
    best_values = {
        "dense_units": float("inf"),
        "dropout_rate": float("inf"),
        "learning_rate": float("inf"),
    }

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
            epochs=1,
            validation_data=val_dataset,
            steps_per_epoch=1,
            validation_steps=1,
        )

        current_trial = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(current_trial)
        print(type(current_trial))
        current_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
        val_loss = current_trial.metrics.get("val_loss")
        if val_loss == float("inf"):
            logging.info("No val_loss found for the current trial.")
        else:
            logging.info(f"Best val_loss for current trial: {val_loss}")

        # Update best values for current trial parameters
        for param in ["dense_units", "dropout_rate", "learning_rate"]:
            current_value = current_trial.get(param)
            if current_value < best_values[param]:
                best_values[param] = current_value
                logging.info(f"New best value for {param}: {current_value}")

        # Update best parameters if current validation loss is better
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {
                "batch_size": batch_size,
                "max_class_imbalance_ratio": max_class_imbalance_ratio,
                "use_class_weight": use_class_weight,
                "augment_class": augment_class,
                "dense_units": current_trial.get("dense_units"),
                "dropout_rate": current_trial.get("dropout_rate"),
                "learning_rate": current_trial.get("learning_rate"),
            }
            logging.info(f"New best parameters found: {best_params}")
            print(f"New best parameters found: {best_params}")

        logging.info(f"Best val_loss to this point: {best_val_loss}")
        logging.info(f"Best params to this point: {best_params}")
        print(f"Best val_loss to this point: {best_val_loss}")
        print(f"Best params to this point: {best_params}")

    return best_params
