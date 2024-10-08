from dataclasses import asdict
from datetime import datetime
import tensorflow as tf
import os
import logging

from src.data import BaldDataset
from src.model import BaldOrNotModel
from src.config_class import BaldOrNotConfig
from src.utils import check_log_exists_decorator
from src.evaluation import evaluate_and_save_results


@check_log_exists_decorator
def train_model(config: BaldOrNotConfig, output_dir_path: str):
    """
    Trains the BaldOrNot model using the specified configuration.

    This function initializes the dataset, constructs the model, compiles it
    with the specified optimizer, loss function, and metrics, and then trains
    the model on the dataset for the number of epochs defined in the
    configuration.

    Args:
        config (BoldOrNotConfig): The configuration object containing model,
        training, and path parameters.
        output_dir_path: The dir path, where is saved model and metrics.
    """
    logging.info("Preparing data...")
    merged_df = BaldDataset.prepare_merged_dataframe(
        subsets_path=config.paths.subsets_path,
        labels_path=config.paths.labels_path
    )

    train_df, val_df, test_df = BaldDataset.create_subset_dfs(merged_df)
    logging.info(
        f"Load train data: {len(train_df)} images,\n"
        f"Load validation data: {len(val_df)} images,\n"
        f"Load test data: {len(test_df)} images."
    )

    logging.info("Starting model training...")

    vector_dim = config.model_params.dense_units
    batch_size = config.training_params.batch_size

    # Log model parameters
    logging.info(
        f"Model configuration: Dense units: {vector_dim}, "
        f"Batch size: {batch_size}"
    )

    datasets = {
        'train': BaldDataset(df=train_df, batch_size=batch_size),
        'val': BaldDataset(df=val_df, batch_size=batch_size),
        'test': BaldDataset(df=test_df, batch_size=batch_size)
    }

    logging.info(
        f"Datasets initialized with batch size {batch_size}"
        f"and vector dim {vector_dim}"
    )


    # Initialize model
    model = BaldOrNotModel(**asdict(config.model_params))
    logging.info("Model initialized")

    # Compile model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.training_params.learning_rate
    )
    model.compile(
        optimizer=optimizer,
        loss=config.training_params.loss_function,
        metrics=config.metrics,
    )
    logging.info(
        f"Model compiled with Adam optimizer and learning rate "
        f"{config.training_params.learning_rate}"
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
                # tf.keras.callbacks.TensorBoard(**callback_dict["args"])
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(output_dir_path, "logs"),
                    histogram_freq=1
                )
            )
            logging.info(
                f"TensorBoard callback added with parameters: "
                f"{callback_dict['args']}"
            )

    # Train the model
    logging.info(
        f"Starting training for {config.training_params.epochs} epochs"
    )
    history = model.fit(
        datasets['train'],
        epochs=config.training_params.epochs,
        validation_data=datasets['val'],
        callbacks=tf_callbacks,
    )
    logging.info("Model training completed")

    for dataset_name, dataset in datasets.items():
        evaluate_and_save_results(
            model, dataset, dataset_name, output_dir_path
        )

    # Save model and plot
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
