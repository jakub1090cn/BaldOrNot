import os
import logging
import shutil
from jsonargparse import CLI
import json
from dataclasses import asdict
from src.config_class import BaldOrNotConfig
from src.constants import LOG_FILE_NAME
from src.logging import setup_logging
from src.model_training import train_model, init_output_dir
from src.plot import plot_metric_curve


def run_experiment(config: BaldOrNotConfig):
    logging.info("Initializing output directory...")
    output_dir_path = init_output_dir(config.training_params.training_name)
    output_file_name = LOG_FILE_NAME

    logging.info("Setting up logging...")
    setup_logging(output_dir_path, output_file_name)

    config_path = os.path.join(output_dir_path, "config.yaml")
    logging.info(
        f"Copying configuration file to output directory: {config_path}"
    )
    shutil.copy("../config.yaml", config_path)

    logging.info(
        f"Running experiment with the following configuration:\n"
        f"{json.dumps(asdict(config), indent=4)}"
    )

    logging.info("Starting model training...")
    history = train_model(config=config, output_dir_path=output_dir_path)

    logging.info(
        "Model training completed. History of training:\n"
        f"{json.dumps(history, indent=4)}"
    )

    logging.info("Plotting metric curve for loss...")
    plot_metric_curve(
        history=history, metric="loss", output_dir_path=output_dir_path
    )
    logging.info("Loss curve saved.")

    logging.info("Plotting metric curve for accuracy...")
    plot_metric_curve(
        history=history, metric="accuracy", output_dir_path=output_dir_path
    )
    logging.info("Accuracy curve saved.")

    logging.info("Experiment completed successfully.")


if __name__ == "__main__":
    logging.info("Parsing configuration from CLI...")
    config = CLI(BaldOrNotConfig)

    logging.info("Running experiment with parsed configuration.")
    run_experiment(config=config)
