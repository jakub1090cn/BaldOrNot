import logging
import os

from src.constants import LOG_FILE_NAME


def setup_logging(training_dir: str):
    """
    Configures logging to output logs to both a file and the console.

    Args:
        training_dir: Directory where logs will be saved.
    """
    log_file = os.path.join(training_dir, LOG_FILE_NAME)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info(f"Logging initialized. Logs will be saved to {log_file}")
