import logging
import os


def setup_logging(logging_dir, file_name):
    """
    Configures logging settings for the project.
    Logs will be saved to both a file and the console.

    Args:
        logging_dir (str): Directory where logs will be saved.
        file_name (str): Name of the log file.
    """
    log_file = os.path.join(logging_dir, file_name)

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # You can adjust the level as needed

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info(f"Logging initialized. Logs will be saved to {log_file}")
