import os
from datetime import datetime


def init_output_dir(training_name: str) -> str:
    """
    Creates and returns a unique directory path for training outputs.

    Args:
        training_name: Base name for the training output directory.

    Returns:
        Path to the created output directory.
    """
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_training = f"{training_name}{current_date}"
    output_dir_path = os.path.join("..", "trainings", current_training)
    os.makedirs(output_dir_path, exist_ok=True)
    return output_dir_path
