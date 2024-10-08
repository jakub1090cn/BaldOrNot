import os
from datetime import datetime


def init_output_dir(training_name: str) -> str:
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_training = f"{training_name}{current_date}"
    output_dir_path = os.path.join("..", "trainings", current_training)
    os.makedirs(output_dir_path, exist_ok=True)
    return output_dir_path
