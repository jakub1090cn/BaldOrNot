import os
import functools
import os
import numpy as np
from config_class import BaldOrNotConfig
from src.constants import LOG_FILE_NAME


def check_log_exists(func):
    """Decorator checking if log file exists in the specified directory."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        output_dir_path = kwargs.get("output_dir_path")
        if output_dir_path is None:
            raise ValueError("output_dir_path is not provided.")

        log_file_path = os.path.join(output_dir_path, LOG_FILE_NAME)
        if not os.path.exists(log_file_path):
            raise RuntimeError(
                f"Log file '{LOG_FILE_NAME}' not found in '{output_dir_path}'. "
                "Ensure logging is initialized."
            )

        return func(*args, **kwargs)

    return wrapper


def save_metrics_report(
    metrics_report: dict,
    output_dir: str,
    config: BaldOrNotConfig,
    filename_prefix: str = "",
):
    """
    Saves the provided metrics report to a specified file.

    Args:
        metrics_report: Dictionary containing metrics for models.
        output_dir: Directory where the file will be saved.
        config: Configuration object with training parameters.
        filename_prefix: Optional prefix for the filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = filename_prefix + config.training_params.metrics_report_filename
    output_file = os.path.join(output_dir, filename)

    with open(output_file, "w") as report_file:
        report_file.write("Metrics Report:\n\n")
        for model_name, metrics in metrics_report.items():
            report_file.write(f"Metrics for '{model_name}':\n")
            for metric, value in metrics.items():
                if isinstance(value, np.ndarray):
                    report_file.write(f"{metric}:\n{value}\n")
                else:
                    report_file.write(f"{metric}: {value:.4f}\n")
            report_file.write("\n")
