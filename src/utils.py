# utils.py
import functools
import os

from src.constants import LOG_FILE_NAME


def check_log_exists(func):
    """Decorator checking, if log file exists in directory."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        output_dir_path = kwargs.get("output_dir_path", None)

        if output_dir_path is None:
            raise ValueError("output_dir_path is not provided.")

        log_file_path = os.path.join(output_dir_path, LOG_FILE_NAME)

        if not os.path.exists(log_file_path):
            raise RuntimeError(
                f"Log file '{LOG_FILE_NAME}' not found in '{output_dir_path}'. "
                "Make sure logging is initialized."
            )

        return func(*args, **kwargs)

    return wrapper
