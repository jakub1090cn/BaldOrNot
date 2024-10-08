import os
import shutil
from dataclasses import asdict

import pytest
import yaml

from src.config_class import BaldOrNotConfig
from src.model import BaldOrNotModel


@pytest.fixture
def test_config() -> BaldOrNotConfig:
    config_path = "test_config.yaml"
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    return BaldOrNotConfig(**config_data)


@pytest.fixture
def model(test_config) -> BaldOrNotModel:
    """
    Fixture for creating an instance of the BaldOrNotModel.

    Returns:
        BaldOrNotModel: An instance of the BaldOrNotModel class.
    """
    return BaldOrNotModel(**asdict(test_config.model_params))


@pytest.fixture
def output_dir():
    """
    Fixture that creates the 'test_output_dir' directory and a 'training.log'
    file before each test and removes them after the test.

    Yields:
        str: The path to the output directory where logging and model outputs
        will be saved.
    """
    output_dir_path = "test_output_dir"
    os.makedirs(output_dir_path, exist_ok=True)

    # Tworzenie pliku logowania
    log_file_path = os.path.join(output_dir_path, "training.log")
    with open(log_file_path, "w") as log_file:
        log_file.write("Log file initialized.\n")

    yield output_dir_path

    shutil.rmtree(output_dir_path)
