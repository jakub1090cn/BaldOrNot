from dataclasses import asdict

import pytest
import yaml

from src.config import BoldOrNotConfig
from src.model import BaldOrNotModel


@pytest.fixture
def test_config() -> BoldOrNotConfig:
    config_path = "test_config.yaml"
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)
    return BoldOrNotConfig(**config_data)


@pytest.fixture
def model(test_config) -> BaldOrNotModel:
    """
    Fixture for creating an instance of the BaldOrNotModel.

    Returns:
        BaldOrNotModel: An instance of the BaldOrNotModel class.
    """
    return BaldOrNotModel(**asdict(test_config.model_params))