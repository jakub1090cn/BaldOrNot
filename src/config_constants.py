import os
import yaml

config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)


DENSE_UNITS = config["dense_units"]
DATA_PATH = config["data_path"]
DATASET_PATH = config["dataset_path"]
IMAGES_PATH = config["images_path"]
