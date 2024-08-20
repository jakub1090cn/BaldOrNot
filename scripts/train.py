import json
from dataclasses import asdict

from jsonargparse import CLI

from src.config import BoldOrNotConfig


def run_experiment(config: BoldOrNotConfig):
    print(json.dumps(asdict(config), indent=4))


if __name__ == "__main__":
    config = CLI(BoldOrNotConfig)
    run_experiment(config=config)

## --------- colab
# import yaml
# from src.config import BoldOrNotConfig
# from scripts.train import run_experiment
#
# CONFIG_PATH = "path/to/my/gdrive/config.yaml"
# with open(CONFIG_PATH, "r") as fp:
#     config = BoldOrNotConfig(yaml.safe_load(fp))
# run_experiment(config=config)