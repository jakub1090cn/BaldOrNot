from jsonargparse import CLI

from src.config import BoldOrNotConfig
from src.model_training import run_experiment

if __name__ == "__main__":
    config = CLI(BoldOrNotConfig)
    run_experiment(config=config)

## --------- colab
# import yaml
# from src.config import BoldOrNotConfig
# from src.model_training import run_experiment
#
# CONFIG_PATH = "path/to/my/gdrive/config.yaml"
# with open(CONFIG_PATH, "r") as fp:
#     config = BoldOrNotConfig(yaml.safe_load(fp))
# run_experiment(config=config)