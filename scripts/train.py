from jsonargparse import CLI

from src.config import BaldOrNotConfig
from src.model_training import run_experiment

if __name__ == "__main__":
    config = CLI(BaldOrNotConfig)
    run_experiment(config=config)
