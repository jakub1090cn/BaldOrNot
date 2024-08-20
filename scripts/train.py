from jsonargparse import CLI

from src.config import BoldOrNotConfig
from src.model_training import run_experiment

if __name__ == "__main__":
    config = CLI(BoldOrNotConfig)
    run_experiment(config=config)
