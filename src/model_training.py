import json
from dataclasses import asdict

from src.config import BoldOrNotConfig


def run_experiment(config: BoldOrNotConfig):
    print(json.dumps(asdict(config), indent=4))