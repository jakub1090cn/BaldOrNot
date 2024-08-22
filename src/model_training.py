import json
from dataclasses import asdict

from src.config import BaldOrNotConfig


def run_experiment(config: BaldOrNotConfig):
    print(json.dumps(asdict(config), indent=4))
