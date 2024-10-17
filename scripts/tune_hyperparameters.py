import os

import pandas as pd

from src.tuning import tune_model
from src.data import BaldDataset
from src.config_class import BaldOrNotConfig
from src.constants import N_CHANNELS_RGB, DEFAULT_IMG_SIZE


def main():
    # Load the configuration
    config = BaldOrNotConfig()

    # Initialize the datasets
    train_csv_path = os.path.join("..", "src", "data", "train.csv")
    train_df = pd.read_csv(train_csv_path)
    train_df = BaldDataset.adjust_class_distribution(
        train_df,
        max_class_ratio=config.tuning_params.max_class_imbalance_ratio,
    )
    train_dataset = BaldDataset(
        train_df,
        batch_size=config.tuning_params.batch_size,
        dim=DEFAULT_IMG_SIZE,
        n_channels=N_CHANNELS_RGB,
        shuffle=True,
        augment_minority_class=config.tuning_params.augment_minority_class,
    )

    val_csv_path = os.path.join("..", "src", "data", "val.csv")
    val_df = pd.read_csv(val_csv_path)
    val_dataset = BaldDataset(
        val_df,
        batch_size=config.tuning_params.batch_size,
        dim=DEFAULT_IMG_SIZE,
        n_channels=N_CHANNELS_RGB,
        shuffle=False,
        augment_minority_class=False,
    )

    # Perform hyperparameter tuning
    print("Starting hyperparameter tuning...")
    tune_model(train_dataset, val_dataset, config)


if __name__ == "__main__":
    main()
