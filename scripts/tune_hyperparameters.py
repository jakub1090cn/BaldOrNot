import pandas as pd

from data_utils import adjust_class_distribution
from src.config_class import BaldOrNotConfig
from src.constants import DEFAULT_IMG_SIZE, N_CHANNELS_RGB
from src.dataset import BaldDataset
from src.tuning import tune_model, update_config_with_best_hps


def main():
    # Load the configuration
    config = BaldOrNotConfig()
    params = config.tuning_params

    # Initialize the datasets
    train_csv_path = config.paths.train_csv_path
    train_df = pd.read_csv(train_csv_path)
    train_df = adjust_class_distribution(
        train_df,
        max_class_ratio=params.max_class_imbalance_ratio,
    )
    train_dataset = BaldDataset(
        train_df,
        batch_size=params.batch_size,
        dim=DEFAULT_IMG_SIZE,
        n_channels=N_CHANNELS_RGB,
        shuffle=True,
        augment_minority_class=params.use_class_weight,
    )

    val_csv_path = config.paths.val_csv_path
    val_df = pd.read_csv(val_csv_path)
    val_dataset = BaldDataset(
        val_df,
        batch_size=params.batch_size,
        dim=DEFAULT_IMG_SIZE,
        n_channels=N_CHANNELS_RGB,
        shuffle=False,
        augment_minority_class=False,
    )

    # Perform hyperparameter tuning
    best_hps = tune_model(train_dataset, val_dataset, config)
    update_config_with_best_hps(best_hps, config.paths.config_yaml_path)


if __name__ == "__main__":
    main()
