import os

import pandas as pd

from data_utils import (
    convert_image_id_column_to_float,
    create_subset_dfs,
    get_cleaned_df,
    prepare_merged_dataframe,
    replace_bald_label,
)
from src.config_class import BaldOrNotConfig
from src.constants import NOT_BALD_LABEL, ORIGINAL_NOT_BALD_LABEL
from src.dataset import BaldDataset


def main():
    # Initialize configuration and paths
    config = BaldOrNotConfig()
    subsets_division_ds_path = config.paths.subsets_division_ds_path
    labels_ds_path = config.paths.labels_ds_path
    images_dir = config.paths.images_dir

    # Load and process data
    subsets_df = pd.read_csv(subsets_division_ds_path)
    labels_df = pd.read_csv(labels_ds_path)

    cleaned_df = get_cleaned_df(labels_df, images_dir)
    merged_df = prepare_merged_dataframe(subsets_df, cleaned_df)
    converted_df = convert_image_id_column_to_float(merged_df)
    corrected_labels_df = replace_bald_label(
        converted_df,
        original_label=ORIGINAL_NOT_BALD_LABEL,
        new_label=NOT_BALD_LABEL,
    )

    # Create data subsets
    train_df, val_df, test_df = create_subset_dfs(corrected_labels_df)

    # Save subsets to CSV files
    data_dir = os.path.join("..", "src", "data")
    os.makedirs(data_dir, exist_ok=True)

    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)


if __name__ == "__main__":
    main()
