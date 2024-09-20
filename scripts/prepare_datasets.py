import os
import pandas as pd
from src.data import BaldDataset
from src.config_class import BaldOrNotConfig

config = BaldOrNotConfig()
subsets_division_ds_path = config.paths.subsets_division_ds_path
labels_ds_path = config.paths.labels_ds_path
images_dir = config.paths.images_dir

subsets_df = pd.read_csv(subsets_division_ds_path)
labels_df = pd.read_csv(labels_ds_path)

cleaned_df = BaldDataset.get_cleaned_df(labels_df, images_dir)
merged_df = BaldDataset.prepare_merged_dataframe(subsets_df, labels_df)
converted_df = BaldDataset.convert_image_id_column_to_float(merged_df)
balanced_df = BaldDataset.balance_classes(
    converted_df,
    X_cols=["image_id", "partition"],
    y_col="labels",
    minor_class_multiplier=config.training_params.minor_class_multiplier,
)
train_df, val_df, test_df = BaldDataset.create_subset_dfs(balanced_df)


data_dir = os.path.join("..", "src", "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
