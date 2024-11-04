import os

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

from constants import BALD_LABEL, NOT_BALD_LABEL, NUMBER_OF_CLASSES


def get_wrong_files_list(directory: str) -> list[str]:
    """Returns a list of filenames in the directory that are empty or corrupted."""
    empty_or_corrupted = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        img = cv2.imread(file_path)
        if img is None or img.size == 0:
            empty_or_corrupted.append(filename)

    return empty_or_corrupted


def get_cleaned_df(df: pd.DataFrame, images_dir: str) -> pd.DataFrame:
    """
    Removes rows in the DataFrame that correspond to empty or corrupted images.
    """
    empty_or_corrupted = get_wrong_files_list(images_dir)
    cleaned_df = df[~df["image_id"].isin(empty_or_corrupted)]
    return cleaned_df


def prepare_merged_dataframe(
    subsets_df: pd.DataFrame, labels_df: pd.DataFrame
) -> pd.DataFrame:
    """Merges subsets and labels dataframes on 'image_id' and renames 'Bald' column to 'label'."""
    df_merged = pd.merge(subsets_df, labels_df, how="inner", on="image_id")
    return df_merged[["image_id", "partition", "Bald"]].rename(
        columns={"Bald": "label"}
    )


def create_subset_dfs(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training, validation, and test sets based on partition values.

    Returns:
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The training, validation, and test DataFrames.
    """
    train_encoding = 0
    test_encoding = 1
    train_df = df[df["partition"] == train_encoding]
    test_df = df[df["partition"] == test_encoding].drop(columns=["partition"])
    train_df, val_df = train_test_split(
        train_df.drop(columns=["partition"]),
        test_size=0.09,
        random_state=42,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def convert_image_id_column_to_float(
    df: pd.DataFrame, image_id_col: str = "image_id"
) -> pd.DataFrame:
    """Converts image IDs from string to float by removing the '.jpg' extension."""
    df[image_id_col] = (
        df[image_id_col].str.replace(".jpg", "", regex=False).astype(float)
    )
    return df


def replace_bald_label(
    df: pd.DataFrame,
    original_label: str,
    new_label: str,
    column_name: str = "label",
) -> pd.DataFrame:
    """Replaces specific labels in the DataFrame's specified column."""
    df[column_name] = df[column_name].replace(original_label, new_label)
    return df


def undersample_classes(
    df: pd.DataFrame, label_col: str, class_sample_sizes: dict
) -> pd.DataFrame:
    """
    Undersamples each class in the DataFrame to the specified sample sizes.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    label_col : str
        Name of the column containing class labels.
    class_sample_sizes : dict
        Dictionary where keys are class labels and values are the desired number of samples for each class.

    Returns:
    -------
    pd.DataFrame
        DataFrame with each class undersampled to the desired number of samples.
    """
    df_undersampled_list = []

    for class_label, target_size in class_sample_sizes.items():
        df_class = df[df[label_col] == class_label]
        n_samples = min(target_size, len(df_class))
        df_class_sampled = df_class.sample(n=n_samples, random_state=42)
        df_undersampled_list.append(df_class_sampled)

    df_undersampled = pd.concat(df_undersampled_list, ignore_index=True)
    return df_undersampled.sample(frac=1, random_state=42).reset_index(
        drop=True
    )


def adjust_class_distribution(
    df: pd.DataFrame, max_class_ratio: float, label_col: str = "label"
) -> pd.DataFrame:
    """
    Ensures the majority class does not exceed the specified ratio relative to the minority class.

    Parameters:
    ----------
    max_class_ratio : float
        The maximum allowed ratio between the majority and minority class.

    Returns:
    -------
    pd.DataFrame
        DataFrame with adjusted class distribution.
    """
    class_counts = df[label_col].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    majority_count = class_counts[majority_class]
    minority_count = class_counts[minority_class]

    if majority_count > max_class_ratio * minority_count:
        new_majority_count = max_class_ratio * minority_count
        majority_class_df = df[df[label_col] == majority_class]
        minority_class_df = df[df[label_col] == minority_class]
        majority_class_df = majority_class_df.sample(
            int(new_majority_count), random_state=42
        )
        adjusted_df = pd.concat([majority_class_df, minority_class_df])
        return adjusted_df.reset_index(drop=True)
    else:
        return df


def get_classes_weights(df: pd.DataFrame) -> dict[str, float]:
    """Calculates class weights based on label distribution in the DataFrame."""
    n_total = len(df)
    n_not_bald = df["label"].value_counts()[NOT_BALD_LABEL]
    n_bald = df["label"].value_counts()[BALD_LABEL]
    not_bald_weight = (1 / n_not_bald) * (n_total / NUMBER_OF_CLASSES)
    bald_weight = (1 / n_bald) * (n_total / NUMBER_OF_CLASSES)
    return {
        NOT_BALD_LABEL: not_bald_weight,
        BALD_LABEL: bald_weight,
    }
