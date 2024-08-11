import os
import cv2
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import pandas as pd


def check_images(directory: str) -> Tuple[List[str], int, int]:
    """
    Checks the images in the specified directory to identify empty or corrupted files.

    Args:
        directory (str): The path to the directory containing the images.

    Returns:
        Tuple[List[str], int, int]:
            - A list of filenames that are either empty or corrupted.
            - The count of empty or corrupted images.
            - The count of correctly loaded images.

    This function iterates through all the files in the given directory, attempting to load each image using OpenCV's `
    cv2.imread` function.
    If an image cannot be loaded (i.e., it is empty or corrupted), it is added to the `empty_or_corrupted` list.
    The function finally returns this list along with the count of corrupted/empty images and the count of successfully
    loaded images.
    """
    empty_or_corrupted: List[str] = []
    num_correct: int = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        img = cv2.imread(file_path)
        if img is None or img.size == 0:
            empty_or_corrupted.append(filename)
        else:
            num_correct += 1

    return empty_or_corrupted, len(empty_or_corrupted), num_correct


def create_data_subsets(subsets_path: str, labels_path: str) -> None:
    """
    Creates and saves training, validation, and test datasets based on the provided subsets and labels.

    Args:
        subsets_path (str): Path to the CSV file containing image IDs and their corresponding partition labels.
        labels_path (str): Path to the CSV file containing image IDs and their corresponding labels.

    Returns:
        None: The function does not return any value but saves three CSV files: `train.csv`, `validation.csv`,
        and `test.csv`.

    This function reads the provided subsets and labels CSV files, merges them based on the `image_id` column,
    and splits the data into training, validation, and test sets. The partition labels are used to separate
    the data into training (partition 0) and test (partition 1) sets. The training set is further split
    into training and validation subsets with 9% of the data allocated to validation. The resulting datasets
    are saved as CSV files.
    """
    subsets = pd.read_csv(subsets_path)
    labels = pd.read_csv(labels_path)
    df = pd.merge(subsets, labels, how="inner", on="image_id")
    train_df = df[df["partition"] == 0]
    test_df = df[df["partition"] == 1].drop(columns=["partition"])
    train_df, val_df = train_test_split(
        train_df.drop(columns=["partition"]), test_size=0.09, random_state=42
    )

    print("Number of samples in the training set:", len(train_df))
    print("Number of samples in the validation set:", len(val_df))
    print("Number of samples in the test set:", len(test_df))

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("validation.csv", index=False)
    test_df.to_csv("test.csv", index=False)
