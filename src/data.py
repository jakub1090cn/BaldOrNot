import os
import cv2
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import pandas as pd


def check_sample_images(directory: str) -> Tuple[List[str], int]:
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

    return empty_or_corrupted, num_correct


def prepare_merged_dataframe(subsets_path: str, labels_path: str) -> pd.DataFrame:
    """
     Prepares a combined DataFrame by merging two CSV files on a common column.

     This function reads two CSV files into DataFrames: one containing subsets of data
     and the other containing labels. It then merges these DataFrames on the "image_id"
     column using an inner join, which means that only the rows with matching "image_id"
     values in both DataFrames will be retained in the final result.

     Parameters:
     subsets_path (str): The file path to the CSV containing the subsets data.
     labels_path (str): The file path to the CSV containing the labels data.

     Returns:
     pd.DataFrame: A merged DataFrame containing data from both input CSVs,
                   joined on the "image_id" column.
     """
    subsets = pd.read_csv(subsets_path)
    labels = pd.read_csv(labels_path)
    return pd.merge(subsets, labels, how="inner", on="image_id")


def create_data_subsets(df: pd.DataFrame) -> None:
    """
    Splits the provided DataFrame into training, validation, and test datasets, and saves them as CSV files.

    Args:
        df (pd.DataFrame): A DataFrame containing the dataset, which includes image IDs, partition labels,
        and other relevant data.

    Returns:
        None: This function does not return any value but saves three CSV files: `train.csv`, `validation.csv`,
        and `test.csv`.

    This function processes the input DataFrame by first filtering the data into training and test sets
    based on the `partition` column.
    The training set (where `partition` is 0) is further split into training and validation subsets,
    with 9% of the data allocated
    to the validation set. The resulting subsets are then saved as separate CSV files:
    - `train.csv` for the training data
    - `validation.csv` for the validation data
    - `test.csv` for the test data

    The function also prints the number of samples in each of these subsets.
    """

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
