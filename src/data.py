import os
import cv2
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import pandas as pd


def check_images(directory: str) -> Tuple[List[str], int, int]:
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
