import os
from typing import List, Tuple
import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.config_class import BaldOrNotConfig
from src.constants import (
    N_CHANNELS_GRAYSCALE,
    N_CHANNELS_RGB,
)
from src.exceptions import BaldOrNotDataError


class BaldDataset(keras.utils.Sequence):
    """
    Generates data for Keras models.

    This class is responsible for creating batches of data to be fed into
    a Keras model. It loads images from a directory, preprocesses them, and
    returns them along with their corresponding labels.

    Attributes:
    ----------
    df : pd.DataFrame
        The DataFrame containing the image IDs and labels.
    batch_size : int
        The number of samples per batch.
    dim : Tuple[int, int]
        The dimensions to which all images will be resized (height, width).
    n_channels : int
        Number of channels in the images. Must be either 1 for grayscale or 3 for RGB.
    shuffle : bool
        Whether to shuffle the order of samples at the end of each epoch.
    indexes : np.ndarray
        Array of indices used to keep track of the current batch.
    list_IDs : pd.Series
        Series containing the list of image IDs.
    config : BaldOrNotConfig
        Configuration object containing paths and settings.

    Methods:
    -------
    __len__() -> int:
        Returns the number of batches per epoch.
    __getitem__(index: int) -> Tuple[np.ndarray, np.ndarray]:
        Generates one batch of data.
    on_epoch_end() -> None:
        Updates indexes after each epoch.
    __data_preprocessing(list_IDs_temp: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        Generates data containing batch_size samples.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        dim: Tuple[int, int],
        n_channels: int,
        shuffle: bool,
        augment_minority_class: bool,
    ) -> None:
        """
        Initialization method for BaldDataset.

        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame containing image IDs and labels.
        batch_size : int, optional
            The number of samples per batch (default is 32).
        dim : Tuple[int, int], optional
            Dimensions to which images will be resized (default is (218, 178)).
        n_channels : int, optional
            Number of channels in the images (default is 3 for RGB).
        shuffle : bool, optional
            Whether to shuffle the data at the beginning of each epoch (default is True).
        """
        super().__init__()
        if n_channels not in [N_CHANNELS_GRAYSCALE, N_CHANNELS_RGB]:
            raise ValueError(
                "n_channels must be either 1 (grayscale) or 3 (RGB)."
            )

        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.list_IDs = df["image_id"]
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.config = BaldOrNotConfig()
        self.on_epoch_end()
        self.augment_minority_class = augment_minority_class
        label_counts = self.df["label"].value_counts()
        self.minority_class_label = label_counts.idxmin()

    def __len__(self) -> int:
        """
        Returns the number of batches per epoch.

        Returns:
        -------
        int
            Number of batches per epoch.
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates one batch of data.

        Parameters:
        ----------
        index : int
            Index of the batch.

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray]
            A batch of images and their corresponding labels.
        """
        indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_preprocessing(list_IDs_temp)
        return X, y

    def on_epoch_end(self) -> None:
        """
        Updates indexes after each epoch.

        This method is called at the end of each epoch and is responsible for
        shuffling the data if the shuffle attribute is set to True.
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _augment_image(self, image: tf.Tensor) -> tf.Tensor:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image

    def __data_preprocessing(
        self, list_IDs_temp: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates data containing batch_size samples.

        Parameters:
        ----------
        list_IDs_temp : List[str]
            List of image IDs to generate data for.

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray]
            A batch of images and their corresponding labels.
        """
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        images_dir = self.config.paths.images_dir

        for i, ID in enumerate(list_IDs_temp):
            reconverted_ID = f"{int(ID):06d}.jpg"
            image_path = os.path.join(images_dir, reconverted_ID)
            image = cv2.imread(image_path)

            if image is None:
                raise BaldOrNotDataError(f"Failed to load image: {image_path}")

            image = cv2.resize(image, self.dim[::-1])

            # If grayscale, convert it to RGB
            if (
                image.shape[-1] == N_CHANNELS_GRAYSCALE
                and self.n_channels == N_CHANNELS_RGB
            ):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            label = self.df.loc[self.df["image_id"] == ID, "label"].values[0]
            y[i] = label
            if (
                self.augment_minority_class
                and label == self.minority_class_label
            ):
                image = self._augment_image(image)

            image = tf.cast(image, tf.float32)
            X[i] = image / 255.0  # Normalize to range [0, 1]
            y[i] = self.df.loc[self.df["image_id"] == ID, "label"].values[0]

        return X, y

    @staticmethod
    def __get_wrong_files_list(directory: str) -> List[str]:
        """
        Scans a directory for empty or corrupted image files.

        Parameters:
        ----------
        directory : str
            The path to the directory containing the image files.

        Returns:
        -------
        List[str]
            A list of filenames that are empty or corrupted.
        """
        empty_or_corrupted: List[str] = []

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            img = cv2.imread(file_path)
            if img is None or img.size == 0:
                empty_or_corrupted.append(filename)

        return empty_or_corrupted

    @staticmethod
    def get_cleaned_df(df: pd.DataFrame, images_dir: str) -> pd.DataFrame:
        """
        Cleans the DataFrame by removing rows corresponding to empty or corrupted images.

        Parameters:
        ----------
        df : pd.DataFrame
            The original DataFrame containing image IDs.
        images_dir : str
            The directory containing the images.

        Returns:
        -------
        pd.DataFrame
            A cleaned DataFrame with rows for empty or corrupted images removed.
        """
        empty_or_corrupted = BaldDataset.__get_wrong_files_list(images_dir)
        cleaned_df = df[~df["image_id"].isin(empty_or_corrupted)]
        return cleaned_df

    @staticmethod
    def prepare_merged_dataframe(
        subsets_df: pd.DataFrame, labels_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepares a combined DataFrame by merging two dataframes on a common column.

        Parameters:
        ----------
        subsets_df : pd.DataFrame
            DataFrame containing the subsets data.
        labels_df : pd.DataFrame
            DataFrame containing the labels data.

        Returns:
        -------
        pd.DataFrame
            A merged DataFrame containing data from both input DataFrames, joined on the "image_id" column.
        """
        df_merged = pd.merge(subsets_df, labels_df, how="inner", on="image_id")
        return df_merged[["image_id", "partition", "Bald"]].rename(
            columns={"Bald": "label"}
        )

    @staticmethod
    def create_subset_dfs(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the DataFrame into training, validation, and test sets.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame to split.

        Returns:
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Three DataFrames: train_df, val_df, and test_df.
        """
        train_encoding = 0
        test_encoding = 1
        train_df = df[df["partition"] == train_encoding]
        test_df = df[df["partition"] == test_encoding].drop(
            columns=["partition"]
        )
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

    @staticmethod
    def convert_image_id_column_to_float(
        df: pd.DataFrame, image_id_col: str = "image_id"
    ) -> pd.DataFrame:
        """
        Converts the "image_id" column to float by removing ".jpg".

        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame containing image IDs.
        image_id_col : str, optional
            Column name for image IDs (default is "image_id").

        Returns:
        -------
        pd.DataFrame
            DataFrame with converted image IDs.
        """
        df[image_id_col] = (
            df[image_id_col].str.replace(".jpg", "", regex=False).astype(float)
        )
        return df

    @staticmethod
    def replace_bald_label(
        df: pd.DataFrame,
        original_label: str,
        new_label: str,
        column_name: str = "label",
    ) -> pd.DataFrame:
        """
        Replaces labels in the specified column.

        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame containing the data.
        original_label : str
            The original label to be replaced.
        new_label : str
            The new label to replace the original.
        column_name : str, optional
            The name of the column where the replacement will happen (default is "label").

        Returns:
        -------
        pd.DataFrame
            DataFrame with replaced labels.
        """
        df[column_name] = df[column_name].replace(original_label, new_label)
        return df

    @staticmethod
    def undersample_classes(
        df: pd.DataFrame, label_col: str, class_sample_sizes: dict
    ) -> pd.DataFrame:
        """
        Function to undersample each class to specified sample sizes.

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

    @staticmethod
    def adjust_class_distribution(
        df: pd.DataFrame, max_class_ratio: float, label_col: str = "label"
    ) -> pd.DataFrame:
        """
        Adjusts class distribution by ensuring that the majority class does not exceed the specified ratio
        compared to the minority class.

        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame containing the data.
        max_class_ratio : float
            The maximum allowed ratio between the majority and minority class.
        label_col : str, optional
            The column containing the class labels (default is "label").

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
                new_majority_count, random_state=42
            )
            adjusted_df = pd.concat([majority_class_df, minority_class_df])
            return adjusted_df.reset_index(drop=True)
        else:
            return df
