import os

import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config_class import BaldOrNotConfig
from src.constants import (
    IMAGE_NORMALIZATION_FACTOR,
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
    dim : tuple[int, int]
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
    __getitem__(index: int) -> tuple[np.ndarray, np.ndarray]:
        Generates one batch of data.
    on_epoch_end():
        Updates indexes after each epoch.
    __data_preprocessing(list_IDs_temp: list[str]) -> tuple[np.ndarray, np.ndarray]:
        Generates data containing batch_size samples.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        dim: tuple[int, int],
        n_channels: int,
        shuffle: bool,
        augment_minority_class: bool,
    ):
        """
        Initialization method for BaldDataset.

        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame containing image IDs and labels.
        batch_size : int, optional
            The number of samples per batch.
        dim : tuple[int, int], optional
            Dimensions to which images will be resized.
        n_channels : int, optional
            Number of channels in the images.
        shuffle : bool, optional
            Whether to shuffle the data at the beginning of each epoch.
        augment_minority_class : bool, optional
            Whether to augment the minority class during data generation.
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

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates one batch of data.

        Parameters:
        ----------
        index : int
            Index of the batch.

        Returns:
        -------
        tuple[np.ndarray, np.ndarray]
            A batch of images and their corresponding labels.
        """
        indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_preprocessing(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.

        This method is called at the end of each epoch and is responsible for
        shuffling the data if the shuffle attribute is set to True.
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _augment_image(self, image: tf.Tensor) -> tf.Tensor:
        """
        Applies random augmentations to an image.

        Parameters:
        ----------
        image : tf.Tensor
            Input image to augment.

        Returns:
        -------
        tf.Tensor
            Augmented image.
        """
        augmentation_params = self.config.augmentation_params
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(
            image, max_delta=augmentation_params.brightness_max_delta
        )
        image = tf.image.random_contrast(
            image,
            lower=augmentation_params.contrast_lower,
            upper=augmentation_params.contrast_upper,
        )
        return image

    def __data_preprocessing(
        self, list_IDs_temp: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a batch of data.

        Parameters:
        ----------
        list_IDs_temp : list[str]
            List of image IDs to generate data for.

        Returns:
        -------
        tuple[np.ndarray, np.ndarray]
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
            X[i] = image / IMAGE_NORMALIZATION_FACTOR
            y[i] = self.df.loc[self.df["image_id"] == ID, "label"].values[0]

        return X, y
