import tensorflow as tf
from src.config import DENSE_UNITS
from src.constants import IMG_LEN, NUM_CHANNELS


class BaldOrNotModel(tf.keras.Model):
    """
    Binary classification model to determine if a person in an image is bald
    or not.

    Args:
        freeze_backbone (bool): Whether to freeze the layers of
        the ConvNeXtTiny backbone model.
        dropout_rate (float or None): Dropout rate (0-1) to apply after
        the Dense layer.
                                      If None, dropout is not applied.

    Attributes:
        convnext_tiny (tf.keras.Model): The ConvNeXtTiny backbone model.
        gap (tf.keras.layers.Layer): GlobalAveragePooling2D layer.
        dense (tf.keras.layers.Layer): Dense layer with `DENSE_UNITS` units and
        ReLU activation.
        dropout (tf.keras.layers.Layer or None): Dropout layer if
        `dropout_rate` is set.
        predictions (tf.keras.layers.Layer): Dense layer with 1 unit and
        sigmoid activation.
    """

    def __init__(self, freeze_backbone=True, dropout_rate=None):
        super().__init__()
        self.convnext_tiny = tf.keras.applications.ConvNeXtTiny(
            include_top=False, input_shape=(IMG_LEN, IMG_LEN, NUM_CHANNELS)
        )
        if freeze_backbone:
            self.convnext_tiny.trainable = False

        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(DENSE_UNITS, activation="relu")
        self.dropout = (
            tf.keras.layers.Dropout(dropout_rate)
            if dropout_rate is not None
            else None
        )
        self.predictions = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (tf.Tensor): Input tensor with shape (batch_size, IMG_LEN,
            IMG_LEN, NUM_CHANNELS).

        Returns:
            tf.Tensor: Output tensor with shape (batch_size, 1), containing
            probabilities.
        """
        x = self.convnext_tiny(inputs)
        x = self.gap(x)
        x = self.dense(x)
        if self.dropout:
            x = self.dropout(x)
        return self.predictions(x)
