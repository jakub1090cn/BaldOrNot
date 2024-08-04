import tensorflow as tf
from src.config import DENSE_UNITS
from src.constants import IMG_LEN, NUM_CHANNELS


class BaldOrNotModel(tf.keras.Model):
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
        x = self.convnext_tiny(inputs)
        x = self.gap(x)
        x = self.dense(x)
        if self.dropout:
            x = self.dropout(x)
        return self.predictions(x)
