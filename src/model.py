import tensorflow as tf
from constants import IMG_LEN, NUM_CHANNELS


class BaldOrNotModel(tf.keras.Model):
    def __init__(self, freeze_backbone=True):
        super().__init__()

        self.convnext_tiny = tf.keras.applications.ConvNeXtTiny(
            include_top=False, input_shape=(IMG_LEN, IMG_LEN, NUM_CHANNELS)
        )
        if freeze_backbone:
            self.convnext_tiny.trainable = False

        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(512, activation="relu")
        self.predictions = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.convnext_tiny(inputs)
        x = self.gap(x)
        x = self.dense(x)
        return self.predictions(x)
