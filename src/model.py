import tensorflow as tf
from src.config_constants import DENSE_UNITS
from src.constants import IMG_LEN, NUM_CHANNELS


class BaldOrNotModel(tf.keras.Model):
    """
    Binary classification model to determine if a person in an image is bald
    or not.

    Args:
        freeze_backbone (bool): Whether to freeze the layers of
        the ConvNeXtTiny backbone model.
        dropout_rate (float or None): Dropout rate (0-1) to apply after
        the Dense layer. If None, dropout is not applied.

    Attributes:
        backbone (tf.keras.Model): The ConvNeXtTiny backbone model.
        classifier (tf.keras.Sequential): Sequential block containing the
        global average pooling, dense, dropout (optional), and final
        sigmoid-activated dense layer.
    """

    def __init__(
        self, freeze_backbone: bool = True, dropout_rate: float | None = None
    ):
        super().__init__()
        self.backbone: tf.keras.Model = tf.keras.applications.ConvNeXtTiny(
            include_top=False, input_shape=(IMG_LEN, IMG_LEN, NUM_CHANNELS)
        )
        if freeze_backbone:
            self.backbone.trainable = False

        self.classifier: tf.keras.Sequential = tf.keras.Sequential()
        self.classifier.add(tf.keras.layers.GlobalAveragePooling2D())
        self.classifier.add(
            tf.keras.layers.Dense(DENSE_UNITS, activation="relu")
        )
        if dropout_rate is not None:
            self.classifier.add(tf.keras.layers.Dropout(dropout_rate))
        self.classifier.add(tf.keras.layers.Dense(1, activation="sigmoid"))


    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs (tf.Tensor): Input tensor with shape (batch_size, IMG_LEN,
            IMG_LEN, NUM_CHANNELS).

        Returns:
            tf.Tensor: Output tensor with shape (batch_size, 1), containing
            probabilities.
        """

        x = self.backbone(inputs)
        return self.classifier(x)
