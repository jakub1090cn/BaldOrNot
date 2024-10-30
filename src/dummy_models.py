import tensorflow as tf
from abc import ABC, abstractmethod

from constants import NOT_BALD_LABEL, BALD_LABEL


class DummyModel(tf.keras.Model, ABC):
    """Abstract base model class for producing dummy predictions."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def call(self, inputs):
        pass


class AlwaysBaldModel(DummyModel):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tf.ones_like(inputs[:, 0], dtype=tf.int32)


class AlwaysNotBaldModel(DummyModel):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tf.zeros_like(inputs[:, 0], dtype=tf.int32)


class RandomModel(DummyModel):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tf.random.uniform(
            shape=(tf.shape(inputs)[0],),
            minval=NOT_BALD_LABEL,
            maxval=BALD_LABEL + 1,
            dtype=tf.int32,
        )
