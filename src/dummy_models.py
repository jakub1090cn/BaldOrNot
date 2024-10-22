import tensorflow as tf
from abc import ABC, abstractmethod


class DummyModel(tf.keras.Model, ABC):
    def __init__(self):
        super(DummyModel, self).__init__()

    @abstractmethod
    def call(self, inputs):
        pass


class AlwaysBaldModel(DummyModel):
    def __init__(self):
        super(AlwaysBaldModel, self).__init__()

    def call(self, inputs):
        return tf.ones_like(inputs[:, 0], dtype=tf.int32)


class AlwaysNotBaldModel(DummyModel):
    def __init__(self):
        super(AlwaysNotBaldModel, self).__init__()

    def call(self, inputs):
        return tf.zeros_like(inputs[:, 0], dtype=tf.int32)


class RandomModel(DummyModel):
    def __init__(self):
        super(RandomModel, self).__init__()

    def call(self, inputs):
        return tf.random.uniform(
            shape=(tf.shape(inputs)[0],), minval=0, maxval=2, dtype=tf.int32
        )
