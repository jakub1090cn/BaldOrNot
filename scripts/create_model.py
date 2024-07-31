import tensorflow as tf

from src.model import create_model

model = create_model()
sample_images = tf.random.normal(shape=(3, 224, 224, 3))
print(model(sample_images))
