import tensorflow as tf
import numpy as np
from src.model import create_model
from constants import IMG_LEN, NUM_CHANNELS


def test_model_compilation():
    model = create_model()
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    assert model.optimizer is not None, "Model optimizer should be set."
    assert model.loss is not None, "Model loss should be set."


def test_model_prediction():
    model = create_model()
    num_images = 3
    fake_data = tf.random.normal(
        shape=(num_images, IMG_LEN, IMG_LEN, NUM_CHANNELS)
    )
    predictions = model.predict(fake_data)
    assert predictions.shape == (
        num_images,
        1,
    ), "Prediction output shape is incorrect."
    assert isinstance(
        predictions, np.ndarray
    ), "Prediction should return a numpy array."
