import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from src.model import create_model
from constants import IMG_LEN, NUM_CHANNELS, NUM_DENSE_UNITS


def test_create_model_structure():
    model = create_model()
    assert isinstance(
        model, Model
    ), "The create_model function should return an instance of a Keras model."


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


def test_input_output_shape():
    model = create_model()
    assert model.input_shape == (
        None,
        IMG_LEN,
        IMG_LEN,
        NUM_CHANNELS,
    ), "Incorrect input shape."
    assert model.output_shape == (None, 1), "Incorrect output shape."


def test_base_model_frozen():
    model = create_model()
    base_model = model.layers[1]
    for layer in base_model.layers:
        assert not layer.trainable, f"The layer {layer.name} should be frozen."


def test_dense_units():
    model = create_model()
    dense_layer = [
        layer
        for layer in model.layers
        if isinstance(layer, Dense) and layer.units == NUM_DENSE_UNITS
    ]
    assert (
        len(dense_layer) == 1
    ), f"The model should have one Dense layer with {NUM_DENSE_UNITS} units."

    assert (
        len(dense_layer) == 1
    ), f"The model should have one Dense layer with {NUM_DENSE_UNITS} units."
