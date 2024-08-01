from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from src.model import create_model
from constants import IMG_LEN, NUM_CHANNELS


def test_create_model_structure():
    model = create_model()
    assert isinstance(
        model, Model
    ), "The create_model function should return an instance of a Keras model."


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
    exp_num_dense_u = 512
    dense_layer = [
        layer
        for layer in model.layers
        if isinstance(layer, Dense) and layer.units == exp_num_dense_u
    ]
    assert (
        len(dense_layer) == 1
    ), f"The model should have one Dense layer with {exp_num_dense_u} units."

    assert (
        len(dense_layer) == 1
    ), f"The model should have one Dense layer with {exp_num_dense_u} units."
