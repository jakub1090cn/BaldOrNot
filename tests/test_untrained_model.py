import pytest
import tensorflow as tf
from src.model import BaldOrNotModel
from constants import IMG_LEN, NUM_CHANNELS


@pytest.fixture
def model():
    return BaldOrNotModel()


def test_model_compile(model):
    try:
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
    except Exception as e:
        pytest.fail(f"Model compilation failed: {e}")


def test_model_prediction_shape(model):
    num_images = 3
    fake_data = tf.random.normal(
        shape=(num_images, IMG_LEN, IMG_LEN, NUM_CHANNELS)
    )
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    try:
        predictions = model.predict(fake_data)
    except Exception as e:
        pytest.fail(f"Model prediction failed: {e}")
    expected_output_shape = (num_images, 1)
    assert predictions.shape == expected_output_shape, (
        f"Expected output shape {expected_output_shape}, "
        f"but got {predictions.shape}"
    )


if __name__ == "__main__":
    pytest.main()
