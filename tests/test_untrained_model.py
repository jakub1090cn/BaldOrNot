import pytest
import tensorflow as tf
from src.model import BaldOrNotModel
from src.constants import IMG_LEN, NUM_CHANNELS


def test_model_compile(model: BaldOrNotModel) -> None:
    """
    Test to ensure the model can be compiled without errors.

    Args:
        model (BaldOrNotModel): An instance of the BaldOrNotModel class.

    Asserts:
        The model compiles without raising an exception.
    """
    try:
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
    except Exception as e:
        pytest.fail(f"Model compilation failed: {e}")


def test_model_prediction_shape(model: BaldOrNotModel) -> None:
    """
    Test to ensure the model's predictions have the correct shape and are
    within the expected range.

    Args:
        model (BaldOrNotModel): An instance of the BaldOrNotModel class.

    Asserts:
        The predictions have the correct shape (num_images, 1).
        The predictions are within the range [0, 1].
    """
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
    else:
        expected_output_shape = (num_images, 1)
        assert predictions.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape}, "
            f"but got {predictions.shape}"
        )
        assert (predictions >= 0).all() and (
                predictions <= 1
        ).all(), f"Predictions should be in range [0, 1], but got {predictions}"


if __name__ == "__main__":
    pytest.main()
