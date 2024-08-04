import pytest
import tensorflow as tf
from src.model import BaldOrNotModel
from src.constants import IMG_LEN, NUM_CHANNELS


@pytest.fixture
def model():
    return BaldOrNotModel()


def test_model_creation(model):
    assert isinstance(model, tf.keras.Model)


def test_model_structure(model):
    assert isinstance(model.convnext_tiny, tf.keras.Model)
    assert isinstance(model.gap, tf.keras.layers.GlobalAveragePooling2D)
    assert isinstance(model.dense, tf.keras.layers.Dense)
    assert isinstance(model.predictions, tf.keras.layers.Dense)


@pytest.mark.parametrize("freeze_backbone", [True, False])
def test_model_trainability(freeze_backbone):
    model = BaldOrNotModel(freeze_backbone=freeze_backbone)
    assert model.convnext_tiny.trainable is not freeze_backbone
    assert model.dense.trainable
    assert model.predictions.trainable


@pytest.mark.parametrize(
    "dropout_rate, should_contain_dropout",
    [
        (None, False),
        (0.5, True),
    ],
)
def test_dropout_possibility(dropout_rate, should_contain_dropout):
    model = BaldOrNotModel(dropout_rate=dropout_rate)
    model.build(input_shape=(None, IMG_LEN, IMG_LEN, NUM_CHANNELS))
    contains_dropout = any(
        isinstance(layer, tf.keras.layers.Dropout) for layer in model.layers
    )
    assert contains_dropout == should_contain_dropout, (
        f"Expected Dropout layer presence: {should_contain_dropout}, "
        f"but got: {contains_dropout}"
    )


if __name__ == "__main__":
    pytest.main()
