import pytest
import tensorflow as tf
from src.model import BaldOrNotModel
from src.constants import IMG_LEN, NUM_CHANNELS


@pytest.fixture
def model() -> BaldOrNotModel:
    """
    Fixture for creating an instance of the BaldOrNotModel.

    Returns:
        BaldOrNotModel: An instance of the BaldOrNotModel class.
    """
    return BaldOrNotModel()


def test_model_creation(model: BaldOrNotModel) -> None:
    """
    Test if the model is an instance of tf.keras.Model.

    Args:
        model (BaldOrNotModel): An instance of the BaldOrNotModel class.

    Asserts:
        bool: True if the model is an instance of tf.keras.Model,
        False otherwise.
    """
    assert isinstance(model, tf.keras.Model)


def test_model_structure(model: BaldOrNotModel) -> None:
    """
    Test the structure of the BaldOrNotModel.

    Args:
        model (BaldOrNotModel): An instance of the BaldOrNotModel class.

    Asserts:
        bool: True if the model has the correct structure, False otherwise.
    """
    assert isinstance(model.convnext_tiny, tf.keras.Model)
    assert isinstance(model.gap, tf.keras.layers.GlobalAveragePooling2D)
    assert isinstance(model.dense, tf.keras.layers.Dense)
    assert isinstance(model.predictions, tf.keras.layers.Dense)


@pytest.mark.parametrize("freeze_backbone", [True, False])
def test_model_trainability(freeze_backbone: bool) -> None:
    """
    Test the trainability of the model's layers based on the freeze_backbone
    parameter.

    Args:
        freeze_backbone (bool): Whether to freeze the backbone model.

    Asserts:
        bool: True if the trainability of the layers is correct,
        False otherwise.
    """
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
def test_dropout_possibility(
    dropout_rate: float | None, should_contain_dropout: bool
) -> None:
    """
    Test the presence of a Dropout layer in the model based on the dropout_rate
    parameter.

    Args:
        dropout_rate (float or None): The dropout rate [0-1] or None if dropout
        is not applied.
        should_contain_dropout (bool): Whether the model should contain
        a Dropout layer.

    Asserts:
        bool: True if the model contains a Dropout layer when expected,
        False otherwise.
    """
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
