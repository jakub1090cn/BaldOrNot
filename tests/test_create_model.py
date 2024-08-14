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
    assert isinstance(model.backbone, tf.keras.Model)
    assert isinstance(model.classifier, tf.keras.Sequential)

    layers = [layer.__class__.__name__ for layer in model.classifier.layers]
    expected_layers = [
        "GlobalAveragePooling2D",
        "Dense",
        "Dropout"
        if any(
            isinstance(layer, tf.keras.layers.Dropout)
            for layer in model.classifier.layers
        )
        else None,
        "Dense",
    ]
    expected_layers = [layer for layer in expected_layers if layer is not None]

    assert (
        layers == expected_layers
    ), f"Expected layers: {expected_layers}, but got: {layers}"


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
    assert model.backbone.trainable is not freeze_backbone

    for layer in model.classifier.layers:
        assert layer.trainable


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
        isinstance(layer, tf.keras.layers.Dropout)
        for layer in model.classifier.layers
    )
    assert contains_dropout == should_contain_dropout, (
        f"Expected Dropout layer presence: {should_contain_dropout}, "
        f"but got: {contains_dropout}"
    )


def test_backbone_output_is_4d_tensor(model: BaldOrNotModel) -> None:
    """
    Test if the output of the backbone is a 4D tensor.

    Args:
        model (BaldOrNotModel): An instance of the BaldOrNotModel class.

    Asserts:
        bool: True if the backbone output is a 4D tensor, False otherwise.
    """
    input_tensor = tf.random.uniform((1, IMG_LEN, IMG_LEN, NUM_CHANNELS))

    backbone_output = model.backbone(input_tensor)

    num_dim_of_backbone_output = 4
    is_4d = len(backbone_output.shape) == num_dim_of_backbone_output
    assert is_4d, (
        f"Expected backbone output to be a 4D tensor, but got shape: "
        f"{backbone_output.shape}"
    )

    assert isinstance(backbone_output, tf.Tensor), (
        f"Expected backbone output to be of type tf.Tensor, but got type: "
        f"{type(backbone_output)}"
    )


def test_classifier_input_compatibility(model: BaldOrNotModel) -> None:
    """
    Test if the classifier accepts the correct input tensor shape as output
    by the backbone.

    Args:
        model (BaldOrNotModel): An instance of the BaldOrNotModel class.

    Asserts:
        bool: True if the classifier accepts the correct input tensor shape,
        False otherwise.
    """
    input_tensor = tf.random.uniform((1, IMG_LEN, IMG_LEN, NUM_CHANNELS))

    backbone_output = model.backbone(input_tensor)

    global_avg_pool_output = model.classifier.layers[0](backbone_output)

    assert backbone_output.shape[-1] == global_avg_pool_output.shape[-1], (
        f"Expected classifier input shape to have {backbone_output.shape[-1]}"
        f" channels but got {global_avg_pool_output.shape[-1]} channels."
    )


if __name__ == "__main__":
    pytest.main()
