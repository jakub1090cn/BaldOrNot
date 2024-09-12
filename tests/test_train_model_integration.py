import os.path
import shutil
from src.config_class import BoldOrNotConfig
from src.model_training import train_model


def test_num_of_epochs(test_config: BoldOrNotConfig, output_dir: str) -> None:
    """
    Tests if the number of epochs in the training history matches the number
    of epochs defined in the configuration.

    Args:
        test_config (BoldOrNotConfig): The model configuration containing the number of epochs.
        output_dir (str): The output directory path provided by the fixture.

    Raises:
        AssertionError: If the number of epochs in the training history does not match
        the number of epochs defined in the configuration.
    """
    history = train_model(config=test_config, output_dir_path=output_dir)

    assert len(history.epoch) == test_config.training_params.epochs, (
        "There is difference between number of epochs in config "
        "and number of epochs in history"
    )


def test_tensorboard_logs_saving(test_config: BoldOrNotConfig, output_dir: str) -> None:
    """
    Tests whether TensorBoard logs are generated during model training and
    saved correctly in the appropriate directory.

    Args:
        test_config (BoldOrNotConfig): The model configuration containing the logging parameters.
        output_dir (str): The output directory path provided by the fixture.

    Raises:
        AssertionError: If the TensorBoard log directory is not created or if it is empty.
    """
    log_dir = "tensorboard_logs_test"

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    train_model(config=test_config, output_dir_path=output_dir)

    assert os.path.exists(log_dir), "Log directory was not created"
    log_files = os.listdir(log_dir)
    assert len(log_files) > 0, "Log directory is empty"

    shutil.rmtree(log_dir)


def test_early_stopping(test_config: BoldOrNotConfig, output_dir: str) -> None:
    """
    Tests if the Early Stopping mechanism works correctly by checking if the model
    stops training after the validation loss stops improving for the specified
    patience value.

    Args:
        test_config (BoldOrNotConfig): The model configuration containing the
        Early Stopping parameters.
        output_dir (str): The output directory path provided by the fixture.

    Raises:
        AssertionError: If the model does not stop training after the expected
        number of epochs due to Early Stopping.
    """
    history = train_model(config=test_config, output_dir_path=output_dir)

    early_stopping_patience = 3
    val_loss_history = history.history['val_loss']
    best_epoch = val_loss_history.index(min(val_loss_history))
    num_epochs_trained = len(history.epoch)

    assert num_epochs_trained - best_epoch - 1 <= early_stopping_patience

