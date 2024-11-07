import numpy as np
import tensorflow as tf

import pytest


@pytest.fixture
def prediction_image_path():
    return r"C:\Users\user\Projekty\BaldOrNot\tests\test_images\BALD4.jpg"


@pytest.fixture
def trained_model():
    model_path = r"C:\Users\user\Projekty\BaldOrNot\trainings\training_name2024-10-14_23-34-02\model.keras"
    return tf.keras.models.load_model(model_path)
