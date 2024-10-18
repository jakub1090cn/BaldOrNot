import tensorflow as tf

import numpy as np

from src.predict_hairstyle import preprocess_image_for_model, make_prediction


def test_make_prediction():
    image = np.ones((100, 100, 3))

    preprocessed_image = preprocess_image_for_model(image)

    model_path = r"C:\Users\user\Projekty\BaldOrNot\trainings\training_name2024-10-14_23-34-02\model.keras"
    model = tf.keras.models.load_model(model_path)

    prediction = make_prediction(model, preprocessed_image)

    assert prediction in [0, 1]


