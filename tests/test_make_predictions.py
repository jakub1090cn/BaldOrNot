import numpy as np

from src.constants import DEFAULT_IMG_SIZE, N_CHANNELS_RGB
from src.data import BaldDataset
from src.predict_hairstyle import make_prediction


def test_make_prediction(prediction_image_path, trained_model):
    preprocessed_image = BaldDataset.preprocess_image(
        prediction_image_path, DEFAULT_IMG_SIZE, N_CHANNELS_RGB
    )
    img_batch = np.expand_dims(preprocessed_image, axis=0)

    prediction = make_prediction(trained_model, img_batch)

    assert prediction in [0, 1]
