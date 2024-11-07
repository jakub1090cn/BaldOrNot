import tensorflow as tf
import os
from jsonargparse import CLI
import numpy as np
from src.config_class import PredictConfig
from src.constants import DEFAULT_IMG_SIZE, N_CHANNELS_RGB
from src.data import BaldDataset
from src.predict_hairstyle import make_prediction


if __name__ == "__main__":
    config = CLI(PredictConfig)

    model_path = config.predict_params.model_path
    image_dir_path = config.predict_params.image_dir_path
    image_name = config.predict_params.image_name

    image_path = os.path.join(image_dir_path, image_name)
    preprocessed_image = BaldDataset.preprocess_image(
        image_path, DEFAULT_IMG_SIZE, N_CHANNELS_RGB
    )
    img_batch = np.expand_dims(preprocessed_image, axis=0)
    model = tf.keras.models.load_model(model_path)

    prediction = make_prediction(model, img_batch)
    print(f"Predicted label: {prediction}")
