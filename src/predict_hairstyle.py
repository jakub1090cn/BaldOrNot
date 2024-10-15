import os
import cv2
import numpy as np

from src.constants import DEFAULT_IMG_SIZE
from src.model import BaldOrNotModel


def load_image_to_predict(img_dir_path: str, img_name: str) -> np.ndarray:
    """Load an image to predict from a given path."""
    img_path = os.path.join(img_dir_path, img_name)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_name}")
    return img


def preprocess_image_for_model(img: np.ndarray) -> np.ndarray:
    """Preprocess an image to be fed into a model."""
    img_resized = cv2.resize(img, DEFAULT_IMG_SIZE)
    if img_resized.shape[-1] == 1:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch


def make_prediction(model: BaldOrNotModel, img_batch: np.ndarray) -> float:
    """Make a prediction using a given model and image batch."""
    prediction = model.predict(img_batch)

    label = round(prediction[0][0])
    return label
