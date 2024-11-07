import numpy as np

from src.model import BaldOrNotModel


def make_prediction(model: BaldOrNotModel, img_batch: np.ndarray) -> float:
    """Make a prediction using a given model and image batch."""
    prediction = model.predict(img_batch)

    return round(prediction[0][0])
