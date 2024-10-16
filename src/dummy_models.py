import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def dummy_always_bald(val_data):
    return np.ones(len(val_data), dtype=int)


def dummy_always_non_bald(val_data):
    return np.zeros(len(val_data), dtype=int)


def dummy_random(val_data):
    return np.random.randint(2, size=len(val_data))


def evaluate_dummy_model(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
