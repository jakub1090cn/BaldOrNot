import os
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import setup_logging
from src.constants import BALD_LABELS
from src.dataset import BaldDataset
from src.model import BaldOrNotModel
from src.utils import check_log_exists


def make_predictions(
    model: tf.keras.Model, dataset: BaldDataset
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Makes predictions on the provided dataset and returns the true and
    predicted labels.

    Args:
        model: The trained model used for making predictions.
        dataset: The dataset to make predictions on.
            This should be an iterable of (images, labels).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - y_true (np.ndarray): The true labels.
            - y_pred (np.ndarray): The predicted labels.
    """
    images, y_true = zip(*dataset)
    images = np.concatenate(images, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    y_pred = model.predict(images)

    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()

    return y_true, y_pred


def get_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculates various evaluation metrics based on the true and predicted labels.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.

    Returns:
        Dict[str, float]: A dictionary containing:
            - accuracy: Accuracy score.
            - precision: Precision score (weighted).
            - recall: Recall score (weighted).
            - f1_score: F1 score (weighted).
            - conf_matrix: Confusion matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    conf_matrix = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "conf_matrix": conf_matrix,
    }


def get_misclassifications(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identifies misclassifications by finding false positives and false negatives.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - false_positives: Indices where false positives occurred.
            - false_negatives: Indices where false negatives occurred.
    """
    # First type: False Positives (predicted 1, but true label is 0)
    false_positives = np.where((y_pred == 1) & (y_true == 0))[0]

    # Second type: False Negatives (predicted 0, but true label is 1)
    false_negatives = np.where((y_pred == 0) & (y_true == 1))[0]

    return false_positives, false_negatives


def drop_confusion_matrix(
    confusion_matrix: np.ndarray, class_names: List[str], output_path: str
) -> None:
    """
    Plots the confusion matrix and saves it as an image file.

    Args:
        confusion_matrix (np.ndarray): Confusion matrix to plot.
        class_names (List[str]): List of class names to label the axes.
        output_path (str): Path where the confusion matrix image will be saved.

    Returns:
        None
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    plt.savefig(output_path)
    plt.close()
    logging.info(f"Confusion matrix saved to {output_path}")


@check_log_exists
def evaluate_and_save_results(
    model: BaldOrNotModel,
    dataset: BaldDataset,
    dataset_name: str,
    output_dir_path: str,
) -> None:
    """
    Evaluates the model on a given dataset, logs the metrics, and saves
    the confusion matrix and misclassifications.

    Args:
        model (tf.keras.Model): The trained model to be evaluated.
        dataset (tf.data.Dataset): The dataset to evaluate the model on.
        dataset_name (str): The name of the dataset (e.g., "train", "val", "test").
        output_dir_path (str): The directory path where results will be saved.

    Returns:
        None
    """
    logging.info(f"Evaluating model on {dataset_name} set...")

    # Make predictions
    y_true, y_pred = make_predictions(model, dataset)

    # Calculate metrics
    metrics = get_metrics(y_true, y_pred)
    logging.info(f"{dataset_name.capitalize()} metrics: ")
    for metric, value in metrics.items():
        if metric != "conf_matrix":
            logging.info(f"{metric}: {value:.4f}")

    # Save confusion matrix
    logging.info(f"Saving confusion matrix for {dataset_name} set...")
    conf_matrix_path = os.path.join(
        output_dir_path, f"{dataset_name}_confusion_matrix.png"
    )
    drop_confusion_matrix(
        metrics["conf_matrix"],
        class_names=BALD_LABELS,
        output_path=conf_matrix_path,
    )

    # Log misclassifications
    logging.info(f"Identifying misclassifications on {dataset_name} set...")
    false_positives, false_negatives = get_misclassifications(y_true, y_pred)
    logging.info(
        f"False positives: {len(false_positives)}, False negatives: {len(false_negatives)}"
    )
