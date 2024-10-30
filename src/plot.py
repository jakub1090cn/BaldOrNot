import logging
import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from keras.src.callbacks import History

from src.constants import BALD_LABELS
from src.utils import check_log_exists


def display_sample_images(df: pd.DataFrame, dir_path: str):
    """
    Displays a sample of images based on the "Bald" attribute.

    Args:
        df: DataFrame with image metadata, including 'image_id' and 'Bald'.
        dir_path: Directory path where images are stored.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    labels = BALD_LABELS

    for i, label in enumerate(labels):
        sample_image = df[df["Bald"] == label].sample()
        image_id = sample_image["image_id"].values[0]
        img_path = os.path.join(dir_path, image_id)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[i].imshow(img_rgb)
        axes[i].axis("off")
        axes[i].set_title(labels[label])

    plt.show()


def plot_proportions(
    column: pd.DataFrame, mapper: dict[int, str], description: list[str]
):
    """
    Plots the proportions of different categories in a DataFrame column as a bar chart.

    Args:
        column: DataFrame column containing categorical data.
        mapper: Dictionary mapping numerical categories to descriptive labels.
        description: List of strings describing the plot title, x-axis, and y-axis labels.
    """
    counts = column.value_counts()
    counts.index = counts.index.map(mapper)
    plt.figure(figsize=(8, 6))
    ax = counts.plot(kind="bar", color=["skyblue", "orange"])
    plt.title(description[0])
    plt.xlabel(description[1])
    plt.ylabel(description[2])
    plt.xticks(rotation=0)

    for p in ax.patches:
        ax.annotate(
            str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005)
        )

    plt.show()


@check_log_exists
def plot_metric_curve(
    history: History, metric_name: str, output_dir_path: str
):
    """
    Plots and saves the curve for a given metric.

    Args:
        history: History object returned by `model.fit()`.
        metric_name: Name of the metric to plot (e.g., 'loss', 'accuracy').
        output_dir_path: Directory path where the plot will be saved.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(
        history.history[metric_name],
        label=f"{metric_name.capitalize()} (training)",
    )

    val_metric = f"val_{metric_name}"
    if val_metric in history.history:
        plt.plot(
            history.history[val_metric],
            label=f"{metric_name.capitalize()} (validation)",
        )

    plt.title(f"{metric_name.capitalize()} Curves")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.capitalize())
    plt.legend()

    plot_path = os.path.join(output_dir_path, f"{metric_name}_plot.png")
    plt.savefig(plot_path)
    plt.close()

    logging.info(f"Plot for {metric_name} saved at: {plot_path}")
