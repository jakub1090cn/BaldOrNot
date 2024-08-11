import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from typing import List, Dict

from constants import BALD_LABELS


def display_sample_images(df: pd.DataFrame, dir_path: str) -> None:
    """
    Displays a sample of images from the dataset based on the "Bald" attribute.

    Args:
        df (pd.DataFrame): DataFrame containing image metadata, including 'image_id' and 'Bald' attributes.
        dir_path (str): Path to the directory where the images are stored.

    Returns:
        None: This function displays the images but does not return any value.

    The function selects one sample image for each value of the "Bald" attribute (e.g., bald and not bald),
    converts the images to RGB, and displays them side by side using Matplotlib. The title above each image
    indicates whether the person in the image is bald or not, based on the 'Bald' attribute in the DataFrame.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    labels = BALD_LABELS

    for i, label in enumerate(labels):
        image = df[df["Bald"] == label]
        sample_image = image.sample()
        image_id = sample_image["image_id"].values[0]
        img_path = os.path.join(dir_path, image_id)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[i].imshow(img_rgb)
        axes[i].axis("off")
        axes[i].set_title(labels[label])

    plt.show()


def plot_proportions(
    column: pd.DataFrame, mapper: Dict[int, str], description: List[str]
) -> None:
    """
    Plots the proportions of different categories in a DataFrame column as a bar chart.

    Args:
        column (pd.DataFrame): A pandas Series or DataFrame column containing categorical data.
        mapper (Dict[int, str]): A dictionary mapping the numerical categories in the column to
        descriptive string labels.
        description (List[str]): A list of strings used to describe the plot, where:
            - description[0]: Title of the plot.
            - description[1]: Label for the x-axis.
            - description[2]: Label for the y-axis.

    Returns:
        None: The function displays a bar chart but does not return any value.

    This function counts the occurrences of each category in the provided column, maps these categories
    to descriptive labels using the provided mapper, and then plots the counts as a bar chart. The chart
    includes the total count for each category displayed on top of each bar.
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
