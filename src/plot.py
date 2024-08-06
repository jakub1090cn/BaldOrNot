import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from typing import List, Dict


def display_sample_images(df: pd.DataFrame, dir_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    labels = {-1: "Not Bald", 1: "Bald"}

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
