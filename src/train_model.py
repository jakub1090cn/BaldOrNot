import yaml
import tensorflow as tf
import pandas as pd
import os
from src.model import BaldOrNotModel
from src.constants import IMG_LEN, NUM_CHANNELS
from src.dataset import BoldDataset


def load_data_from_csv(csv_path: str, images_dir: str):
    data = pd.read_csv(csv_path)
    data_paths = [os.path.join(images_dir, img_name) for img_name in data.iloc[:, 0]]
    labels = data.iloc[:, 1].tolist()
    return data_paths, labels


