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


def train_model(model_config: dict):
    data_paths_train, labels_train = load_data_from_csv(
        model_config['paths']['train_path'],
        model_config['images_dir']
    )

    data_paths_val, labels_val = load_data_from_csv(
        model_config['paths']['val_path'],
        model_config['images_dir']
    )

    train_generator = BoldDataset(
        data_paths=data_paths_train,
        labels=labels_train,
        batch_size=model_config['training_params']['batch_size'],
        input_shape=(IMG_LEN, IMG_LEN, NUM_CHANNELS),
        shuffle=True
    )

    val_generator = BoldDataset(
        data_paths=data_paths_val,
        labels=labels_val,
        batch_size=model_config['training_params']['batch_size'],
        input_shape=(IMG_LEN, IMG_LEN, NUM_CHANNELS),
        shuffle=False
    )


