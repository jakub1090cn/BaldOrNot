import pandas as pd
import os

from constants import DUMMY_METRICS_FILE_NAME_PREFIX
from src.dummy_models import AlwaysBaldModel, AlwaysNotBaldModel, RandomModel
from src.evaluation import get_metrics
from src.config_class import BaldOrNotConfig
from utils import save_metrics_report


def main():
    # Initiate config
    config = BaldOrNotConfig()

    # Load the validation data
    val_csv_path = config.paths.val_csv_path
    val_df = pd.read_csv(val_csv_path)
    val_labels = val_df["label"].values

    # Reshape val_labels to be 2D, as models expect input with at least two dim
    val_labels_reshaped = val_labels.reshape(-1, 1)

    # Instantiate the models
    always_bald_model = AlwaysBaldModel()
    always_not_bald_model = AlwaysNotBaldModel()
    random_model = RandomModel()

    # Calculate predictions and metrics for all dummy models
    metrics_report = {}

    # Evaluate the AlwaysBaldModel
    bald_predictions = always_bald_model.predict(val_labels_reshaped)
    bald_metrics = get_metrics(val_labels, bald_predictions)
    metrics_report["Always Bald"] = bald_metrics

    # Evaluate the AlwaysNotBaldModel
    not_bald_predictions = always_not_bald_model.predict(val_labels_reshaped)
    not_bald_metrics = get_metrics(val_labels, not_bald_predictions)
    metrics_report["Always Not-Bald"] = not_bald_metrics

    # Evaluate the RandomModel
    random_predictions = random_model.predict(val_labels_reshaped)
    random_metrics = get_metrics(val_labels, random_predictions)
    metrics_report["Random"] = random_metrics

    # Set up output directory and file
    output_dir = config.paths.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Write metrics to the file
    save_metrics_report(
        metrics_report=metrics_report,
        output_dir=output_dir,
        config=config,
        filename_prefix=DUMMY_METRICS_FILE_NAME_PREFIX,
    )


if __name__ == "__main__":
    main()
