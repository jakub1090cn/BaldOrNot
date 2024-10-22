import pandas as pd
import os
import numpy as np

from constants import DUMMY_METRICS_FILE_NAME
from src.dummy_models import AlwaysBaldModel, AlwaysNotBaldModel, RandomModel
from src.evaluation import get_metrics
from src.config_class import BaldOrNotConfig


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
    output_file = os.path.join(output_dir, DUMMY_METRICS_FILE_NAME)

    # Write metrics to the file
    with open(output_file, "w") as report_file:
        report_file.write("Metrics for Dummy Models on Validation Data:\n\n")

        for model_name, metrics in metrics_report.items():
            report_file.write(f"Metrics for '{model_name}' Dummy Model:\n")
            for metric, value in metrics.items():
                if isinstance(value, np.ndarray):
                    report_file.write(f"{metric}:\n{value}\n")
                else:
                    report_file.write(f"{metric}: {value:.4f}\n")
            report_file.write("\n")

    print(f"Report saved to {output_file}")


if __name__ == "__main__":
    main()
