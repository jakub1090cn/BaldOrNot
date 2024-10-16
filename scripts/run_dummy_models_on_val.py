import pandas as pd
import os
from src.dummy_models import (
    dummy_random,
    dummy_always_bald,
    dummy_always_non_bald,
    evaluate_dummy_model,
)

val_csv_path = os.path.join("..", "src", "data", "val.csv")
val_df = pd.read_csv(val_csv_path)
val_labels = val_df["label"].values

output_dir = os.path.join("..", "results")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "dummy_model_on_val_metrics_report.txt")

with open(output_file, "w") as report_file:
    report_file.write("Metrics for Dummy Models on Validation Data:\n\n")

    bald_predictions = dummy_always_bald(val_labels)
    bald_metrics = evaluate_dummy_model(bald_predictions, val_labels)

    report_file.write("Metrics for 'Always Bald' Dummy Model:\n")
    for metric, value in bald_metrics.items():
        report_file.write(f"{metric}: {value:.4f}\n")

    non_bald_predictions = dummy_always_non_bald(val_labels)
    non_bald_metrics = evaluate_dummy_model(non_bald_predictions, val_labels)

    report_file.write("\nMetrics for 'Always Non-Bald' Dummy Model:\n")
    for metric, value in non_bald_metrics.items():
        report_file.write(f"{metric}: {value:.4f}\n")

    random_predictions = dummy_random(val_labels)
    random_metrics = evaluate_dummy_model(random_predictions, val_labels)

    report_file.write("\nMetrics for Random Dummy Model:\n")
    for metric, value in random_metrics.items():
        report_file.write(f"{metric}: {value:.4f}\n")

print(f"Report saved to {output_file}")
