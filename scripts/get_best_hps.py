import os
import json


def find_trial_with_max_val_loss(logs_dir):
    max_val_f1_score = float("-inf")
    best_hyperparameters = None

    for trial_folder in os.listdir(logs_dir):
        trial_path = os.path.join(logs_dir, trial_folder, "trial.json")

        if os.path.exists(trial_path):
            with open(trial_path, "r") as f:
                trial_data = json.load(f)
                val_f1_score = trial_data["metrics"]["metrics"][
                    "val_f1_score"
                ]["observations"][0]["value"][0]

                if val_f1_score > max_val_f1_score:
                    max_val_f1_score = val_f1_score
                    best_hyperparameters = trial_data["hyperparameters"][
                        "values"
                    ]
                    best_hyperparameters["val_f1_score"] = val_f1_score

    if best_hyperparameters:
        return {
            "dense_units": best_hyperparameters.get("dense_units"),
            "dropout_rate": best_hyperparameters.get("dropout_rate"),
            "learning_rate": best_hyperparameters.get("learning_rate"),
            "val_f1_score": best_hyperparameters.get("val_f1_score"),
        }
    else:
        return None


def main():
    logs_dir = os.path.join("..", "tuning_logs", "hyperband_tuning_50")
    print(find_trial_with_max_val_loss(logs_dir))


if __name__ == "__main__":
    main()
