import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * (
            (precision * recall)
            / (precision + recall + tf.keras.backend.epsilon())
        )

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


def get_metrics(metrics_list):
    """
    Returns a list of metrics based on the provided names in metrics_list.

    Args:
        metrics_list: List of metric names as strings.

    Returns:
        List of metric instances for evaluation.
    """
    metrics_mapping = {
        "accuracy": tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        "precision": tf.keras.metrics.Precision(name="precision"),
        "recall": tf.keras.metrics.Recall(name="recall"),
    }

    metrics = [
        metrics_mapping[metric]
        for metric in metrics_list
        if metric in metrics_mapping
    ]

    if "f1_score" in metrics_list:
        for metric in ["precision", "recall"]:
            if metric not in metrics_list:
                metrics.append(metrics_mapping[metric])

        metrics.append(F1Score(name="f1_score"))

    return metrics
