from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef
)
import numpy as np

def binary_eval_metrics(y_true, y_pred, name_list):
    """
    Calculation of evaluation matrix
    :param y_true: a list of true labels
    :param y_pred: a list of predicted labels
    :param name_list: names of metrics
    :return: a dictionary of metrics values

    Scenario	                    Recommended Metric(s)
    Balanced classes	            Accuracy, F1 Score
    Imbalanced classes	            Precision, Recall, F1 Score, MCC, G-Mean
    Cost of false alarms is high	Precision
    Cost of missing faults is high	Recall
    """
    metrics = {}

    if "accuracy" in name_list:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

    if "precision" in name_list:
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)

    if "recall" in name_list:
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)

    if "f1_score" in name_list:
        metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)

    if "confusion_matrix" in name_list:
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    if "matthews_corrcoef" in name_list:
        metrics["matthews_corrcoef"] = matthews_corrcoef(y_true, y_pred)

    if "geometric_mean" in name_list:
        # Compute confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        gmean = np.sqrt(sensitivity * specificity)
        metrics["geometric_mean"] = gmean

    if len(metrics) > 0:
        return metrics
    else:
        raise ValueError("Expected metrics are not found!")