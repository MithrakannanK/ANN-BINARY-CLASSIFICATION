"""
Evaluation module for computing classification metrics across all models.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Compute comprehensive binary classification evaluation metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "report": report
    }


def format_metrics_table(metrics_list: List[Dict[str, Any]]) -> str:
    """
    Format a clean comparison markdown table of evaluation metrics.
    """
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    col_widths = [32, 12, 12, 12, 12, 12]

    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    separator_line = "| " + " | ".join("-" * w for w in col_widths) + " |"

    rows = []
    for m in metrics_list:
        row = [
            m["model_name"].ljust(col_widths[0]),
            f"{m['accuracy']*100:.2f}%".ljust(col_widths[1]),
            f"{m['precision']*100:.2f}%".ljust(col_widths[2]),
            f"{m['recall']*100:.2f}%".ljust(col_widths[3]),
            f"{m['f1_score']*100:.2f}%".ljust(col_widths[4]),
            f"{m['roc_auc']:.4f}".ljust(col_widths[5]),
        ]
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join([header_line, separator_line] + rows)
