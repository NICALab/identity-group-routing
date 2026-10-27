from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true_arr.size == 0:
        return {"acc": float("nan"), "bal_acc": float("nan"), "macro_f1": float("nan")}
    acc = float(np.mean(y_true_arr == y_pred_arr))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro"))
    return {"acc": acc, "bal_acc": bal_acc, "macro_f1": macro_f1}
