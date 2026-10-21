"""Accuracy, precision, recall and F1 (in %) from a confusion matrix (rows = true, columns = predicted)."""

from __future__ import annotations

import torch


def safe_div(num, den) -> float:
    return float(num) / float(den) if den else 0.0


def summarize_confusion(confusion: torch.Tensor):
    """Return ``(metrics, per_class)``.

    ``metrics`` holds overall accuracy and the macro and support-weighted precision, recall and F1;
    ``per_class`` holds lists of support, accuracy, precision, recall and F1. A class with no support or
    no predictions scores 0 on the undefined quantities. Macro averages run over all classes, as in the
    cascade-era reports; the meta-classifier protocol uses ``selection_protocol.metrics_from_confusion``,
    which averages over classes with support.
    """
    num_classes = confusion.size(0)
    support = [int(v) for v in confusion.sum(dim=1)]
    predicted = [int(v) for v in confusion.sum(dim=0)]
    tp = [int(v) for v in confusion.diag()]

    accuracy_per_class = [100.0 * safe_div(tp[c], support[c]) for c in range(num_classes)]
    precision_per_class = [100.0 * safe_div(tp[c], predicted[c]) for c in range(num_classes)]
    recall_per_class = [100.0 * safe_div(tp[c], support[c]) for c in range(num_classes)]
    f1_per_class = []
    for c in range(num_classes):
        p = precision_per_class[c] / 100.0
        r = recall_per_class[c] / 100.0
        f1_per_class.append(100.0 * safe_div(2 * p * r, p + r))

    def weighted(values):
        return safe_div(sum(values[i] * support[i] for i in range(num_classes)), sum(support))

    metrics = {
        "overall_accuracy": 100.0 * safe_div(sum(tp), sum(support)),
        "macro_precision": sum(precision_per_class) / num_classes,
        "macro_recall": sum(recall_per_class) / num_classes,
        "macro_f1": sum(f1_per_class) / num_classes,
        "weighted_precision": weighted(precision_per_class),
        "weighted_recall": weighted(recall_per_class),
        "weighted_f1": weighted(f1_per_class),
    }
    per_class = {
        "support": support,
        "accuracy": accuracy_per_class,
        "precision": precision_per_class,
        "recall": recall_per_class,
        "f1": f1_per_class,
    }
    return metrics, per_class
