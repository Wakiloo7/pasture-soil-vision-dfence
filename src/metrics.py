from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, object]:
    """
    Calculate basic classification metrics.

    Useful for image-level pasture/soil condition classification.
    """
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def calculate_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Calculate Intersection over Union for two bounding boxes.

    Box format:
    [x1, y1, x2, y2]
    """
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box_a_area = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    box_b_area = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union_area = box_a_area + box_b_area - intersection_area

    if union_area == 0:
        return 0.0

    return round(float(intersection_area / union_area), 4)


def mean_iou(true_boxes: List[List[float]], pred_boxes: List[List[float]]) -> float:
    """
    Calculate mean IoU for matched true and predicted boxes.

    This simple version assumes boxes are already paired.
    """
    if not true_boxes or not pred_boxes:
        return 0.0

    pair_count = min(len(true_boxes), len(pred_boxes))
    iou_values = [calculate_iou(true_boxes[i], pred_boxes[i]) for i in range(pair_count)]

    return round(float(np.mean(iou_values)), 4)