import numpy as np
import torch


def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def denormalize_box(box_norm, img_w, img_h):
    cx, cy, w, h = box_norm
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def match_predictions_to_targets(pred_boxes, target_boxes, iou_threshold=0.5):
    tp = 0
    fp = len(pred_boxes)
    used_targets = set()

    for pred_box in pred_boxes:
        best_iou = 0
        best_target_idx = -1

        for target_idx, target_box in enumerate(target_boxes):
            if target_idx in used_targets:
                continue
            iou = compute_iou(pred_box, target_box)
            if iou > best_iou:
                best_iou = iou
                best_target_idx = target_idx

        if best_iou > iou_threshold and best_target_idx != -1:
            tp += 1
            fp -= 1
            used_targets.add(best_target_idx)

    fn = len(target_boxes) - len(used_targets)
    return tp, fp, fn


def compute_fpr(predictions, targets, confidence_threshold=0.5, img_size=800):
    fp = 0
    tn = 0

    for pred, target in zip(predictions, targets):
        has_true_objects = len(target["boxes"]) > 0

        pred_scores = (
            pred["scores"].cpu().numpy()
            if isinstance(pred["scores"], torch.Tensor)
            else pred["scores"]
        )
        pred_has_objects = np.sum(pred_scores > confidence_threshold) > 0

        if not has_true_objects and not pred_has_objects:
            tn += 1
        elif not has_true_objects and pred_has_objects:
            fp += 1

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return fpr, fp, tn


def compute_recall(
    predictions, targets, confidence_threshold=0.5, iou_threshold=0.5, img_size=800
):
    tp_total = 0
    fn_total = 0

    for pred, target in zip(predictions, targets):
        pred_scores = (
            pred["scores"].cpu().numpy()
            if isinstance(pred["scores"], torch.Tensor)
            else pred["scores"]
        )
        pred_boxes = (
            pred["boxes"].cpu().numpy()
            if isinstance(pred["boxes"], torch.Tensor)
            else pred["boxes"]
        )
        target_boxes = (
            target["boxes"].cpu().numpy()
            if isinstance(target["boxes"], torch.Tensor)
            else target["boxes"]
        )

        filtered_mask = pred_scores > confidence_threshold
        pred_boxes_filtered = pred_boxes[filtered_mask]

        target_boxes_pixel = np.array(
            [denormalize_box(box, img_size, img_size) for box in target_boxes]
        )
        pred_boxes_pixel = np.array(
            [denormalize_box(box, img_size, img_size) for box in pred_boxes_filtered]
        )

        if len(target_boxes_pixel) > 0:
            tp, fp, fn = match_predictions_to_targets(
                pred_boxes_pixel, target_boxes_pixel, iou_threshold
            )
            tp_total += tp
            fn_total += fn

    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    return recall, tp_total, fn_total


def compute_all_metrics(
    predictions, targets, confidence_threshold=0.5, iou_threshold=0.5, img_size=800
):
    fpr, fp, tn = compute_fpr(predictions, targets, confidence_threshold, img_size)
    recall, tp, fn = compute_recall(
        predictions, targets, confidence_threshold, iou_threshold, img_size
    )
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "fpr": fpr,
        "recall": recall,
        "precision": precision,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
