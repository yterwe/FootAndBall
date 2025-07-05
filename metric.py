import sys
import typing as t
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import average_precision_score

def getGT(data, img_height=1080, img_width=1920) -> t.Tuple[np.array, int]:
    """
    Returns
    -------
    gt :
        ground truth array
    cntgt :
        number of objects
    """
    gt = np.zeros((img_width, img_height), dtype=bool)

    for _, row in enumerate(data):
        X1 = min(int(row[0]), img_width)
        Y1 = min(int(row[1]), img_height)
        X2 = min(int(row[2]), img_width)
        Y2 = min(int(row[3]), img_height)

        for i in range(X1, X2 - 1):
            for j in range(Y1, Y2 - 1):
                gt[i][j] = 1

    return gt, len(data)

def compute_ap_map(detections, ground_truths, iou_threshold=0.5, num_classes=2):
    """
    detections: list of dicts with keys ['boxes', 'scores', 'labels']
    ground_truths: list of dicts with keys ['boxes', 'labels']
    returns: dict {class_id: ap_value, ..., 'mAP': float}
    """

    aps = {}

    for class_id in range(num_classes):
        y_true = []
        y_scores = []

        for det, gt in zip(detections, ground_truths):
            # Get predicted boxes and scores for this class
            det_boxes = [b for b, l in zip(det["boxes"], det["labels"]) if l == class_id]
            det_scores = [s for s, l in zip(det["scores"], det["labels"]) if l == class_id]

            # Get GT boxes for this class
            gt_boxes = [b for b, l in zip(gt["boxes"], gt["labels"]) if l == class_id]

            matched = [False] * len(gt_boxes)

            for box, score in zip(det_boxes, det_scores):
                iou_max = 0.0
                matched_idx = -1
                for i, gt_box in enumerate(gt_boxes):
                    iou = IoU_box(box, gt_box)
                    if iou > iou_max:
                        iou_max = iou
                        matched_idx = i

                if iou_max >= iou_threshold and not matched[matched_idx]:
                    y_true.append(1)
                    matched[matched_idx] = True
                else:
                    y_true.append(0)
                y_scores.append(score)

            # False negatives (missed GTs)
            for m in matched:
                if not m:
                    y_true.append(1)
                    y_scores.append(0)

        if len(y_true) == 0 or len(set(y_true)) == 1:
            aps[class_id] = 0.0
        else:
            aps[class_id] = average_precision_score(y_true, y_scores)

    aps['mAP'] = np.mean(list(aps.values()))
    return aps

def IoU(result, gt):
    # result, gt - np boolean arrays
    overlap = result * gt  # Logical AND
    union = result + gt  # Logical OR

    IOU = overlap.sum() / float(union.sum())  # Treats "True" as 1,
    # sums number of Trues
    # in overlap and union
    # and divides
    return IOU


if __name__ == "__main__":
    ...
