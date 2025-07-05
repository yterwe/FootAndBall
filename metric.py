import sys
import typing as t
import xml.etree.ElementTree as ET
import numpy as np
import torch
from sklearn.metrics import average_precision_score
import numpy as np

BALL_LABEL = 1
PLAYER_LABEL = 2
BALL_BBOX_SIZE = 40  # Adjust if needed


def IoU_box(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
    box2_area = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def compute_ap_map(detections, ground_truths, iou_threshold=0.5):
    """
    Compute AP for each class and mAP.
    """
    aps = {}
    class_ids = [BALL_LABEL, PLAYER_LABEL]

    for class_id in class_ids:
        scores = []
        matches = []

        n_positives = 0

        for det, gt in zip(detections, ground_truths):
            det_boxes = [b for b, l in zip(det["boxes"], det["labels"]) if l == class_id]
            det_scores = [s for s, l in zip(det["scores"], det["labels"]) if l == class_id]
            gt_boxes = [b for b, l in zip(gt["boxes"], gt["labels"]) if l == class_id]

            n_positives += len(gt_boxes)
            matched = [False] * len(gt_boxes)

            for box, score in zip(det_boxes, det_scores):
                iou_max = 0.0
                matched_idx = -1
                for i, gt_box in enumerate(gt_boxes):
                    iou = IoU_box(box, gt_box)
                    if iou > iou_max:
                        iou_max = iou
                        matched_idx = i
                if iou_max >= iou_threshold and matched_idx != -1 and not matched[matched_idx]:
                    matches.append(1)  # True positive
                    matched[matched_idx] = True
                else:
                    matches.append(0)  # False positive
                scores.append(score)

        if len(matches) == 0 or n_positives == 0:
            aps[class_id] = 0.0
            continue

        # Sort by score descending
        scores = np.array(scores)
        matches = np.array(matches)
        sorted_idxs = np.argsort(-scores)
        matches = matches[sorted_idxs]

        tp_cumsum = np.cumsum(matches)
        fp_cumsum = np.cumsum(1 - matches)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / (n_positives + 1e-6)

        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = precisions[recalls >= t].max() if np.any(recalls >= t) else 0
            ap += p / 11.0
        aps[class_id] = ap

    aps["mAP"] = np.mean(list(aps.values()))
    return aps


def getGT(xgtf_path: str) -> t.List[t.Dict[str, torch.Tensor]]:
    """
    Parses the .xgtf ground truth file and returns a list of frame-wise dicts.
    Each dict contains 'boxes': Tensor[N,4], 'labels': Tensor[N]
    """
    tree = ET.parse(xgtf_path)
    root = tree.getroot()
    gt_by_frame = {}

    for obj in root.findall('.//{http://lamp.cfar.umd.edu/viper#}object'):
        name = obj.get('name')
        framespan = obj.get('framespan')
        if framespan is None:
            continue
        frame_start, frame_end = map(int, framespan.split(':'))

        if name == "BALL":
            for attr in obj.findall('.//{http://lamp.cfar.umd.edu/viper#}attribute[@name="BallPos"]'):
                for point in attr.findall('{http://lamp.cfar.umd.edu/viperdata#}point'):
                    frame_id = int(point.attrib['framespan'].split(':')[0])
                    x = int(point.attrib['x'])
                    y = int(point.attrib['y'])
                    half = BALL_BBOX_SIZE // 2
                    box = [x - half, y - half, x + half, y + half]

                    if frame_id not in gt_by_frame:
                        gt_by_frame[frame_id] = {'boxes': [], 'labels': []}
                    gt_by_frame[frame_id]['boxes'].append(box)
                    gt_by_frame[frame_id]['labels'].append(BALL_LABEL)

        elif name == "Person":
            for attr in obj.findall('.//{http://lamp.cfar.umd.edu/viper#}attribute[@name="LOCATION"]'):
                for bbox in attr.findall('{http://lamp.cfar.umd.edu/viperdata#}bbox'):
                    frame_id = int(bbox.attrib['framespan'].split(':')[0])
                    x = int(bbox.attrib['x'])
                    y = int(bbox.attrib['y'])
                    width = int(bbox.attrib['width'])
                    height = int(bbox.attrib['height'])
                    box = [x, y, x + width, y + height]

                    if frame_id not in gt_by_frame:
                        gt_by_frame[frame_id] = {'boxes': [], 'labels': []}
                    gt_by_frame[frame_id]['boxes'].append(box)
                    gt_by_frame[frame_id]['labels'].append(PLAYER_LABEL)

    ground_truths = []
    frame_ids = sorted(gt_by_frame.keys())
    #print(f"[DEBUG] Ground truth frame range: {frame_ids[0]} ~ {frame_ids[-1]}")

    for frame_id in sorted(gt_by_frame.keys()):
        boxes = [list(map(float, box)) for box in gt_by_frame[frame_id]['boxes']]
        labels = list(gt_by_frame[frame_id]['labels'])
        ground_truths.append({'boxes': boxes, 'labels': labels})
    return ground_truths, frame_ids[0]


def IoU(result, gt):
    """Compute IoU between two boolean masks"""
    overlap = result * gt
    union = result + gt
    return overlap.sum() / float(union.sum())


if __name__ == "__main__":
    print("This module is intended to be imported and used by other scripts.")
