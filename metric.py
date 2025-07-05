import sys
import typing as t
import matplotlib.pyplot as plt
import numpy as np


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
