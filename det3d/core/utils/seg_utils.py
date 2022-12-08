from os import name
import numpy as np
# from det3d.datasets.semantickitti.semkitti_common import learning_map
import pickle
import os
import numba as nb

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iou_func(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop_func(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    return hist


@nb.jit(nopython=True)
def nb_get_statistics_in_frame(output, target, num_class):
    for i in range(0, num_class):
        if i == 0:
            continue
        else: 
            seen = np.sum(target == i)
            correct = np.sum((target == i) & (output == target))
            positive = np.sum(output == i)

    return seen, correct, positive


