import abc
import sys
import time
from collections import OrderedDict
from functools import reduce

import numba
import numpy as np
import copy
from copy import deepcopy

from det3d.core.bbox import box_np_ops
import cv2
import math
import torch
import numba as nb



def points_random_flip(points, probability=0.5, flip_coor=None):
    # x flip 
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )
    if enable:
        points[:, 1] = -points[:, 1]
    
    # y flip 
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )
    if enable:
        if flip_coor is None:
            points[:, 0] = -points[:, 0]
        else:
            points[:, 0] = flip_coor * 2 - points[:, 0]
    
    return points


def points_random_jitter(points, probability=0.5, sigma=0.01, clip=0.05):
    # x flip 
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )
    if enable:
        jitter_noise = np.clip(sigma * np.random.randn(points.shape[0], 3), -1*clip, clip)
        points[:, 0:3] += jitter_noise
    
    return points


def points_global_rotation(points, rotation=np.pi / 4):
    if not isinstance(rotation, list):
        rotation = [-rotation, rotation]
    noise_rotation = np.random.uniform(rotation[0], rotation[1])
    points[:, :3] = box_np_ops.rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2
    )
    return points


def points_global_scaling_v2(points, min_scale=0.95, max_scale=1.05):
    noise_scale = np.random.uniform(min_scale, max_scale)
    # print("==> noise_scale: ", noise_scale)
    
    points[:, :3] *= noise_scale
    return points


def points_global_translate_(points, noise_translate_std):
    """
    Apply global translation to points.
    """

    if not isinstance(noise_translate_std, (list, tuple, np.ndarray)):
        noise_translate_std = np.array(
            [noise_translate_std, noise_translate_std, noise_translate_std]
        )

    if all([e == 0 for e in noise_translate_std]):
        return points

    noise_translate = np.array(
        [
            np.random.normal(0, noise_translate_std[0], 1),
            np.random.normal(0, noise_translate_std[1], 1),
            np.random.normal(0, noise_translate_std[0], 1),
        ]
    ).T

    # print("==> noise_translate: ", noise_translate)
    points[:, :3] += noise_translate

    return points


def points_global_flip(points, probability=0.5):
    """
    点(x0, y0)沿着直线y=Ax的对称点坐标(x, y)为
    x = x0 - 2A * ( (A*x0 - y0) / (A^2 + 1) )
    y = y0 +  2 * ( (A*x0 - y0) / (A^2 + 1) )
    """
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )

    # print("==> enable:", enable)

    if enable:
        # pos or neg
        # pos_neg = np.random.uniform(0, 1) >= 0
        pos_neg = np.random.choice(
            [1., -1.0], replace=False, p=[0.5, 0.5]
        )

        # random theta
        # 0 < theta < pi/2
        theta = 0.5 * np.pi * np.random.uniform(0, 1)
        
        # 0 < theta < pi/2 or -pi/2 < theta < 0 
        theta = pos_neg * theta

        A = np.tan(theta)
        # print("==> A:", A)

        points_x0 = deepcopy(points[:, 0])
        points_y0 = deepcopy(points[:, 1])
        denominator = A ** 2 + 1
        numerator = A * points_x0 - points_y0

        points[:, 0] = points_x0 - 2 * A * ( numerator / denominator) 
        points[:, 1] = points_y0 + 2 * ( numerator / denominator) 

    return points

