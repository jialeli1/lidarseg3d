import os.path as osp
from turtle import shape
import warnings
import numpy as np
from functools import reduce

import pycocotools.mask as maskUtils

from pathlib import Path
from copy import deepcopy
from det3d import torchie
from det3d.core import box_np_ops
import pickle 
import os

from det3d.core.sampler import segpreprocess 
from ..registry import PIPELINES



def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points



def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def read_file(path, tries=2, num_point_feature=4, painted=False):
    if painted:
        dir_path = os.path.join(*path.split('/')[:-2], 'painted_'+path.split('/')[-2])
        painted_path = os.path.join(dir_path, path.split('/')[-1]+'.npy')
        points =  np.load(painted_path)
        points = points[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] # remove ring_index from features 
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, painted=False):
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), painted=painted).T
    points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T


def read_single_semnusc_sweep(sweep, num_point_feature=5, painted=False, remove_close_flag=False):
    # NOTE: remove_close() will make the points.shape[0] and label.shape[0] mismatched.

    points_sweep = read_file(str(sweep["lidar_path"]), num_point_feature=num_point_feature, painted=painted).T
    
    if remove_close_flag:
        min_distance = 1.0
        points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T



def read_single_waymo(obj):
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)
    
    return points 

def read_single_waymo_sweep(sweep):
    obj = get_obj(sweep['path'])

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T # 5 x N

    nbr_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot( 
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]

    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
    
    return points_sweep.T, curr_times.T



def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 



@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)
        self.use_img = kwargs.get("use_img", False)

    def __call__(self, res, info):
        """
        The semantic segmentation related datasets are denoted with prefix "Semantic".
        """

        res["type"] = self.type

        # nusc det3d case preserved from CenterPoint
        if self.type == "NuScenesDataset":

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), painted=res["painted"])

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )

            for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep, painted=res["painted"])
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])
        
        # waymo det3d case preserved from CenterPoint
        elif self.type == "WaymoDataset":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            obj = get_obj(path)
            points = read_single_waymo(obj)
            res["lidar"]["points"] = points

            if nsweeps > 1: 
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should be equal to the list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )

                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])
        
        # kitti seg3d
        elif self.type in ["SemanticKITTIDataset"]:
            path = info["path"]
            # simplify loading of SemanticKITTIDataset

            points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = None
            res["lidar"]["combined"] = None
            # info["dim"]["points"] = points.shape[-1]

        # waymo seg3d
        elif self.type in ["SemanticWaymoDataset"]:
            # TODO: Support SemanticWaymoDataset
            pass

        # nusc seg3d
        elif self.type == "SemanticNuscDataset":

            lidar_path = Path(info["lidar_path"])
            nsweeps = res["lidar"]["nsweeps"]

            # print("==> lidar_path: {}", lidar_path)
            # ==> lidar_path: {} data/SemanticNusc/samples/LIDAR_TOP/n008-2018-08-28-16-16-48-0400__LIDAR_TOP__1535488338796590.pcd.bin

            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
            res["lidar"]["points"] = points
 

        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        self.with_bbox = with_bbox


    def __call__(self, res, info):
        """
        The semantic segmentation related datasets are denoted with prefix "Semantic".
        """

        # nusc det3d case preserved from CenterPoint
        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            res["lidar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }
        # waymo det3d case preserved from CenterPoint
        elif res["type"] == 'WaymoDataset' and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
            }
        # kitti seg3d
        elif res["type"] == 'SemanticKITTIDataset':
            path = info["path"]
            learning_map = info["learning_map"]

            # get *.label path from *.bin path
            label_path = path.replace("velodyne", "labels").replace(".bin", ".label")
            # all_labels = np.fromfile(label_path, dtype=np.int32).reshape(-1)
            annotated_data = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
            
            # semantic labels
            sem_labels = annotated_data & 0xFFFF
            # instance labels
            # inst_labels = annotated_data 
            inst_labels = annotated_data.astype(np.float32) 

            # label mapping 
            sem_labels = (np.vectorize(learning_map.__getitem__)(sem_labels)).astype(np.float32)

            res["lidar"]["annotations"] = {
                "point_sem_labels": sem_labels,
                "point_inst_labels": inst_labels,
            }
            # info["dim"]["sem_labels"] = 1
        
        # waymo seg3d
        elif res["type"] == 'SemanticWaymoDataset':
            # TODO: Support SemanticWaymoDataset
            pass

        # nusc seg3d
        elif res["type"] == 'SemanticNuscDataset':
            learning_map = res['learning_map']
            data_root = 'data/SemanticNusc'
            lidarseg_labels_filename = os.path.join(data_root, info['seganno_path'])
            # print("==> lidarseg_labels_filename: ", lidarseg_labels_filename)
            # data/SemanticNusc/lidarseg/v1.0-trainval/e6ca15bc5803457cba8d05f5e78f4e40_lidarseg.bin

            point_sem_labels = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape(-1)
            point_sem_labels = np.vectorize(learning_map.__getitem__)(point_sem_labels).astype(np.float32)
            
            # NOTE: We have only parsed the semantic labels. If you want to use the instance labels, please check them carefully.
            point_inst_labels = np.zeros_like(point_sem_labels)
            
            res["lidar"]["annotations"] = {
                "point_sem_labels": point_sem_labels,
                "point_inst_labels":point_inst_labels
            }

        else:
            pass 

        return res, info


