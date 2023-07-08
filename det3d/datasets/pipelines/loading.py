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
import cv2

from ..registry import PIPELINES



def select_points_in_frustum(points_2d, x1, y1, x2, y2):
    """
    for SemanticKITTI
    Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
    :param points_2d: point cloud projected into 2D
    :param points_3d: point cloud
    :param x1: left bound
    :param y1: upper bound
    :param x2: right bound
    :param y2: lower bound
    :return: points (2D and 3D) that are in the frustum
    """
    keep_ind = (points_2d[:, 0] > x1) * \
                (points_2d[:, 1] > y1) * \
                (points_2d[:, 0] < x2) * \
                (points_2d[:, 1] < y2)

    return keep_ind




def read_calib_semanticKITTI(calib_path):
    """
    for SemanticKITTI
    :param calib_path: Path to a calibration text file.
    :return: dict with calibration matrices.
    """
    calib_all = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if line == '\n':
                break
            key, value = line.split(':', 1)
            calib_all[key] = np.array([float(x) for x in value.split()])

    # reshape matrices
    calib_out = {}
    calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
    calib_out['Tr'] = np.identity(4)  # 4x4 matrix
    calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

    return calib_out



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

            if self.use_img:
                # ..../sequences/00/velodyne/004540.bin
                # the calib.txt is shared by a sequence
                calib_path = path[:-11].replace("velodyne", "calib.txt")
                # print("==> path: {}, calib_path: {}".format(path, calib_path))
                # ==> path: data/SemanticKITTI/dataset/sequences/01/velodyne/000746.bin, calib_path: data/SemanticKITTI/dataset/sequences/01/calib.txt

                calib = read_calib_semanticKITTI(calib_path)
                proj_matrix = np.matmul(calib["P2"], calib["Tr"]) # shape: (3, 4)

                # keep only points in front of the vehicle
                pts_uv_all = np.ones([points.shape[0], 3]).astype(np.float32) * -100



                points_hcoords = np.concatenate([points[:, :3], np.ones([points.shape[0], 1], dtype=np.float32)], axis=1)
                img_points = (proj_matrix @ points_hcoords.T).T
                img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points

                # fix the image size
                im_width, im_height = 1224, 370
                # frustum_mask & front_mask
                frustum_mask = select_points_in_frustum(img_points, 0, 0, im_width, im_height)
                mask = frustum_mask & (points[:, 0] > 0)

                # pts_uv_all: shape [npoints, 3], 3 for [cam_id, idx_of_width, idx_of_height]
                # cam_id starts from 1, following the waymo style
                pts_uv_all[mask, 0] = 1
                pts_uv_all[mask, 1:3] = img_points[mask, 0:2]

                
                # (npoints, 3), [cam_id, idx_of_width, idx_of_height]
                res["lidar"]["points_cp"] = pts_uv_all


        # waymo seg3d
        elif self.type in ["SemanticWaymoDataset"]:
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]

            # obj = get_obj(path)
            # points = read_single_waymo(obj)

            example_obj = get_obj(path)
            points = read_single_waymo(example_obj)

            res["lidar"]["points"] = points
            # save num_points_of_top_lidar for selecting ri_return1/ri_return2
            res["metadata"]["num_points_of_top_lidar"] = example_obj["lidars"]["num_points_of_top_lidar"]


            if self.use_img:
                # waymo dataset provides points_cp (pointwise camera projection)
                points_cp = example_obj["lidars"]["points_cp"] # (npoints, 3), [cam_id, idx_of_width, idx_of_height]
                res["lidar"]["points_cp"] = points_cp



        # nusc seg3d
        elif self.type == "SemanticNuscDataset":

            lidar_path = Path(info["lidar_path"])
            nsweeps = res["lidar"]["nsweeps"]

            # print("==> lidar_path: {}", lidar_path)
            # ==> lidar_path: {} data/SemanticNusc/samples/LIDAR_TOP/n008-2018-08-28-16-16-48-0400__LIDAR_TOP__1535488338796590.pcd.bin

            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
            res["lidar"]["points"] = points
 
            # get points_cp (pointwise camera projection) before augmentation
            if self.use_img:
                # cam_name = res['cam']['names']
                cam_chan = res["cam"]["chan"]

                im_shape = (900, 1600, 3)

                # final shape: (npoints, 3)
                # [cam_id, idx_of_width, idx_of_height]
                # without normalization here
                pts_uv_all = np.ones([points.shape[0], 3]).astype(np.float32) * -100

                for cam_id, cam_sensor in enumerate(cam_chan):
                    cam_from_global = info["cams_from_global"][cam_sensor]
                    cam_intrinsic = info["cam_intrinsics"][cam_sensor]

                    # lidar to global
                    ref_to_global = info["ref_to_global"]
                    pts_hom = np.concatenate([points[:, :3], np.ones([points.shape[0], 1])], axis=1)
                    pts_global = ref_to_global.dot(pts_hom.T)  # 4 * N

                    # global to cam
                    pts_cam = cam_from_global.dot(pts_global)[:3, :]  # 3 * N

                    # cam to uv
                    pts_uv = view_points(pts_cam, np.array(cam_intrinsic), normalize=True).T  # N * 3

                    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
                    mask = (pts_cam[2, :] > 0) & (pts_uv[:, 0] > 1) & (pts_uv[:, 0] < im_shape[1] - 1) & (
                            pts_uv[:, 1] > 1) & (pts_uv[:, 1] < im_shape[0] - 1)

                    # mask = (pts_cam[2, :] > 0) \
                    #     & (pts_uv[:, 0] > 0) & (pts_uv[:, 0] < im_shape[1] - 1) \
                    #     & (pts_uv[:, 1] > 0) & (pts_uv[:, 1] < im_shape[0] - 1)

                    pts_uv_all[mask, :2] = pts_uv[mask, :2]
                    # NOTE: cam_id starts from 1, following the waymo style
                    pts_uv_all[mask, 2] = float(cam_id) + 1

                # reformat as semantciwaymo: [cam_id, idx_of_width, idx_of_height]
                # NOTE: BE CAREFUL
                points_cp = pts_uv_all[:, [2, 0, 1]]

                # (npoints, 3), [cam_id, idx_of_width, idx_of_height]
                res["lidar"]["points_cp"] = points_cp


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
            # TYPE_UNDEFINED: 0
            assert info["seg_annotated"], "==> Seg annotated frames only!"
            anno_path = info['anno_path']
            obj = get_obj(anno_path)
            semantic_anno = obj["seg_labels"]["points_seglabel"] # (numpoints_toplidar, 2), [ins, sem]

            num_points_top_lidar = semantic_anno.shape[0]
            num_points_all_lidars = res["lidar"]["points"].shape[0]

            assert num_points_top_lidar == res["metadata"]["num_points_of_top_lidar"]["ri_return1"] + res["metadata"]["num_points_of_top_lidar"]["ri_return2"]
            semantic_anno_padded = np.zeros(shape=(num_points_all_lidars, semantic_anno.shape[-1]), dtype=semantic_anno.dtype)
            semantic_anno_padded[:num_points_top_lidar, :] = semantic_anno

            res["lidar"]["annotations"] = {
                "point_sem_labels": semantic_anno_padded[:, 1],
                "point_inst_labels": semantic_anno_padded[:, 0],
            }


        # nusc seg3d
        elif res["type"] == 'SemanticNuscDataset':
            learning_map = res['learning_map']
            data_root = 'data/SemanticNusc'
            # lidarseg_labels_filename: data/SemanticNusc/lidarseg/v1.0-trainval/e6ca15bc5803457cba8d05f5e78f4e40_lidarseg.bin
            lidarseg_labels_filename = os.path.join(data_root, info['seganno_path'])

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




@PIPELINES.register_module
class LoadImageFromFile(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, res, info):

        dataset_type = res["type"]

        if dataset_type == "NuScenesDataset":
            raise NotImplementedError
        
        elif dataset_type in ["WaymoDataset"]:
            raise NotImplementedError
        
        elif dataset_type in ["SemanticKITTIDataset"]:
            path = info["path"]
            # image_2_path:  data/SemanticKITTI/dataset/sequences/01/image_2/000732.png
            image_2_path = path.replace('velodyne', 'image_2').replace('.bin', '.png')

            # reformat as waymo and nusc 
            cam_paths = {'1': image_2_path}
            cam_names = res["cam"]["names"]
            ori_images = [cv2.imread(cam_paths[cam_id]) for cam_id in cam_names]

            res["images"] = ori_images
            

        elif dataset_type in ["SemanticWaymoDataset"]:
            cam_names = res["cam"]["names"] # a list of ['1', '2', '3', '4', '5']
            cam_paths = info["cam_paths"]   # a dict. The key is set as the cam_id   
            ori_images = [cv2.imread(cam_paths[cam_id]) for cam_id in cam_names]

            res["images"] = ori_images

        elif dataset_type in ["SemanticNuscDataset"]:
            # cam_chan: a list of ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
            # cam_names: a list of ['1', '2', '3', '4', '5', '6']
            cam_chan = res["cam"]["chan"]
            cam_names = res["cam"]["names"] 
            

            cam_paths = info["cam_paths"] 
            # img shape for all cameras: (900, 1600, 3)
            ori_images = [cv2.imread(cam_paths[cam_sensor]) for cam_sensor in cam_chan]

            res["images"] = ori_images


        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadImageAnnotations(object):
    def __init__(self, **kwargs):
        self.points_cp_radius = kwargs.get("points_cp_radius", 1)


    def __call__(self, res, info):

        dataset_type = res["type"]

        if dataset_type == "NuScenesDataset":
            raise NotImplementedError
        
        elif dataset_type in ["WaymoDataset"]:
            raise NotImplementedError
            

        elif dataset_type in ["SemanticWaymoDataset", "SemanticNuscDataset", "SemanticKITTIDataset"]:
            # project the pointwise label as the sparse pixelwise label for image branch
            # set this pipline after LoadPointCloudFromFile/LoadImageFromFile/LoadPointCloudAnnotations
            
            cam_names = res["cam"]["names"] # a list of ['1', '2', '3', '4', '5']

            ori_images = res["images"]
            ori_points_cp = res["lidar"]["points_cp"]
            ori_point_sem_labels = res["lidar"]["annotations"]["point_sem_labels"]
            ori_image_sem_maps = []
            for cam_id, ori_image in zip(cam_names, ori_images):
                H, W = ori_image.shape[0], ori_image.shape[1] 
                cur_ori_sem_map = np.zeros((H, W), dtype=ori_image.dtype) # uint8
                
                # [npoints, 3] -> [cur_npoints, 3] 
                point_cam_id_mask = ori_points_cp[:, 0] == int(cam_id)
                cur_ori_points_cp = ori_points_cp[point_cam_id_mask]
                # [cur_npoints,] 
                cur_wid_coords = list(cur_ori_points_cp[:, 1]) # for calling the cv2.circle
                cur_hei_coords = list(cur_ori_points_cp[:, 2])
                cur_sem_labels = list(ori_point_sem_labels[point_cam_id_mask])
                for i in range(len(cur_wid_coords)):
                    if cur_sem_labels[i] > 0:
                        cv2.circle(
                            cur_ori_sem_map, 
                            center=(int(cur_wid_coords[i]), int(cur_hei_coords[i])),
                            radius=self.points_cp_radius,
                            color=int(cur_sem_labels[i]),
                            thickness=-1, 
                            # lineType=cv2.LINE_AA, 
                        )

                ori_image_sem_maps.append(cur_ori_sem_map)


            # add to res
            img_gt_dict = {
                "image_sem_labels": ori_image_sem_maps,
            } 

        else:
            raise NotImplementedError

        res["cam"]["annotations"] = img_gt_dict

        return res, info







