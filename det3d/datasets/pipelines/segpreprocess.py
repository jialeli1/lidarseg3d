import numpy as np
import numba as nb
import pickle
import time
from copy import deepcopy
import torch

from det3d.core.sampler import segpreprocess as segprep
from det3d.core.input.voxel_generator import VoxelGenerator
from ..registry import PIPELINES


@PIPELINES.register_module
class SegPreprocess(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = cfg.shuffle_points
        
        self.mode = cfg.mode

        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_translate_std = cfg.get('global_translate_std', 0)
            
            # self.npoints = cfg.get("npoints", -1)

        self.npoints = cfg.get("npoints", -1)

        self.no_augmentation = cfg.get('no_augmentation', False)

    def __call__(self, res, info):

        res["mode"] = self.mode

        if res["type"] in ["SemanticKITTIDataset", "SemanticWaymoDataset", "SemanticNuscDataset"]:
            points = res["lidar"]["points"]
        else:
            raise NotImplementedError

        if self.mode in ["train"]:
            anno_dict = res["lidar"]["annotations"]

            gt_dict = {
                "point_sem_labels": anno_dict["point_sem_labels"], # shaped in (N, )
                "point_inst_labels": anno_dict["point_inst_labels"],
            }

        if self.mode == "train" and not self.no_augmentation:
            # augmentation on points for semantic segmentation            
            points = segprep.points_random_flip(points)

            points = segprep.points_global_rotation(points, rotation=self.global_rotation_noise)

            points = segprep.points_global_scaling_v2(points, *self.global_scaling_noise)

            points = segprep.points_global_translate_(points, noise_translate_std=self.global_translate_std)


        elif self.no_augmentation:
            pass


        if self.shuffle_points:
            # NOTE: need operating on BOTH points and sem_labels
            idx = np.arange(points.shape[0])
            np.random.shuffle(idx)

            points = points[idx, :]
            points_shuffle_idx = idx

            if self.mode == "train":
                # lable shuffle for training
                point_sem_labels = gt_dict["point_sem_labels"][idx]
                point_inst_labels = gt_dict["point_inst_labels"][idx]


                # NOTE: 
                # Combining the points and labels for facilitating the production of voxel labels in SegVoxelization.
                # We temporarily add 1 to point_sem_labels for distinguishing the padded zeros in SegVoxelization, which will be recovered in SegAssignLabel.
                points_with_labels = np.concatenate([
                    points, 
                    point_sem_labels[:, None].astype(np.float32) + 1, 
                    point_inst_labels[:, None].astype(np.float32), # NOTE: We do not ensure the reasonableness of point_inst_labels!!!
                ], axis=-1)

                gt_dict["point_sem_labels"] = point_sem_labels
                gt_dict["point_inst_labels"] = point_inst_labels
        else:
            # pass
            points_shuffle_idx = np.arange(points.shape[0])


        # depcopy before the number of points is changed.
        all_points = deepcopy(points)



        if points.shape[0] > self.npoints and self.npoints > 0:
            points = points[:self.npoints, :]
            points_shuffle_idx = points_shuffle_idx[:self.npoints] 

            if self.mode in ["train"]:
                points_with_labels = points_with_labels[:self.npoints]
                gt_dict["point_sem_labels"] = gt_dict["point_sem_labels"][:self.npoints]
                gt_dict["point_inst_labels"] = gt_dict["point_inst_labels"][:self.npoints]


        res["lidar"]["points"] = points 
        res["lidar"]["all_points"] = all_points 
        res["lidar"]["points_shuffle_idx"] = points_shuffle_idx 

        if self.mode in ["train"]:
            res["lidar"]["annotations"] = gt_dict
            res["lidar"]["points_with_labels"] = points_with_labels

        return res, info



@PIPELINES.register_module
class SegVoxelization(object):
    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = [cfg.max_voxel_num, cfg.max_voxel_num] if isinstance(cfg.max_voxel_num, int) else cfg.max_voxel_num

        self.double_flip = cfg.get('double_flip', False)

        self.tta_flag = cfg.get('tta_flag', False)
        self.num_tta_tranforms = cfg.get('num_tta_tranforms', -1)

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )

    def __call__(self, res, info):
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size

        if res["mode"] == "train":
            max_voxels = self.max_voxel_num[0]
        else:
            max_voxels = self.max_voxel_num[1]


        if res["mode"] in ["train"]:
            points_to_voxelize = res["lidar"]["points_with_labels"]
        else:
            points_to_voxelize = res["lidar"]["points"]

        voxels, coordinates, num_points = self.voxel_generator.generate(
            points_to_voxelize, max_voxels=max_voxels 
        )
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        res["lidar"]["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=grid_size,
            range=pc_range,
            size=voxel_size
        )



        # -------begin: double_flip based TTA for det3d preserved from CenterPoint--------------
        double_flip = self.double_flip and (res["mode"] != 'train')
        if double_flip:
            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["yflip_points"],
                max_voxels=max_voxels 
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["yflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["xflip_points"],
                max_voxels=max_voxels 
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["xflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["double_flip_points"],
                max_voxels=max_voxels 
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["double_flip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )            
        # -------end: double_flip based TTA for det3d preserved from CenterPoint --------------




        # -------begin: TTA for seg3d from SDSeg3d--------------
        # tta_flag = self.tta_flag and (res["mode"] != 'train')
        tta_flag = self.tta_flag
        if tta_flag:
            for i in range(1, self.num_tta_tranforms):
                point_key_i = "tta_%s_points" %i
                tta_voxels, tta_coordinates, tta_num_points = self.voxel_generator.generate(
                    res["lidar"][point_key_i],
                    max_voxels=max_voxels
                )
                tta_num_voxels = np.array([tta_voxels.shape[0]], dtype=np.int64)

                voxel_key_i = "tta_%s_voxels" %i
                res["lidar"][voxel_key_i] = dict(
                    voxels=tta_voxels,
                    coordinates=tta_coordinates,
                    num_points=tta_num_points,
                    num_voxels=tta_num_voxels,
                    shape=grid_size,
                    range=pc_range,
                    size=voxel_size
                )
        # -------end: TTA for seg3d from SDSeg3d--------------


        return res, info



@nb.jit()
def nb_encode_major_value_as_label_fast(voxel_labels, encoded_labels, ignore_id=0):
    """
    voxel_labels: (N_voxel, N_point)
    encoded_labels: (N_voxel, ) zeors
    返回出现次数最多的非零元素
    NOTE: to choose parameters in @nb.jit(nopython=True, cache=True, parallel=False)
    np.unique(return_counts=True) is not supported by @nb.jit(nopython=True).
    """
    for i in range(voxel_labels.shape[0]):
        # NOTE: 这里去掉0来统计是因为之前+1
        cur_voxel_labels = voxel_labels[i][voxel_labels[i] > 0]
        # 统计唯一的元素及其出现的次数
        # u, counts = np.unique(cur_voxel_labels, return_counts=True)
        u = np.unique(cur_voxel_labels)
        counts = np.zeros_like(u)

        # try numba style
        # much faster
        for j, u_j in enumerate(u):
            for k, label_k in enumerate(cur_voxel_labels):
                if label_k == u_j:
                    counts[j] += 1
            
        
        # 索引次数最多的那个元素
        most_idx = np.argmax(counts)
        encoded_labels[i] = u[most_idx]

    return encoded_labels


@nb.jit()
def nb_encode_compact_value_as_label_fast(voxel_labels, encoded_labels, ignore_id=0):
    """
    voxel_labels: (N_voxel, N_point)
    encoded_labels: (N_voxel, ) zeors
    返回之里面只有1种类别的体素标签, 对那些有多义性的体素, 标签设为0 (+1)
    NOTE: to choose parameters in @nb.jit(nopython=True, cache=True, parallel=False)
    np.unique(return_counts=True) is not supported by @nb.jit(nopython=True).
    """
    for i in range(voxel_labels.shape[0]):
        # NOTE: 这里去掉0来统计是因为之前+1
        cur_voxel_labels = voxel_labels[i][voxel_labels[i] > 0]
        # 统计唯一的元素及其出现的次数
        # u, counts = np.unique(cur_voxel_labels, return_counts=True)
        u = np.unique(cur_voxel_labels)

        # debug一下
        # print("==> cur_voxel_labels: {}, u: {}".format(cur_voxel_labels, u))
        # 当出现两个及以上的类别的时候，忽略掉这个体素的损失，label设置为0.
        # 避免奇怪的梯度产生.
        # print("==> u.shape: ", u.shape) # (1, )
        if u.shape[0] > 1:
            encoded_labels[i] = ignore_id + 1
            # print("==> set as ignored for ", u)
        elif u.shape[0] == 1:
            encoded_labels[i] = u[0]
        else:
            raise NotImplementedError

    return encoded_labels



@PIPELINES.register_module
class SegAssignLabel(object):
    def __init__(self, **kwargs):

        assigner_cfg = kwargs["cfg"]
        self.voxel_label_enc = assigner_cfg.voxel_label_enc


    def __call__(self, res, info):
        """
        voxel_with_label: [N_voxel, N_point, d1+d2]
        voxel_features: [N_voxel, N_point, d1]
        voxel_sem_labels: [N_voxel, N_point, d2]
        """
        example = {}
        if res["mode"] in ["train"]:
            dim_feature, dim_semlabel = info["dim"]["points"], info["dim"]["sem_labels"]
            voxel_with_label = res["lidar"]["voxels"]["voxels"]
        
            split_arrs = np.split(voxel_with_label, [dim_feature, dim_feature + dim_semlabel], axis=-1)

            # [N_voxel, N_pointpervoxel, d1]
            voxel_features = split_arrs[0]
            # update voxel feature for training
            res["lidar"]["voxels"]["voxels"] = voxel_features


            # [N_voxel, N_pointpervoxel, 1] -> [N_voxel, N_pointpervoxel]
            voxel_sem_labels = np.squeeze(split_arrs[1], axis=-1)

            # different strategies for voxel_sem_labels 
            # [N_voxels, N_pointpervoxel] -> [N_voxels, ]
            # NOTE: We restore the original label by subtracting 1, which is mentioned in SegPreprocess.
            if self.voxel_label_enc == "major_value":
                # nb_encode_major_value_as_label_fast, numba style, 0.04s
                init_zeros = np.zeros(voxel_sem_labels.shape[0], dtype=voxel_sem_labels.dtype)
                # start_time = time.time()
                major_enc_sem_labels = nb_encode_major_value_as_label_fast(voxel_labels=voxel_sem_labels, encoded_labels=init_zeros)
                enc_sem_labels = major_enc_sem_labels - 1 
                # end_time = time.time()
                # print("==> major_value encoding by numba, time:%s"  % (end_time-start_time))  
            
                example.update({
                    "voxel_sem_labels": enc_sem_labels,
                    "point_sem_labels": res["lidar"]["annotations"]["point_sem_labels"], 
                })

            elif self.voxel_label_enc == "compact_value":
                # Voxels containing multiple semantic categories are ignored (as 0).
                init_zeros = np.zeros(voxel_sem_labels.shape[0], dtype=voxel_sem_labels.dtype)
                comp_enc_sem_labels = nb_encode_compact_value_as_label_fast(voxel_labels=voxel_sem_labels, encoded_labels=init_zeros)
                enc_sem_labels = comp_enc_sem_labels - 1 
            
                example.update({
                    "voxel_sem_labels": enc_sem_labels,
                    "point_sem_labels": res["lidar"]["annotations"]["point_sem_labels"], 
                })

            else: 
                raise NotImplementedError


        else:
            pass
        
        res["lidar"]["targets"] = example

        return res, info


