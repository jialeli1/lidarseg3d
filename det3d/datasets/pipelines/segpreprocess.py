import numpy as np
import numba as nb
import pickle
import time
from copy import deepcopy
import torch

from det3d.core.sampler import segpreprocess as segprep
from det3d.core.input.voxel_generator import VoxelGenerator
from ..registry import PIPELINES


import cv2
from torchvision.transforms import ColorJitter
from .img_transforms import RandomCrop, RandomRescale, image_and_points_cp_and_label_resize, image_and_points_cp_and_label_random_horizon_flip, image_input_transform, jpeg_compression




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



@PIPELINES.register_module
class SegImagePreprocess(object):
    def __init__(self, cfg=None, **kwargs):
        # set img aug parameters
        self.shuffle_points = cfg.get('shuffle_points', False)
        self.random_horizon_flip = cfg.get('random_horizon_flip', False)
        random_color_jitter_cfg = cfg.get('random_color_jitter_cfg', None)
        if random_color_jitter_cfg is not None:
            self.random_color_jitter = True
            self.random_color_jitter_transform = ColorJitter(**random_color_jitter_cfg)
        else: 
            self.random_color_jitter = False
            self.random_color_jitter_transform = None


        random_jpeg_compression_cfg = cfg.get('random_jpeg_compression_cfg', None)
        if random_jpeg_compression_cfg is not None:
            self.random_jpeg_compression = True
            self.random_jpeg_compression_cfg = random_jpeg_compression_cfg
        else:
            self.random_jpeg_compression = False

        random_rescale_cfg = cfg.get('random_rescale_cfg', None)
        if random_rescale_cfg is not None:
            self.random_rescale = True
            self.random_rescale_transform = RandomRescale(**random_rescale_cfg)
        else:
            self.random_rescale = False
            self.random_rescale_transform = None


        random_crop_cfg = cfg.get('random_crop_cfg', None)
        if random_crop_cfg is not None:
            self.random_crop = True
            self.random_crop_transform = RandomCrop(**random_crop_cfg)
        else:
            self.random_crop = False  
            self.random_crop_transform = None 

        self.save_img_for_tta = kwargs.get("save_img_for_tta", False)
        self.no_augmentation = cfg.get('no_augmentation', False)

    def random_color_jitter_wrap(self, img):
        """
        img: [H, W, 3] np.array in BGR mode
        img.dtype should be float for image with pixel value in 0~1
        img.dtype should be uint8 for image with pixel value in 0~255
        """

        # BGR-> RGB, np.array -> a cpu tensor
        img_tensor = torch.Tensor(img[:, :, (2,1,0)]).type(torch.uint8)
        # [H, W, 3] -> [3, H, W]
        img_tensor = img_tensor.permute(2,0,1)

        # jittering
        jittered_img_tensor = self.random_color_jitter_transform(img_tensor)
        # [3, H, W] -> [H, W, 3]
        jittered_img = jittered_img_tensor.permute(1,2,0).numpy()
        
        # RGB -> BGR
        img = jittered_img[:, :, (2,1,0)]

        return img


    def random_rescale_wrap(self, image, points_cp, image_label=None):

        rescale_in_dict = dict(
            image=image,
            points_cp=points_cp,
        )
        if image_label is not None:
            rescale_in_dict["image_label"] = image_label

        rescale_in_dict = self.random_rescale_transform(rescale_in_dict)
        
        image = rescale_in_dict["image"] 
        points_cp = rescale_in_dict["points_cp"]

        rescaled_shape = rescale_in_dict["rescaled_shape"]

        if image_label is not None:
            image_label = rescale_in_dict["image_label"]

        return image, points_cp, image_label


    def random_crop_wrap(self, image, points_cp, image_label=None):

        crop_in_dict = dict(
            image=image,
            points_cp=points_cp,
        )
        if image_label is not None:
            crop_in_dict["image_label"] = image_label

        crop_in_dict = self.random_crop_transform(crop_in_dict)
        
        image = crop_in_dict["image"] 
        points_cp = crop_in_dict["points_cp"]

        cropped_shape = crop_in_dict["cropped_shape"]
        crop_valid = crop_in_dict["crop_valid"]

        if image_label is not None:
            image_label = crop_in_dict["image_label"]

        return image, points_cp, image_label


    def __call__(self, res, info):
        # res["mode"] = self.mode
        mode = res["mode"]
        dataset_type = res["type"]
        if dataset_type == "NuScenesDataset":
            raise NotImplementedError
        
        elif dataset_type in ["WaymoDataset"]:
            raise NotImplementedError
            
        elif dataset_type == "SemanticNuScenesDataset":
            raise NotImplementedError

        elif dataset_type in ["SemanticWaymoDataset", "SemanticNuscDataset", "SemanticKITTIDataset"]:

            points_cp = res["lidar"]["points_cp"] # (npoints, 3), [cam_id, idx_of_width, idx_of_height]
            points = res["lidar"]["points"]
            ori_images = res["images"] 
            assert points_cp.shape[0] == points.shape[0]
            cam_names = res["cam"]["names"] 
            cam_attributes = res["cam"]["attributes"]  
            resized_shape_cv = res["cam"]["resized_shape"]


            # 3: [camid, u, v]
            points_cuv_all = np.ones([points.shape[0], 3]).astype(np.float32) * -100
                
            img_has_annotations = res["cam"]["annotations"] is not None

            # training: resize the multicamera images as the same shape, then do augmentations
            # inference: resize the multicamera images as the same shape
            resized_images = []
            resized_image_sem_labels = []
            for cam_id, ori_image in zip(cam_names, ori_images):
                point_cam_id_mask = points_cp[:, 0] == int(cam_id)
                idx = int(cam_id)-1 
                
                if img_has_annotations:
                    ori_image_sem_label = res["cam"]["annotations"]["image_sem_labels"][idx]
                    resized_image, points_cp[point_cam_id_mask], image_label = image_and_points_cp_and_label_resize(
                        image=ori_image, # (1280, 1920, 3) or (886, 1920, 3)
                        points_cp=points_cp[point_cam_id_mask], 
                        image_label=ori_image_sem_label, 
                        resized_shape=resized_shape_cv,
                    )
                    resized_images.append(resized_image)
                    resized_image_sem_labels.append(image_label)
                else:
                    resized_image, points_cp[point_cam_id_mask], _ = image_and_points_cp_and_label_resize(
                        image=ori_image, 
                        points_cp=points_cp[point_cam_id_mask], 
                        image_label=None, 
                        resized_shape=resized_shape_cv,
                    )
                    resized_images.append(resized_image)


            if mode == "train" and not self.no_augmentation:
                for cam_id, img in zip(cam_names, resized_images):
                    point_cam_id_mask = points_cp[:, 0] == int(cam_id)
                    idx = int(cam_id)-1 
                    
                    if img_has_annotations:
                        img_sem_label = resized_image_sem_labels[idx]
            
                        if self.random_horizon_flip:
                            img, points_cp[point_cam_id_mask, 1], img_sem_label = image_and_points_cp_and_label_random_horizon_flip(
                                image=img, 
                                points_cp=points_cp[point_cam_id_mask, 1],
                                image_label=img_sem_label
                            )

                        if self.random_color_jitter:
                            img = self.random_color_jitter_wrap(img)


                        if self.random_jpeg_compression:
                            img = jpeg_compression(
                                image=img, 
                                quality_noise=self.random_jpeg_compression_cfg["quality_noise"], 
                                probability=self.random_jpeg_compression_cfg["probability"],
                            )

                        if self.random_rescale:
                            img, points_cp[point_cam_id_mask], img_sem_label = self.random_rescale_wrap(
                                image=img, 
                                points_cp=points_cp[point_cam_id_mask], 
                                image_label=img_sem_label
                            )

                        if self.random_crop:
                            img, points_cp[point_cam_id_mask], img_sem_label = self.random_crop_wrap(
                                image=img, 
                                points_cp=points_cp[point_cam_id_mask], 
                                image_label=img_sem_label
                            )


                        resized_images[idx] = img
                        resized_image_sem_labels[idx] = img_sem_label
                    else:
                        assert False

                    

            if self.save_img_for_tta:
                res["images_for_tta"] = deepcopy(resized_images)



            for cam_id, img in zip(cam_names, resized_images):
                idx = int(cam_id)-1 
                img = image_input_transform(
                    img, 
                    mean=cam_attributes[cam_id]["mean"], 
                    std=cam_attributes[cam_id]["std"],
                ).astype(np.float32)
                resized_images[idx] = img


            # NOTE: synchronous shuffle_points
            if self.shuffle_points:
                points_shuffle_idx = res["lidar"]["points_shuffle_idx"]
                assert points_shuffle_idx.shape[0] <= points_cp.shape[0]
                points_cp = points_cp[points_shuffle_idx]



            # num_cam [H, W, 3] -> [num_cam, H, W, 3] -> [num_cam, 3, H, W]
            images = np.stack(resized_images, axis=0).transpose((0,3,1,2))
            res_shape = images.shape[-2:] # [H, W]

            if img_has_annotations:
                # num_cam [H, W] -> [num_cam, H, W]
                images_sem_labels = np.stack(resized_image_sem_labels, axis=0)
                res["images_sem_labels"] = images_sem_labels.astype(np.float32)


            # normalize the camera projection coordinates to [-1, 1], in order to use F.grid_sample later
            # F.grid_sample supports 3D tensor interpolation, so the cam_id can also be normalized to [-1, 1]
            # input: (N, C, D_\text{in}, H_\text{in}, W_\text{in})
            # output: (N, C, D_\text{out}, H_\text{out}, W_\text{out})
            # input: grid (N, D_out, H_out​, W_out​, 3) 3: [d, h, w]
            if len(cam_names) > 1:
                points_cuv_all[:, 0] = (points_cp[:, 0] - 1) / (len(cam_names) - 1) * 2 - 1
            else:
                # 0 is fine for nusc
                points_cuv_all[:, 0] = 0
                # points_cuv_all[:, 0] = -1


            points_cuv_all[:, 1] = points_cp[:, 2] / (res_shape[0] - 1) * 2 -1
            points_cuv_all[:, 2] = points_cp[:, 1] / (res_shape[1] - 1) * 2 -1


            # points_cp_valid: (points.shape[0], 1)
            points_cp_valid = (points_cp[:, 0:1] > 0).astype(points_cuv_all.dtype)  
            # points_cuv_all: (points.shape[0], 4), the inner 4 dims: [valid, normed_camid, normed_h_coord, normed_w_coord] 
            
            res["lidar"]["points_cp"] = points_cp
            res["lidar"]["points_cuv"] = np.concatenate([points_cp_valid, points_cuv_all], axis=1) # normalized camera projection coordinates


            res["images"] = images

        return res, info

