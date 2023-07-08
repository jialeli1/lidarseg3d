from re import S
from turtle import distance

import nuscenes
import det3d
import os
import sys
import pickle
import json
import random
import operator
import numpy as np
import argparse

from pathlib import Path
from copy import deepcopy
import pickle

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.lidarseg.validate_submission import validate_submission
except:
    print("nuScenes devkit not found!")


from det3d.core.utils.seg_utils import per_class_iou_func, fast_hist_crop_func
from det3d.datasets.nuscenes.semnuscenes_common import learning_map, labels_16, labels

from det3d.datasets.registry import DATASETS
from det3d.datasets.pipelines import Compose
from torch.utils.data import Dataset


def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 


@DATASETS.register_module
class SemanticNuscDataset(Dataset):
    NumPointFeatures = 5  # x, y, z, intensity, ring_index
    CLASSES = 17

    def __init__(
        self,
        info_path,
        root_path,
        cfg=None,
        pipeline=None,
        class_names=None,
        cam_names=None,
        cam_chan=None,
        cam_attributes=None,
        img_resized_shape=None,
        test_mode=False,
        sample=False,
        nsweeps=1,
        load_interval=1,
        version="v1.0-trainval",
        **kwargs,
        ):
        super().__init__()
        self.test_mode = test_mode
        self._root_path = root_path
        self._class_names = class_names
        self._use_img = cam_names is not None

        if self._use_img: 
            self._cam_names = cam_names
            self._cam_chan = cam_chan
            self.img_resized_shape = img_resized_shape
            
            _cam_attributes = {}
            for cam_id, cam_attribute in cam_attributes.items():
                mean_np = np.array(cam_attribute["mean"], dtype=np.float32).reshape(1,1,3)
                std_np = np.array(cam_attribute["std"], dtype=np.float32).reshape(1,1,3)
                _cam_attributes[cam_id] = {"mean": mean_np, "std": std_np}
            self._cam_attributes = _cam_attributes


        self.learning_map = learning_map
        self.load_interval = load_interval 
        self.sample = sample
        self.nsweeps = nsweeps
        print("Using {} sweeps".format(nsweeps))


        # data/SemanticNusc/infos_train_01sweeps_segdet_filter_zero_gt.pkl
        self._info_path = info_path 
        
        self._class_names = class_names
        self._num_point_features = SemanticNuscDataset.NumPointFeatures if nsweeps == 1 else SemanticNuscDataset.NumPointFeatures+1

        self._set_group_flag()

        if pipeline is None:
            self.pipeline = None
        else:
            self.pipeline = Compose(pipeline)


        if self.test_mode:
            # init self.nusc
            self.nusc = NuScenes(version=version, dataroot=str(self._root_path), verbose=True)


    def reset(self):
        assert False 

    def load_infos(self, info_path):
        print("==> Using info: {}".format(info_path))

        with open(info_path, "rb") as f:    
            _semantic_nusc_infos_all = pickle.load(f)  

        # downsample dataset if self.load_interval > 1
        self._semantic_nusc_infos = _semantic_nusc_infos_all[::self.load_interval]
        print("Using seg annotated {} frames out of {} frames".format(len(self._semantic_nusc_infos), len(_semantic_nusc_infos_all)))


    def __len__(self):
        if not hasattr(self, "_semantic_nusc_infos"):
            self.load_infos(self._info_path)

        return len(self._semantic_nusc_infos)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.ones(len(self), dtype=np.uint8)
        # self.flag = np.zeros(len(self), dtype=np.uint8)
        # for i in range(len(self)):
        #     img_info = self.img_infos[i]
        #     if img_info['width'] / img_info['height'] > 1:
        #         self.flag[i] = 1



    def get_sensor_data(self, idx):
        info = self._semantic_nusc_infos[idx]  

        info["dim"]={
                "points": self._num_point_features,
                "sem_labels": 1, 
                }

        if self._use_img:
            res_cam = {
                "names": self._cam_names, # NOTE: 必须要设置成一个有序的list!
                "chan": self._cam_chan, # NOTE: 必须要设置成一个有序的list!
                "attributes": self._cam_attributes,
                "resized_shape": self.img_resized_shape,
                "annotations": None
            }
        else:
            res_cam = {}

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "nsweeps": self.nsweeps,
                # "ground_plane": -gp[-1] if with_gp else None,
                "annotations": None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
                "num_points_of_top_lidar": None,
            },
            "calib": None,
            "cam": res_cam,
            "mode": "val" if self.test_mode else "train",
            "learning_map": self.learning_map,
        }


        data, _ = self.pipeline(res, info) 

        return data


    def __getitem__(self, idx):
        return self.get_sensor_data(idx)


    def get_anno_for_eval(self, token):

        lidar_sd_token = self.nusc.get('sample',token)['data']['LIDAR_TOP']
        lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                self.nusc.get('lidarseg', lidar_sd_token)['filename'])

        point_labels = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape(-1)
        point_labels = np.vectorize(self.learning_map.__getitem__)(point_labels).astype(np.float32)
        # points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        anno_dict = {
            "point_sem_labels": point_labels,
            # "points": points,
        }
        return anno_dict


    def evaluation(self, detections, output_dir=None, testset=False, **kwargs):
        """
        detections(a dict)
            token_i(a dict):
                "pred_point_sem_labels": cpu tensor
                "point_sem_labels": cpu tensor or not exist
                "voxel_sem_labels": cpu tensor or not exist
        res: evaluation metric dict
        """

        if not testset:
            # compute mIoU like Cylinder3D
            SemNuScene_label_name = labels_16 
            unique_label = np.asarray(sorted(list(SemNuScene_label_name.keys())))[1:] - 1
            unique_label_str = [SemNuScene_label_name[x] for x in unique_label + 1]

            hist_list = []
            for token, pred_dict in detections.items():
                anno_dict = self.get_anno_for_eval(token)
                assert "point_sem_labels" in anno_dict
                
                pred_point_sem_labels = pred_dict["pred_point_sem_labels"].numpy()
                gt_point_sem_labels = anno_dict["point_sem_labels"]
                
                assert pred_point_sem_labels.shape[0] == gt_point_sem_labels.shape[0], "pred_point_sem_labels.shape: {}, gt_point_sem_labels.shape: {}".format(pred_point_sem_labels.shape, gt_point_sem_labels.shape)


                hist_list.append(fast_hist_crop_func(
                    output=pred_point_sem_labels, 
                    target=gt_point_sem_labels, 
                    unique_label=unique_label))
            
            # compute iou
            per_class_ious = per_class_iou_func(sum(hist_list))
            miou = np.nanmean(per_class_ious)
            

            detail = {}
            result = {"mIoU": miou*100}
            for class_name, class_iou in zip(unique_label_str, per_class_ious):
                # print('==> %s : %.2f%%' % (class_name, class_iou * 100))
                result[class_name] = class_iou * 100

            res = {
                "results": result,
                "detail": detail
                }
        else:  
            # test set
            # find the saved file at work_dirs/cfg/...

            res = None
            output_dir_ = output_dir + "/results_folder/lidarseg/test"
            json_dir = output_dir + "/results_folder/test"
            results_dir = output_dir + "/results_folder"
            
            meta_= {
                "meta":{
                    "use_camera":   self._use_img,        
                    "use_lidar":    True,         
                    "use_radar":    False,          
                    "use_map":      False,          
                    "use_external": False,
                    } 
                }        

            file_name = json_dir + '/submission.json' 
            if not os.path.exists(os.path.dirname(file_name)):
                os.makedirs(os.path.dirname(file_name))
            with open(file_name,'w') as file_object:
                json.dump(meta_,file_object)
                
            # save as *.bin for 6008 testing files.
            for token, pred_dict in detections.items():         
                # lidar_sd_token = self.nusc.get('sample',token)['data']['LIDAR_TOP']
                sample = self.nusc.get('sample', token)
                lidar_sd_token = sample['data']['LIDAR_TOP']

                pred_point_sem_labels = pred_dict["pred_point_sem_labels"].numpy()

                # output_dir = 'output/' + output_dir + "/results_folder/lidarseg/test"
                bin_file_path = output_dir_ + "/" + lidar_sd_token + '_lidarseg.bin'
                if not os.path.exists(os.path.dirname(bin_file_path)):
                    os.makedirs(os.path.dirname(bin_file_path))
                np.array(pred_point_sem_labels).astype(np.uint8).tofile(bin_file_path)

            # check and zip
            validate_submission(nusc = self.nusc,   
                                results_folder = results_dir,
                                eval_set= "test",
                                verbose = True,
                                zip_out = output_dir)


        return res, None

