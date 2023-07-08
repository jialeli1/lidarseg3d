import pickle
from matplotlib import use
import numpy as np
import random

import os

from det3d.datasets.custom import PointCloudDataset

from det3d.datasets.registry import DATASETS
from det3d.datasets.pipelines import Compose
from torch.utils.data import Dataset

from det3d.core.utils.seg_utils import per_class_iou_func, fast_hist_crop_func
from det3d.datasets.waymo.semanticwaymo_common import semantic_labels

def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 

@DATASETS.register_module
class SemanticWaymoDataset(Dataset):
    NumPointFeatures = 5  # x, y, z, intensity, elongation
    CLASSES = 23 # for semanticwaymo
    # The 23 classes includes: Car, Truck, Bus, Motorcyclist, Bicyclist, Pedestrian, Sign, Traffic Light, Pole, Construction Cone, Bicycle, Motorcycle, Building, Vegetation, Tree Trunk, Curb, Road, Lane Marker, Walkable, Sidewalk, Other Ground, Other Vehicle, Undefined
    def __init__(
        self,
        info_path,
        root_path,
        cfg=None,
        pipeline=None,
        class_names=None,
        cam_names=None,
        cam_attributes=None,
        img_resized_shape=None,
        test_mode=False,
        sample=False,
        nsweeps=1,
        load_interval=1,
        **kwargs,
        ):
        super().__init__()
        self.test_mode = test_mode
        self._root_path = root_path
        self._class_names = class_names
        self._use_img = cam_names is not None
        
        if self._use_img:
            self._cam_names = cam_names
            self.img_resized_shape = img_resized_shape
            
            _cam_attributes = {}
            for cam_id, cam_attribute in cam_attributes.items():
                mean_np = np.array(cam_attribute["mean"], dtype=np.float32).reshape(1,1,3)
                std_np = np.array(cam_attribute["std"], dtype=np.float32).reshape(1,1,3)
                _cam_attributes[cam_id] = {"mean": mean_np, "std": std_np}
            self._cam_attributes = _cam_attributes


        self.load_interval = load_interval 
        self.sample = sample
        self.nsweeps = nsweeps
        print("Using {} sweeps".format(nsweeps))

        # data/SemanticWaymo/infos_train_01sweeps_segdet_filter_zero_gt.pkl
        self._info_path = info_path 
        self._class_names = class_names
        self._num_point_features = SemanticWaymoDataset.NumPointFeatures if nsweeps == 1 else SemanticWaymoDataset.NumPointFeatures+1

        self._set_group_flag()

        if pipeline is None:
            self.pipeline = None
        else:
            for pip_cfg in pipeline:
                if (pip_cfg["type"] == "SegPreprocess") and (not self.test_mode):
                    # 在train mode下进行instance—aug   
                    if pip_cfg["cfg"].get('instance_augmentation', False):
                        new_instance_augmentation_cfg = {
                            "instance_pkl_path": self.instance_pkl_path,
                            "thing_list": self.thing_list,
                            "class_weight": self.cls_loss_weight,  
                        }
                        pip_cfg["cfg"]["instance_augmentation_cfg"].update(new_instance_augmentation_cfg)
                if (pip_cfg["type"] == "SegGetCentroid"):
                    pip_cfg["cfg"].update({
                        "thing_list": self.thing_list,
                    })

            self.pipeline = Compose(pipeline)


    def reset(self):
        assert False 


    def load_infos(self, info_path):
        
        with open(info_path, "rb") as f:
            _waymo_infos_all = pickle.load(f)

        semantic_waymo_infos = []
        for info in _waymo_infos_all:
            if info["seg_annotated"]:
                semantic_waymo_infos.append(info)

        self._semantic_waymo_infos = semantic_waymo_infos[::self.load_interval]

        print("Using seg annotated {} frames out of {} frames".format(len(self._semantic_waymo_infos), len(_waymo_infos_all)))


    def __len__(self):

        if not hasattr(self, "_semantic_waymo_infos"):
            self.load_infos(self._info_path)

        return len(self._semantic_waymo_infos)

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
        """
        """

        info = self._semantic_waymo_infos[idx]  

        info["dim"]={
                "points": self._num_point_features,
                "sem_labels": 1, 
                }

        if self._use_img:
            res_cam = {
                "names": self._cam_names,
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
                "seg_annotated": info["seg_annotated"],
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
        }



        data, _ = self.pipeline(res, info) 

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)


    def get_anno_for_eval(self, token, split='val'):

        anno_path = os.path.join(self._root_path, split, 'annos', token)
        anno_info = get_obj(anno_path)
        points_seglabel = anno_info["seg_labels"]["points_seglabel"]
        
        assert points_seglabel.shape[0] > 0, "Waymo did not annotated {} frame!".format(token)

        # [ins, sem]
        anno_dict = {
            "point_sem_labels": points_seglabel[:, 1],
            # "points": points,
        }
        return anno_dict



    def evaluation(self, detections, output_dir=None, testset=False):
        if not testset:
            # evaluate mIoU like Cylinder3D
            SemWaymo_label_name = semantic_labels
            unique_label = np.asarray(sorted(list(SemWaymo_label_name.keys())))[1:] - 1
            unique_label_str = [SemWaymo_label_name[x] for x in unique_label + 1]

            hist_list = []
            for token, pred_dict in detections.items():
                anno_dict = self.get_anno_for_eval(token)
                assert "point_sem_labels" in anno_dict
                
                pred_point_sem_labels = pred_dict["pred_point_sem_labels"].numpy()
                gt_point_sem_labels = anno_dict["point_sem_labels"]
                
                
                # pred_point_sem_labels includes ri1 and ri2
                # gt_point_sem_labels includes ri1
                if pred_point_sem_labels.shape[0] > gt_point_sem_labels.shape[0]:
                    pred_point_sem_labels = pred_point_sem_labels[:gt_point_sem_labels.shape[0],]


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
            res = None
            # test
            print('Generate predictions for test split')

            from .semanticwaymo_common import _create_pd_segmentation, reorganize_info

            infos = self._semantic_waymo_infos
            infos = reorganize_info(infos)

            _create_pd_segmentation(detections, infos, output_dir, testset)


        return res, None



