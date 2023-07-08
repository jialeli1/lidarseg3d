import det3d
import os
import sys
import pickle
import json
import random
import operator
import numpy as np
import argparse
import torch
import os.path as osp

from functools import reduce, total_ordering
from pathlib import Path
from copy import deepcopy
import pickle
import errno

from numpy.lib import info
from det3d.core.utils.seg_utils import per_class_iou_func, fast_hist_crop_func
from det3d.datasets.semantickitti.semkitti_common import learning_map, learning_map_inv, labels, thing_class, content

from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.registry import DATASETS
from det3d.datasets.pipelines import Compose

from torch.utils.data import Dataset


def get_SemKITTI_label_name(learning_map_dict, labels_dict):
    SemKITTI_label_name = dict()
    for i in sorted(list(learning_map_dict.keys()))[::-1]:
        SemKITTI_label_name[learning_map_dict[i]] = labels_dict[i]

    return SemKITTI_label_name


@DATASETS.register_module
class SemanticKITTIDataset(Dataset):
    NumPointFeatures = 4  # x, y, z, intensity
    CLASSES = 20 # for KITTI

    def __init__(
        self,
        info_path,
        root_path,
        sequences,
        nsweeps=1, 
        load_interval=1, 
        cfg=None,
        pipeline=None,
        class_names=None,
        cam_names=None,
        cam_attributes=None,
        img_resized_shape=None,
        test_mode=False,
        version="v0.0",
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



        self._num_point_features = SemanticKITTIDataset.NumPointFeatures

        self.nsweeps = nsweeps
        assert self.nsweeps > 0, "A least input one sweep please!"

        # assert load_interval == 1 
        self.load_interval = load_interval

        self.version = version

        self.seqs = sequences
        files = []
        frame_names = []
        for seq in sequences:
            # _root_path should be "data/SemanticKITTI/dataset/sequences"
            frame_idx_in_seq = sorted(
                os.listdir(os.path.join(self._root_path, seq, "velodyne"))
            )
            
            frame_names_in_seq = [
                os.path.join(seq, "velodyne", x) for x in frame_idx_in_seq
            ]
            frame_names.extend(frame_names_in_seq)
            
            files_in_seq = [ 
                os.path.join(self._root_path, seq, "velodyne", x) for x in frame_idx_in_seq
            ]
            files.extend(files_in_seq)
        
        # downsample dataset
        if load_interval > 1:
            files = files[::load_interval]
            frame_names = frame_names[::load_interval]
        

        self.files = files
        self.frame_names = frame_names

        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.labels = labels
        # for panoptic segmentation
        self.thing_list = [cl for cl, ignored in thing_class.items() if ignored]        


        self._set_group_flag()

        if pipeline is None:
            self.pipeline = None
        else:
            self.pipeline = Compose(pipeline)

    def reset(self):
        assert False

    def load_infos(self, idx):
        """
        create info for SemanticKitti
        """
        simple_info = {
            "path": self.files[idx],
            "token": self.frame_names[idx],
            "learning_map": self.learning_map, 
            "learning_map_inv": self.learning_map_inv,
            "dim": {
                "points": self._num_point_features,
                "sem_labels": 1,
                "inst_labels": 1,
            }
        }

        return simple_info


    def __len__(self):
        return len(self.files)


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


    def ground_truth_annotations(self):
        assert False

    def get_sensor_data(self, idx):

        info = self.load_infos(idx)

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
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
            },
            "calib": None,
            "cam": res_cam,
            "mode": "val" if self.test_mode else "train",
            "painted": False, 
        }
        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)


    def get_anno_for_eval(self, token):

        path = os.path.join(self._root_path, token)

        label_path = path.replace("velodyne", "labels").replace(".bin", ".label")
        all_labels = np.fromfile(label_path, dtype=np.int32).reshape(-1)
        # semantic labels
        sem_labels = all_labels & 0xFFFF
        # instance labels
        inst_labels = (all_labels >> 16).astype(np.int32) 


        # label mapping 
        sem_labels = (np.vectorize(learning_map.__getitem__)(sem_labels)).astype(np.uint8) 
        

        anno_dict = {
            "point_sem_labels": sem_labels,
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
            SemKITTI_label_name = get_SemKITTI_label_name(self.learning_map, self.labels)
            unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
            unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

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
            # test
            res = None
            print('Generate predictions for test split')
            with torch.no_grad():
                for token, pred_dict in detections.items():    
                    pred_point_sem_labels = pred_dict["pred_point_sem_labels"].numpy()
                    pred_point_sem_labels = np.expand_dims(pred_point_sem_labels,axis=1)
 
                    # 'out/SemKITTI_test/sequences/11/predictions/000001.label'
                    output_path = osp.join(output_dir, 'out/SemKITTI_test')
                    save_dir = output_path + '/sequences/' + token.replace('velodyne','predictions')[:-3]+'label'
                    if not os.path.exists(os.path.dirname(save_dir)):
                        try:
                            os.makedirs(os.path.dirname(save_dir))
                        except OSError as exc:
                            if exc.errno != errno.EEXIST:
                                raise
                    pred_point_sem_labels = pred_point_sem_labels.astype(np.uint32)
                    pred_point_sem_labels.tofile(save_dir)
                        
            print('Predicted test labels are saved in %s.' % output_path)
            print('Need to be shifted to original label format before submitting to the Competition website.')
            print('Remapping script can be found in semantic-kitti-api.')

        return res, None



    def save_instance(self, out_dir='data/SemanticKITTI/dataset', min_points=10):
        """
        instance data preparation from Panoptic-PolarNet.
        """
        instance_dict={label:[] for label in self.thing_list}
        for data_path in self.files:
            print("==> Processing instance for: " + data_path)
            # get x,y,z,ref,semantic label and instance label
            raw_data = np.fromfile(data_path, dtype=np.float32).reshape((-1, 4))
            annotated_data = np.fromfile(data_path.replace('velodyne','labels')[:-3]+'label', dtype=np.uint32).reshape((-1,1))
            sem_data = annotated_data & 0xFFFF #delete high 16 digits binary
            sem_data = np.vectorize(self.learning_map.__getitem__)(sem_data)
            inst_data = annotated_data

            # instance mask
            mask = np.zeros_like(sem_data,dtype=bool)
            for label in self.thing_list:
                mask[sem_data == label] = True

            # create unqiue instance list
            inst_label = inst_data[mask].squeeze()
            unique_label = np.unique(inst_label)
            num_inst = len(unique_label)

            inst_count = 0
            for inst in unique_label:
                # get instance index
                index = np.where(inst_data == inst)[0]
                # get semantic label
                class_label = sem_data[index[0]]
                # skip small instance
                if index.size < min_points: continue
                # save
                _,dir2 = data_path.split('/sequences/',1)
                # new_save_dir = out_dir + '/sequences/' + dir2.replace('velodyne','instance')[:-4]+'_'+str(inst_count)+'.bin'
                # NOTE change this path to avoid mix the raw data.
                new_save_dir = out_dir + '/instances_in_sequences/' + dir2.replace('velodyne','instance')[:-4]+'_'+str(inst_count)+'.bin'
                if not os.path.exists(os.path.dirname(new_save_dir)):
                    try:
                        os.makedirs(os.path.dirname(new_save_dir))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                inst_fea = raw_data[index]
                # print('==> tofile path: %s' %(new_save_dir))
                inst_fea.tofile(new_save_dir)
                instance_dict[int(class_label)].append(new_save_dir)
                inst_count+=1

        with open(out_dir+'/instance_path.pkl', 'wb') as f:
            pickle.dump(instance_dict, f)
        print('==> instance_path.pkl saved at: %s' %(out_dir+'/instance_path.pkl'))




if __name__ == '__main__':
    # instance preprocessing
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('-d', '--data_path', default='data')
    # parser.add_argument('-o', '--out_path', default='data')
    # args = parser.parse_args()
    
    data_root = "data/SemanticKITTI/dataset/sequences" #"data/nuScenes"
    out_path = 'data/SemanticKITTI/dataset'
    instance_pkl = "data/SemanticKITTI/dataset/instance_path.pkl"
    train_seq = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']


    train_pt_dataset = SemanticKITTIDataset(
        info_path=None,
        root_path=data_root,
        sequences=train_seq,
        test_mode=False,
    )
    train_pt_dataset.save_instance(out_path)

    print('Instance preprocessing finished.')