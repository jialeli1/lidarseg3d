from .. import builder
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 
import time
import os
import numpy as np

@DETECTORS.register_module
class SegNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        point_head,
        neck=None,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        **kwargs,
    ):
        # set pretrained as none here
        super(SegNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained=None
        )
        # build the point head for segentation task
        self.point_head = builder.build_point_head(point_head)
        
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            load_checkpoint(self, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))


    def extract_feat(self):
        """ 
        not used for SegNet
        """
        assert False
        

    def forward(self, example, return_loss=True, **kwargs):
        """
        example: a dict 
        """
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]
        batch_size = len(num_voxels)
        # ensure that the points just include [bs_idx, x, y, z]
        points = example["points"][:, 0:4]
        

        # construct a batch_dict like pv-rcnn
        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            # coors=coordinates,
            voxel_coords=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
            points=points,
        )

        # VFE
        # input_features = self.reader(data["features"], data["num_voxels"])
        input_features = self.reader(data["features"], data["num_voxels"], data["voxel_coords"])
        data["voxel_features"] = input_features
        
        # backbone
        data = self.backbone(data)

        # prepare labels for training
        if return_loss:
            data["voxel_sem_labels"] = example["voxel_sem_labels"]
            data["point_sem_labels"] = example["point_sem_labels"]

        # point head
        data = self.point_head(batch_dict=data, return_loss=return_loss)


        if return_loss:
            seg_loss_dict = {}
            point_loss, point_loss_dict = self.point_head.get_loss()

            # this item for Optimizer, formating as loss per task
            opt_loss = [point_loss]
            seg_loss_dict["loss"] = opt_loss

            # reformat for text logger
            for k, v in point_loss_dict.items():
                repeat_list = [v for i in range(len(opt_loss))]
                seg_loss_dict[k] = repeat_list

            return seg_loss_dict
        else:
            return self.point_head.predict(example=example, test_cfg=self.test_cfg)
