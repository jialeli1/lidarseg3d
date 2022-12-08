from det3d import torchie

from ..registry import PIPELINES
from .compose import Compose
from det3d.core.sampler import segpreprocess as segprep
import numpy as np
import copy

@PIPELINES.register_module
class SegDoubleFlip(object):
    def __init__(self):
        pass

    def __call__(self, res, info):
        # y flip
        points = res["lidar"]["all_points"].copy()
        points[:, 1] = -points[:, 1]

        res["lidar"]['yflip_points'] = points

        # x flip
        points = res["lidar"]["all_points"].copy()
        points[:, 0] = -points[:, 0]

        res["lidar"]['xflip_points'] = points

        # x y flip
        points = res["lidar"]["all_points"].copy()
        points[:, 0] = -points[:, 0]
        points[:, 1] = -points[:, 1]

        res["lidar"]["double_flip_points"] = points  

        return res, info 



@PIPELINES.register_module
class SegCompoundAug(object):
    def __init__(self, cfg):
        """
        Generate input variants by using multiple Compound Transformations with different random parameters.
        """
        self.global_rotation_noise = cfg.get("global_rotation_noise", [-0.78539816, 0.78539816])
        self.global_scaling_noise = cfg.get("global_scaling_noise", [0.95, 1.05])
        self.global_translate_std = cfg.get("global_translate_std", 0.5)
        self.global_flip_prob = cfg.get("global_flip_prob", 1.0)

        self.num_tta_tranforms = cfg["num_tta_tranforms"]


    def compound_trans(self, points):
        # points_global_flip() includes the rotation and flip
        points = segprep.points_global_flip(
            points=points, 
            probability=self.global_flip_prob
        )

        # translate
        points = segprep.points_global_translate_(
            points, 
            noise_translate_std=self.global_translate_std
        )

        # scaling
        points = segprep.points_global_scaling_v2(
            points, 
            *self.global_scaling_noise,
        )
        
        return points


    def __call__(self, res, info):
        """
        1 identical point cloud and (num_tta_tranforms - 1) transformed point clouds.
        """

        assert self.num_tta_tranforms > 1
        for i in range(1, self.num_tta_tranforms):
            point_key_i = "tta_%s_points" %i

            points = self.compound_trans(
                res["lidar"]["all_points"].copy()
            )
            res["lidar"][point_key_i] = points


        return res, info





