from torch._C import set_flush_denormal
from det3d import torchie
import numpy as np
import torch

from ..registry import PIPELINES


class DataBundle(object):
    def __init__(self, data):
        self.data = data


@PIPELINES.register_module
class TTAReformat(object):
    def __init__(self, cfg, **kwargs):
        self.tta_flag = cfg.get('tta_flag', False)
        self.num_tta_tranforms = cfg.get('num_tta_tranforms', -1)


    def __call__(self, res, info):
        meta = res["metadata"]
        points = res["lidar"]["points"]
        voxels = res["lidar"]["voxels"]
        all_points = res["lidar"]["all_points"]


        data_bundle = dict(
            metadata=meta,
            points=points,
            voxels=voxels["voxels"],
            shape=voxels["shape"],
            num_points=voxels["num_points"],
            num_voxels=voxels["num_voxels"],
            coordinates=voxels["coordinates"],
            all_points=all_points, 
        )

        if res["mode"] == "train":
            data_bundle.update(res["lidar"]["targets"])
        elif res["mode"] == "val":
            data_bundle.update(dict(metadata=meta, ))

            if self.tta_flag:
                data_bundle_list = [data_bundle]
                assert self.num_tta_tranforms > 1
                for i in range(1, self.num_tta_tranforms):
                    point_key_i = "tta_%s_points" %i
                    voxel_key_i = "tta_%s_voxels" %i

                    tta_points = res["lidar"][point_key_i]
                    tta_voxels = res["lidar"][voxel_key_i] 
                    tta_data_bundle = dict(
                        metadata=meta,
                        points=tta_points,
                        voxels=tta_voxels["voxels"],
                        shape=tta_voxels["shape"],
                        num_points=tta_voxels["num_points"],
                        num_voxels=tta_voxels["num_voxels"],
                        coordinates=tta_voxels["coordinates"],
                    )
                    data_bundle_list.append(tta_data_bundle)

                return data_bundle_list, info


        return data_bundle, info



