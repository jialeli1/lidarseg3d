# from .resnet_mmcv import ResNet18
# from .resnet import ResNet18

# from .dla import DLASeg
# from .hrnet import HRNet

from .fcn_head import FCNHead
from .fcn_mseg3d_head import FCNMSeg3DHead



__all__ = [
    "FCNHead",
    "FCNMSeg3DHead",
]
