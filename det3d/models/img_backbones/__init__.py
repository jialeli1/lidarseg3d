# from .resnet_mmcv import ResNet18
# from .resnet import ResNet18

# from .dla import DLASeg
from .hrnet import HRNet


from .resnet_mmcv import ResNet, ResNetV1c, ResNetV1d

__all__ = [
    # "ResNet18", 
    # 'DLASeg', 
    'HRNet',
    'ResNet',
    'ResNetV1c',
    'ResNetV1d',
]
