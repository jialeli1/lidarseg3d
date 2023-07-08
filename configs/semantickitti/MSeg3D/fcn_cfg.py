import itertools
import logging
from typing import Sequence

from addict.addict import Dict

# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True) # mmcv style
norm_cfg = dict(type='BN', requires_grad=True) # det3d style


fcn_head=dict(
    type='FCNHead',
    in_channels=[18, 36, 72, 144],
    in_index=(0, 1, 2, 3),
    channels=sum([18, 36, 72, 144]),
    input_transform='resize_concat',
    kernel_size=1,
    num_convs=1,
    concat_input=False,
    dropout_ratio=-1,
    num_classes=19,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', 
        use_sigmoid=False, 
        loss_weight=1.0
    )
)