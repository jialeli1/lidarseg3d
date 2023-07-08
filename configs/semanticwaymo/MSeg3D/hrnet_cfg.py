import itertools
import logging
from typing import Sequence

from addict.addict import Dict


# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True) # mmcv style
norm_cfg = dict(type='BN', requires_grad=True) # det3d style


hrnet_w18=dict(
    type='HRNet',
    pretrained='./work_dirs/pretrained_models/hrnetv2_w18-00eb2006.pth',
    norm_cfg=norm_cfg,
    norm_eval=False,
    extra=dict(
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block='BOTTLENECK',
            num_blocks=(4, ),
            num_channels=(64, )),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block='BASIC',
            num_blocks=(4, 4),
            num_channels=(18, 36)),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(18, 36, 72)),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block='BASIC',
            num_blocks=(4, 4, 4, 4),
            num_channels=(18, 36, 72, 144)),
        )
    )


hrnet_w48=dict(
    type='HRNet',
    # pretrained='open-mmlab://msra/hrnetv2_w48', # download from internet
    pretrained='./work_dirs/pretrained_models/hrnetv2_w48-d2186c55.pth', # file system

    norm_cfg=norm_cfg,
    norm_eval=False,
    extra=dict(
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block='BOTTLENECK',
            num_blocks=(4, ),
            num_channels=(64, )
            ),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block='BASIC',
            num_blocks=(4, 4),
            # num_channels=(18, 36),    # w18
            num_channels=(48, 96),      # w48
            ),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='BASIC',
            num_blocks=(4, 4, 4),
            # num_channels=(18, 36, 72),    # w18
            num_channels=(48, 96, 192),     # w48
            ),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block='BASIC',
            num_blocks=(4, 4, 4, 4),
            # num_channels=(18, 36, 72, 144)    # w18
            num_channels=(48, 96, 192, 384)     # w48
            ),
        )
    )