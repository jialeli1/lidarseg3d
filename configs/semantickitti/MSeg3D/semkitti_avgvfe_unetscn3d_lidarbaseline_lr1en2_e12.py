import itertools
import logging
from typing import Sequence

from addict.addict import Dict
# from hrnet_cfg import hrnet_w48
from hrnet_cfg import hrnet_w18
from fcn_cfg import fcn_head


num_class=20
ignore_class=0



# training and testing settings
point_cloud_range=[-75.2, -75.2, -4, 75.2, 75.2, 2]
voxel_size=[0.1, 0.1, 0.15]




# model settings
model = dict(
    type="SegNet",
    pretrained=None,
    reader=dict(
        type="ImprovedMeanVoxelFeatureExtractor",
        num_input_features=4, 
    ),
    backbone=dict(
        type="UNetSCN3D", 
        num_input_features=4+8,  
        ds_factor=8, 
        us_factor=8,
        point_cloud_range=point_cloud_range,  
        voxel_size=voxel_size,  
        model_cfg=dict(
            SCALING_RATIO=2, # channel scaling
        ),
    ),
    point_head=dict(
        type="PointSegBatchlossHead",
        class_agnostic=False, 
        num_class=num_class,
        model_cfg=dict(
            CONV_IN_DIM=32,  
            CONV_CLS_FC=[64],
            CONV_ALIGN_DIM=64,
            OUT_CLS_FC=[64, 64],
            IGNORED_LABEL=0,
        )
    )
)

train_cfg = dict()
test_cfg = dict()






# dataset settings
dataset_type = "SemanticKITTIDataset"
data_root = "data/SemanticKITTI/dataset/sequences"  
nsweeps = 1


train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    npoints=500000,
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.95, 1.05], 
    global_translate_std=0.5,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)

test_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)



voxel_generator = dict(
    range=point_cloud_range,
    voxel_size=voxel_size,
    max_points_in_voxel=5,
    max_voxel_num=[500000, 500000], 
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=False),
    dict(type="SegPreprocess", cfg=train_preprocessor),
    dict(type="SegVoxelization", cfg=voxel_generator),
    dict(type="SegAssignLabel", cfg=dict(voxel_label_enc="compact_value")),
    dict(type="Reformat"),
]
val_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="SegPreprocess", cfg=val_preprocessor),
    dict(type="SegVoxelization", cfg=voxel_generator),
    dict(type="Reformat"),
]
test_pipeline = []

train_anno = None
val_anno = None
test_anno = None

train_seq = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
val_seq = ['08']
test_seq = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

data = dict(
    samples_per_gpu=4,  
    workers_per_gpu=8,  
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        sequences=train_seq,
        nsweeps=nsweeps,
        load_interval=1, 
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        sequences=val_seq,
        nsweeps=nsweeps,
        load_interval=1,
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        test_mode=True,
        ann_file=test_anno,
        sequences=test_seq,
        nsweeps=nsweeps,
        pipeline=test_pipeline,
    ),
)



optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.01, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,  
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)


total_epochs = 12
# total_epochs = 24

device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
workflow = [('train', 1)]

sync_bn_type = "torch"