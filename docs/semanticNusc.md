## Getting Started with lidarseg3d on nuScenes

### Prepare data

#### Download offical data and organise as follows
```
# For nuScenes Dataset         
└── lidarseg3d
    └── data    
        └── SemanticNusc 
            ├── samples       <-- key frames
            ├── sweeps        <-- frames without annotation
            ├── maps          <-- unused
            ├── lidarseg      <-- from offical nuScenes-lidarseg-all-v1.0.tar
            ├── panoptic      <-- from offical nuScenes-panoptic-v1.0-all.tar.gz
            ├── v1.0-trainval <-- metadata
            ├── v1.0-test     <-- metadata
```


#### Create data
Data creation should be under the gpu environment.
```
# nuScenes
python tools/create_data.py semanticnuscenes_data_prep --root_path=data/SemanticNusc --version="v1.0-trainval" --nsweeps=10
```




In the end, the data and info files should be organized as follows

```
# For nuScenes Dataset 
└── lidarseg3d
    └── data    
        └── SemanticNusc 
            ├── samples       <-- key frames
            ├── sweeps        <-- frames without annotation
            ├── maps          <-- unused
            ├── lidarseg      <-- from offical nuScenes-lidarseg-all-v1.0.tar
            ├── panoptic      <-- from offical nuScenes-panoptic-v1.0-all.tar.gz
            ├── v1.0-trainval <-- metadata
            ├── v1.0-test     <-- metadata
            |── infos_train_10sweeps_segdet_withvelo_filter_True.pkl    <-- train annotations
            |── infos_val_10sweeps_segdet_withvelo_filter_True.pkl      <-- val annotations
            |── infos_test_10sweeps_segdet_withvelo_filter_True.pkl     <-- test annotations
```

### Prepare the pretrained image backbones (Optional for Multimodal 3D Semantic Segmentation)
The publicly available pth files are downloaded directly from mmsegmentation. Two downloaded pth files are also provided [here](https://drive.google.com/drive/folders/1x1oZZMstVdQyV3aPR_pe-qU4aAISHxdm?usp=sharing) for quick experiments with HRNet-w18 and HRNet-w48. Please organise your downloaded pth files as follows.
``` 
└── lidarseg3d
    └── data  
    ├── ...  
    └── work_dirs
        └── pretrained_models              <--- shared for different datasets
            |── hrnetv2_w18-00eb2006.pth   <--- HRNet-w18
            └── hrnetv2_w48-d2186c55.pth   <--- HRNet-w48
```




### Train & Evaluation in Command Line
Now we only support training and evaluation with gpu. Cpu only mode is not supported. The following template commands are available for all datasets via the specific config.

Use the following command to start a distributed training using 4 GPUs. You can decide how many GPUs to use to train by --nproc_per_node=X. The models and logs will be saved to ```work_dirs/CONFIG_NAME``` 


```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py CONFIG_PATH

# example1: 
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py configs/semanticnusc/SDSeg3D/semnusc_transvfe_unetscn3d_batchloss_e48.py

# example for MSeg3D
# after code cleanup and optimization, this sample cfg using HRNet-w18 as the image backbone network has achieved 80.12mIoU, trained on 4 GeForce RTX 3090 GPUs.
# you can modify the cfg with larger image backbone HRNet-w48 and 24 epochs 
# for more training time and segmentation performance.
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py configs/semanticnusc/MSeg3D/semnusc_avgvfe_unetscn3d_hrnetw18_lr1en2_e12.py --tcp_port 17045 
```

For distributed testing with 4 gpus

```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth 

# example1: normal evaluation without TTA
python -m torch.distributed.launch --nproc_per_node=4 ./tools/dist_test.py configs/semanticnusc/SDSeg3D/semnusc_transvfe_unetscn3d_batchloss_e48.py --work_dir work_dirs/semnusc_transvfe_unetscn3d_batchloss_e48 --checkpoint work_dirs/semnusc_transvfe_unetscn3d_batchloss_e48/latest.pth 

# example2: evaluation with TTA
python -m torch.distributed.launch --nproc_per_node=1 ./tools/dist_test.py configs/semanticnusc/SDSeg3D/semnusc_transvfe_unetscn3d_batchloss_e48_tta.py --work_dir work_dirs/semnusc_transvfe_unetscn3d_batchloss_e48_tta --checkpoint work_dirs/semnusc_transvfe_unetscn3d_batchloss_e48/latest.pth 
```

For testing with one gpu and see the inference time by "--speed_test"

```bash
python ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth --speed_test 
```
