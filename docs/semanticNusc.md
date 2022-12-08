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



### Train & Evaluate in Command Line
Now we only support training and evaluation with gpu. Cpu only mode is not supported. The following template commands are available for all datasets via the specific config.

Use the following command to start a distributed training using 4 GPUs. You can decide how many GPUs to use to train by --nproc_per_node=X. The models and logs will be saved to ```work_dirs/CONFIG_NAME``` 


```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py CONFIG_PATH

# example1: 
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py configs/semanticnusc/SDSeg3D/semnusc_transvfe_unetscn3d_batchloss_e48.py
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
