## Getting Started with lidarseg3d on Waymo

### Prerequisite 
- Follow [INSTALL.md](INSTALL.md) to install all required libraries. 
- Tensorflow 
- Waymo-open-dataset devkit

```bash
conda activate lidarseg

cd lidarseg3d
pip install -r requirements.txt
# or 
pip install waymo-open-dataset-tf-2-6-0==1.4.3
```


### Prepare data

#### Download offical data and organise as follows
```
# For Waymo Dataset         
└── lidarseg3d
    └── data    
        └── SemanticWaymo 
            ├── tfrecord_training       
            ├── tfrecord_validation     
            ├── tfrecord_testing       
```


#### Create data
Convert the tfrecord data to pickle files.

```bash
# train set 
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path 'data/SemanticWaymo/tfrecord_training/*.tfrecord'  --root_path 'data/SemanticWaymo/train/'

# validation set 
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path 'data/SemanticWaymo/tfrecord_validation/*.tfrecord'  --root_path 'data/SemanticWaymo/val/'

# testing set 
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path 'data/SemanticWaymo/tfrecord_testing/*.tfrecord'  --root_path 'data/SemanticWaymo/test/'
```


#### Create info files

```bash
# One Sweep Infos
# train set 
python tools/create_data.py semanticwaymo_data_prep --root_path=data/SemanticWaymo --split train --nsweeps=1

# validation set 
python tools/create_data.py semanticwaymo_data_prep --root_path=data/SemanticWaymo --split val --nsweeps=1

# testing set 
python tools/create_data.py semanticwaymo_data_prep --root_path=data/SemanticWaymo --split test --nsweeps=1
```

In the end, the data and info files should be organized as follows

```
└── lidarseg3d
    └── data    
        └── SemanticWaymo 
            ├── tfrecord_training       
            ├── tfrecord_validation     
            ├── tfrecord_testing   
            ├── train   <-- all training frames and annotations 
            ├── val     <-- all validation frames and annotations 
            ├── test    <-- all testing frames and annotations 
            ├── infos_train_01sweeps_segdet_filter_zero_gt.pkl  <-- will be specified in cfg 
            ├── infos_val_01sweeps_segdet_filter_zero_gt.pkl    <-- will be specified in cfg
            ├── infos_test_01sweeps_segdet_filter_zero_gt.pkl   <-- will be specified in cfg
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


# example for mseg3d:
# you can modify the cfg with larger image backbone HRNet-w48 and 24 epochs for more training time and segmentation performance
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py configs/semanticwaymo/MSeg3D/semwaymo_avgvfe_unetscn3d_hrnetw18_lr1en2_e12.py --tcp_port 16045 
```

For distributed testing with 4 gpus

```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth 
```

For testing with one gpu and see the inference time by "--speed_test"

```bash
python ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth --speed_test 
```
