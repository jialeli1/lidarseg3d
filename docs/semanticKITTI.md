## Getting Started with lidarseg3d on semanticKITTI

### Prepare data

#### Download offical data and organise as follows
```
# For semanticKITTI Dataset         
└── lidarseg3d
       └── data    
           └── SemanticKITTI
                └── dataset       
                    └── sequences     
                        ├── 00        
                            ├── calib.txt 
                            ├── poses.txt
                            ├── times.txt
                            ├── image_2   <-- front camera image files
                            ├── velodyne  <-- point cloud files
                            └── labels    <-- annotations for train set
                        ├── 01   
                        ├── 02   
                        ├── ...
                        ├── 20 
                        ├── 21  
```

The semanticKITTI is relatively simple, and we did not do additional conversions.



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



### Train & Evaluate in Command Line

Now we only support training and evaluation with gpu. Cpu only mode is not supported. The following template commands are available for all datasets via the specific config.

Use the following command to start a distributed training using 4 GPUs. You can decide how many GPUs to use to train by --nproc_per_node=X. The models and logs will be saved to ```work_dirs/CONFIG_NAME``` 


```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py CONFIG_PATH

# example1: 
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py configs/semantickitti/SDSeg3D/semkitti_transVFE_unetscn3d_batchloss_e10.py


# example for mseg3d:
# you can modify the cfg with larger image backbone HRNet-w48 and 24 epochs for more training time and segmentation performance
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py configs/semantickitti/MSeg3D/semkitti_avgvfe_unetscn3d_hrnetw18_lr1en2_e12.py
```

For distributed testing with 4 gpus

```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth 

# example1: normal evaluation without TTA
python -m torch.distributed.launch --nproc_per_node=4 ./tools/dist_test.py configs/semantickitti/SDSeg3D/semkitti_transVFE_unetscn3d_batchloss_e10.py --work_dir work_dirs/semkitti_transVFE_unetscn3d_batchloss_e10 --checkpoint work_dirs/semkitti_transVFE_unetscn3d_batchloss_e10/latest.pth 

# example2: evaluation with TTA
python -m torch.distributed.launch --nproc_per_node=1 ./tools/dist_test.py configs/semantickitti/SDSeg3D/semkitti_transVFE_unetscn3d_batchloss_e10_tta.py --work_dir work_dirs/semkitti_transVFE_unetscn3d_batchloss_e10_tta --checkpoint work_dirs/semkitti_transVFE_unetscn3d_batchloss_e10/latest.pth 

```

For testing with one gpu and see the inference time by "--speed_test"

```bash
python ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth --speed_test 
```

