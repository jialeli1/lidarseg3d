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
                            ├── velodyne  <-- point cloud files
                            └── labels    <-- annotations for train set
                        ├── 01   
                        ├── 02   
                        ├── ...
                        ├── 20 
                        ├── 21  
```

The semanticKITTI is relatively simple, and we did not do additional conversions.


### Train & Evaluate in Command Line

Now we only support training and evaluation with gpu. Cpu only mode is not supported. The following template commands are available for all datasets via the specific config.

Use the following command to start a distributed training using 4 GPUs. You can decide how many GPUs to use to train by --nproc_per_node=X. The models and logs will be saved to ```work_dirs/CONFIG_NAME``` 


```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py CONFIG_PATH

# example1: 
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py configs/semantickitti/SDSeg3D/semkitti_transVFE_unetscn3d_batchloss_e10.py
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

