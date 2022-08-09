# LiDARSeg3D


A generic repository for LiDAR 3D semantic segmentation in autonomous driving scenarios. Also the official implementation of our ECCV 2022 paper: Self-Distillation for Robust LiDAR Semantic Segmentation in Autonomous Driving (SDSeg3D).



## News

<!-- - [2022-07-14] Initial release for the implementation of SDSeg3D.   -->


- [2022-07-04] Our LiDAR-only method SDSeg3D (Self-Distillation for Robust LiDAR Semantic Segmentation in Autonomous Driving) is accepted as a poster paper at ECCV 2022. [Paper](add_url).


<!-- ## Contact
Any questions or suggestions are welcome! 

Jiale Li [jialeli@zju.edu.cn](mailto:jialeli@zju.edu.cn) (ZJU), and
Hang Dai [hang.dai.cs@gmail.com](mailto:hang.dai.cs@gmail.com) (MBZUAI) -->


## Highlights

- **Simple:** Modules and pipelines can be instantiated via cfg files like [mmsegmentation](add_url), but more easily applicable to LiDAR 3D point clouds for voxelization, sparse convolution, devoxelization, etc. 

- **Extensible**: Simple replacement and integration for any network components in your novel algorithms. Smooth compatibility for 3D object detector such as [CenterPoint](https://github.com/tianweiy/CenterPoint), since we try our best to preserve the features inherited from [CenterPoint](https://github.com/tianweiy/CenterPoint). 

- **Fast and Accurate**: Accelerated by 3D sparse convolution with top performance achived on SemantiKITTI, nuScenes, and Waymo benchmarks. 




# Methods
## SDSeg3D
> [**Self-Distillation for Robust LiDAR Semantic Segmentation in Autonomous Driving**](add_url)            
> Jiale Li, Hang Dai, and Yong Ding        
 

### Abstract
We propose a new and effective self-distillation framework with our new Test-Time Augmentation (TTA) and Transformer based Voxel Feature Encoder (TransVFE) for robust LiDAR semantic segmentation in autonomous driving, where the robustness is mission-critical but usually neglected. The proposed framework enables the knowledge to be distilled from a teacher model instance to a student model instance, while the two model instances are with the same network architecture for jointly learning and evolving. This requires a strong teacher model to evolve in training. Our TTA strategy effectively reduces the uncertainty in the inference stage of the teacher model. Thus, we propose to equip the teacher model with TTA for providing privileged guidance while the student continuously updates the teacher with better network parameters learned by itself. To further enhance the teacher model, we propose a TransVFE to improve the point cloud encoding by modeling and preserving the local relationship among the points inside each voxel via multi-head attention. The proposed modules are generally designed to be instantiated with different backbones. Evaluations on SemanticKITTI and nuScenes datasets show that our method achieves state-of-the-art performance. 


### Citation
    @inproceedings{sdseg3d_eccv2022,
    author    = {Jiale Li and
                Hang Dai and
                Yong Ding},
    title     = {Self-Distillation for Robust {LiDAR} Semantic Segmentation in Autonomous Driving},
    booktitle = {ECCV},
    pages     = { },
    year      = {2022},
    }


## Use LiDARSeg3D


### Installation
Please stay tuned until the code is cleaned up.
Follow [INSTALL](docs/INSTALL.md) to set up libraries needed for distributed training and sparse convolution.

### Benchmark Evaluation and Training 

Please refer to [GETTING_START](docs/GETTING_START.md) to prepare the data. Then follow the instruction there to reproduce our segmentation results. All segmentation configurations are included in [configs](configs).



### ToDo List
- [ ] Code cleanup and release for SDSeg3D ASAP
- [ ] Support multiple modalities for LiDAR and multi-cameras 
- [ ] Support detection with detail instruction




# Acknowlegement
This project is mainly constructed on [CenterPoint](https://github.com/tianweiy/CenterPoint) as well as multiple great opensourced codebases. We list some notable examples below. 

* [CenterPoint](https://github.com/tianweiy/CenterPoint)
* [PointAugmenting](https://github.com/VISION-SJTU/PointAugmenting)
* [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
* [Cylinder3D](https://github.com/xinge008/Cylinder3D)
* [det3d](https://github.com/poodarchu/det3d)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [mmcv](https://github.com/open-mmlab/mmcv)