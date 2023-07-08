## Installation
Modified from [CenterPoint](https://github.com/tianweiy/CenterPoint)'s original document. Therefore, you may consider looking for a solution from the CenterPoint issues if you are experiencing installation problems.

### Our Environment

- Linux
- Python 3.7
- PyTorch 1.7.1+cu110
- CUDA 11.0
- CMake 3.18.4.post1
- [APEX](https://github.com/nvidia/apex)
- [spconv](https://github.com/traveller59/spconv) 

#### Notes
- A rule of thumb is that your pytorch cuda version must match the cuda version of your systsem for other cuda extensions to work properly. 
- We don't have enough resources to test different environments, please consider using our code in a similar environment.
- More packages should be installed for MSeg3D according to the updated "requirements.txt". We specified the version we used for stability, other versions can be tried if your environment has difficulties with the installation.

### Basic Installation 

```bash
# activate a created virtual environment as lidarseg
conda activate lidarseg

# please follow the official instructions to install pytorch and torchvision in advance

git clone https://github.com/jialeli1/lidarseg3d.git
cd lidarseg3d
pip install -r requirements.txt

# NOTE: add lidarseg3d to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly) and reactivating it
export PYTHONPATH="${PYTHONPATH}:PATH_TO_lidarseg3d"
```

### Advanced Installation 

#### [nuScenes dev-kit](https://github.com/nutonomy/nuscenes-devkit)

```bash
# we recommend installing nuscenes-devkit with this tested version (1.1.6)
pip install nuscenes-devkit==1.1.6

# add the following line to ~/.bashrc and reactivate bash (remember to change the PATH_TO_NUSCENES_DEVKIT value)
# $ export PYTHONPATH="${PYTHONPATH}:PATH_TO_NUSCENES_DEVKIT/python-sdk"
# in our case:
export PYTHONPATH="${PYTHONPATH}:PATH_TO_lidarseg3d/nuscenes-devkit/python-sdk"
```

#### Cuda Extensions

```bash
# set the cuda path (please change the path to your own cuda location!).
# depending on your machine, some additional environment variables that may be required: CUDNN_INCLUDE_DIR, CUDNN_LIB_DIR, CUDNN_PATH, CUDNN_LIBRARY, CPLUS_INCLUDE_PATH
export PATH=/usr/local/cuda-11.0/bin:$PATH
export CUDA_PATH=/usr/local/cuda-11.0
export CUDA_HOME=/usr/local/cuda-11.0
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH

bash setup.sh 
```


#### [mmcv](https://github.com/open-mmlab/mmcv)

```bash
# please follow the offical installation instructions from mmcv, which usually takes about 10 minutes or so.
# just be careful to choose the right mmcv version according to your cuda and pytorch.

# in our case:
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
```



#### [APEX](https://github.com/nvidia/apex) (Optional)

```bash
# CenterPoint uses "apex" for sync-bn, which is also preserved in our code. 
# but we recommend using "torch" instead of "apex" for sync-bn. 
# therefore, apex is not required to be installed if you don't use it.
# you can control it by setting the "sync_bn_type" parameter in the model config file.
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 5633f6  # recent commit doesn't build in our system 
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

#### [spconv](https://github.com/traveller59/spconv) 
```bash
# you may encounter some installation problems, but the solutions can be found from the issues of spconv repository.
# we use the spconv with commit fad3000249d27ca918f2655ff73c41f39b0f3127.
sudo apt-get install libboost-all-dev

# please take care to check if the "spconv/third_party" directory has been downloaded successfully.
git clone https://github.com/traveller59/spconv.git --recursive

python setup.py bdist_wheel
cd ./dist && pip install *
```

#### Check out [GETTING_START](GETTING_START.md) to prepare the data and play with model training/evaluation. 