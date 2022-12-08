cd det3d/ops/dcn 
python setup.py build_ext --inplace

cd .. && cd iou3d_nms
python setup.py build_ext --inplace

cd .. && cd pointnet2_batch
python setup.py build_ext --inplace

cd .. && cd pointnet2_stack
python setup.py build_ext --inplace

cd .. && cd roiaware_pool3d
python setup.py build_ext --inplace

cd .. && cd voxel
python setup.py build_ext --inplace
