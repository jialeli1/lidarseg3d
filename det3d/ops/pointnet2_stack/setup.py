from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointnet2_stack', # 这个name好像没啥用?
    ext_modules=[
        CUDAExtension('pointnet2_stack_cuda', [
            'src/pointnet2_api.cpp',
            'src/ball_query.cpp',
            'src/ball_query_gpu.cu',
            'src/group_points.cpp',
            'src/group_points_gpu.cu',
            'src/sampling.cpp',
            'src/sampling_gpu.cu', 
            'src/interpolate.cpp', 
            'src/interpolate_gpu.cu',
            'src/voxel_query.cpp', 
            'src/voxel_query_gpu.cu',
            # 'src/cube_query.cpp', 
            # 'src/cube_query_gpu.cu',
        ],
        extra_compile_args={'cxx': ['-g', '-I /usr/local/cuda/include'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
