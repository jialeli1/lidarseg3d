from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='voxel_layer', # 这个name好像没啥用?
    ext_modules=[
        CUDAExtension('voxel_layer', [
            'src/voxelization.cpp',
            'src/scatter_points_cpu.cpp',
            'src/scatter_points_cuda.cu',
            'src/voxelization_cpu.cpp',
            'src/voxelization_cuda.cu',
        ],
        define_macros=[('WITH_CUDA', None)], # NOTE THIS.
        extra_compile_args={'cxx': ['-g', '-I /usr/local/cuda/include'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
