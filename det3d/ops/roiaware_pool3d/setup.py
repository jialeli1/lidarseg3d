from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='roiaware_pool3d', # 这个name好像没啥用?
    ext_modules=[
        CUDAExtension('roiaware_pool3d_cuda', [
            'src/roiaware_pool3d.cpp',
            'src/roiaware_pool3d_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g', '-I /usr/local/cuda/include'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
