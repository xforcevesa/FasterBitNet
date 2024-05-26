from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
      name='gemm_lowbit_cpp',
      version='0.1',
      ext_modules=[
            cpp_extension.CppExtension(
                  'gemm_lowbit_cpp',
                  ['gemm_lowbit_kernel.cu'],
                  extra_compile_args=['-std=c++20', '-O3']
            )
      ],
      cmdclass={
          'build_ext': cpp_extension.BuildExtension
      })