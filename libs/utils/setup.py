# adopt from https://github.com/happyharrycn/actionformer_release/blob/main/libs/utils/setup.py

import torch
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name='nms_1d_cpu',
    ext_modules=[
        CppExtension(
            name = 'nms_1d_cpu',
            sources = ['./csrc/nms_cpu.cpp'],
            extra_compile_args=['-fopenmp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
