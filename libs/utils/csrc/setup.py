from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='nms_1d_cpu',
    ext_modules=[
        cpp_extension.CppExtension(
            name='nms_1d_cpu',
            sources=['nms_cpu.cpp']
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)