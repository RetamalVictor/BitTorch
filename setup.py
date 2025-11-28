import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory containing this file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Source files for the extension
ext_modules = [
    CUDAExtension(
        name="bittorch._C",
        sources=[
            "csrc/core/dispatch.cpp",
            "csrc/bindings/bittorch_bindings.cpp",
        ],
        include_dirs=[
            os.path.join(ROOT_DIR, "csrc"),
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3"],
        },
    )
]

setup(
    name="bittorch",
    version="0.0.1",
    description="High-performance low-precision backend for PyTorch",
    author="RetamalVictor",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
)
