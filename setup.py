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
            "csrc/kernels/ternary_gemm.cu",              # Unpacked weights
            "csrc/kernels/ternary_gemm_production.cu",   # Packed transposed (production)
            "csrc/kernels/ternary_gemm_tensor_core.cu",  # Tensor Core kernel (experimental)
        ],
        include_dirs=[
            os.path.join(ROOT_DIR, "csrc"),
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            # -arch=sm_86 for Ampere (RTX 30xx, A100, etc.)
            # Enables mma.sync INT8 instructions
            "nvcc": ["-O3", "-arch=sm_86"],
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
