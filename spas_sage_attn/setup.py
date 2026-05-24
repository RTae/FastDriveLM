"""
Build script for spas_sage_attn CUDA extensions (_qattn, _fused).

uv builds this automatically when the package is installed as a path dependency:

    # In the root pyproject.toml:
    # [tool.uv.sources]
    # spas-sage-attn = { path = "spas_sage_attn", editable = true }

Place your CUDA/C++ source files under:
    spas_sage_attn/csrc/qattn/   -> compiled into spas_sage_attn._qattn
    spas_sage_attn/csrc/fused/   -> compiled into spas_sage_attn._fused
"""

import os
import runpy
from glob import glob
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Resolve paths relative to this file so the build works regardless of CWD.
here = os.path.dirname(os.path.abspath(__file__))


def generate_instantiations() -> None:
    """Generate explicit template-instantiation sources required by qattn."""
    for rel_path in (
        os.path.join("csrc", "qattn", "instantiations_sm80", "autogen.py"),
        os.path.join("csrc", "qattn", "instantiations_sm89", "autogen.py"),
        os.path.join("csrc", "qattn", "instantiations_sm90", "autogen.py"),
    ):
        runpy.run_path(os.path.join(here, rel_path), run_name="__main__")


def should_build_sm90_sources() -> bool:
    """SM90 qattn kernels use WGMMA instructions that do not assemble for SM120."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    return (major, minor) == (9, 0)


def find_sources(subdir: str) -> list[str]:
    """Return all .cu and .cpp files found under csrc/<subdir>."""
    csrc = os.path.join(here, "csrc", subdir)
    sources = glob(os.path.join(csrc, "**", "*.cu"), recursive=True) + glob(os.path.join(csrc, "**", "*.cpp"), recursive=True)
    if not sources:
        raise FileNotFoundError(
            f"No CUDA/C++ sources found in {csrc!r}. "
            "Make sure you have copied the csrc/ folder from SpargeAttn."
        )
    if not should_build_sm90_sources():
        sources = [source for source in sources if "sm90" not in source]
    return [os.path.relpath(source, here) for source in sources]


generate_instantiations()


common_compile_args = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]

ext_modules = [
    CUDAExtension(
        name="spas_sage_attn._qattn",
        sources=find_sources("qattn"),
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": common_compile_args,
        },
    ),
    CUDAExtension(
        name="spas_sage_attn._fused",
        sources=find_sources("fused"),
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": common_compile_args,
        },
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
