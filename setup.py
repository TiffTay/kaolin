import os
from setuptools import setup, find_packages
import importlib
import warnings
import logging
import glob
import subprocess

# Skip all version checks – Replicate forces PyTorch 2.7.0+
import torch
import numpy
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

# Setup logging and working directory
cwd = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger()
logging.basicConfig(format='%(levelname)s - %(message)s')

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    return raw_output, release[0], release[1][0]

# Handle CUDA availability
if not torch.cuda.is_available() and os.getenv('FORCE_CUDA', '0') == '1':
    logging.warning(
        "Torch did not find available GPUs. Assuming cross-compilation.\n"
        "Default architectures: Pascal (6.0, 6.1, 6.2), Volta (7.0), Turing (7.5),\n"
        "Ampere (8.0) if CUDA >= 11.0, Hopper (9.0) if CUDA >= 11.8, Blackwell (12.0) if CUDA >= 12.8\n"
        "Set TORCH_CUDA_ARCH_LIST for specific architectures."
    )
    if os.getenv("TORCH_CUDA_ARCH_LIST") is None:
        _, major, minor = get_cuda_bare_metal_version(CUDA_HOME)
        major, minor = int(major), int(minor)
        if major == 11:
            if minor == 0:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
            elif minor < 8:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9;9.0"
        elif major == 12:
            if minor <= 6:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0"
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0;12.0"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"
        print(f'TORCH_CUDA_ARCH_LIST: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
elif not torch.cuda.is_available():
    logging.warning(
        "No GPUs found. Installing with CPU support only. "
        "Set FORCE_CUDA=1 for GPU cross-compilation."
    )

# Package metadata
PACKAGE_NAME = 'kaolin'
DESCRIPTION = 'Kaolin: A PyTorch library for accelerating 3D deep learning research'
URL = 'https://github.com/NVIDIAGameWorks/kaolin'
AUTHOR = 'NVIDIA'
LICENSE = 'Apache License 2.0'
LONG_DESCRIPTION = """
Kaolin is a PyTorch library aiming to accelerate 3D deep learning research. Kaolin provides efficient implementations
of differentiable 3D modules for use in deep learning systems. With functionality to load and preprocess several popular
3D datasets, and native functions to manipulate meshes, pointclouds, signed distance functions, and voxel grids, Kaolin
mitigates the need to write wasteful boilerplate code. Kaolin packages together several differentiable graphics modules
including rendering, lighting, shading, and view warping. Kaolin also supports an array of loss functions and evaluation
metrics for seamless evaluation and provides visualization functionality to render the 3D results. Importantly, we curate
a comprehensive model zoo comprising many state-of-the-art 3D deep learning architectures, to serve as a starting point
for future research endeavours.
"""

version_txt = os.path.join(cwd, 'version.txt')
with open(version_txt) as f:
    version = f.readline().strip()

def write_version_file():
    version_path = os.path.join(cwd, 'kaolin', 'version.py')
    with open(version_path, 'w') as f:
        f.write(f"__version__ = '{version}'\n")

write_version_file()

def get_requirements():
    requirements = []
    with open(os.path.join(cwd, 'tools', 'viz_requirements.txt'), 'r') as f:
        requirements.extend(line.strip() for line in f)
    with open(os.path.join(cwd, 'tools', 'requirements.txt'), 'r') as f:
        requirements.extend(line.strip() for line in f)
    return requirements

def get_scripts():
    return ['kaolin/experimental/dash3d/kaolin-dash3d']

def get_extensions():
    extra_compile_args = {'cxx': ['-O3']}
    define_macros = []
    include_dirs = []
    sources = glob.glob('kaolin/csrc/**/*.cpp', recursive=True)
    is_cuda = torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1'

    if is_cuda:
        define_macros += [("WITH_CUDA", None), ("THRUST_IGNORE_CUB_VERSION_CHECK", None)]
        sources += glob.glob('kaolin/csrc/**/*.cu', recursive=True)
        extension = CUDAExtension
        extra_compile_args['nvcc'] = ['-O3', '-DWITH_CUDA', '-DTHRUST_IGNORE_CUB_VERSION_CHECK']
        include_dirs = get_include_dirs()
    else:
        extension = CppExtension

    extensions = [
        extension(
            name='kaolin._C',
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs
        )
    ]
    
    for ext in extensions:
        ext.libraries = ['cudart_static' if x == 'cudart' else x for x in ext.libraries]

    use_cython = True
    ext_suffix = '.pyx' if use_cython else '.cpp'
    cython_extensions = [
        CppExtension(
            'kaolin.ops.conversions.mise',
            sources=[f'kaolin/cython/ops/conversions/mise{ext_suffix}'],
        ),
    ]
    
    if use_cython:
        from Cython.Build import cythonize
        from Cython.Compiler import Options
        compiler_directives = Options.get_directive_defaults()
        compiler_directives["emit_code_comments"] = False
        cython_extensions = cythonize(
            cython_extensions,
            language='c++',
            compiler_directives=compiler_directives
        )
    
    return extensions + cython_extensions

def get_include_dirs():
    include_dirs = []
    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        _, major, _ = get_cuda_bare_metal_version(CUDA_HOME)
        if "CUB_HOME" in os.environ:
            logging.warning(f"Including CUB_HOME: {os.environ['CUB_HOME']}")
            include_dirs.append(os.environ["CUB_HOME"])
        elif int(major) < 11:
            logging.warning(f"Including default CUB: {os.path.join(cwd, 'third_party/cub')}")
            include_dirs.append(os.path.join(cwd, 'third_party/cub'))
    return include_dirs

if __name__ == '__main__':
    setup(
        name=PACKAGE_NAME,
        version=version,
        author=AUTHOR,
        description=DESCRIPTION,
        url=URL,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        python_requires='~=3.7',
        packages=find_packages(exclude=('docs', 'tests', 'examples')),
        scripts=get_scripts(),
        include_package_data=True,
        install_requires=get_requirements(),
        zip_safe=False,
        ext_modules=get_extensions(),
        cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)}
    )
