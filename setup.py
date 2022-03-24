from setuptools import setup, Extension
from distutils.sysconfig import get_config_vars
from distutils.util import get_platform
from distutils.cmd import Command
from distutils.command.build_ext import build_ext

import os
import numpy

VERSION = (1, 4, 0)


def version():
    v = ".".join(str(v) for v in VERSION)
    cnt = f'__version__ = "{v}" \n__version_full__ = __version__'
    with open('pyobs/version.py', 'w') as f:
        f.write(cnt)
    return v


numpy_path = numpy.__file__.replace('__init__.py', '')
incpath = os.path.join(numpy_path, 'core/include')
ext = Extension('pyobs/core/mftools', ['pyobs/core/mftools.cc'],
                include_dirs=[incpath])


class SymLinkExt(Command):

    def initialize_options(self):
        vv = get_config_vars().get('VERSION')
        self.bdir = f'{os.getcwd()}/build/lib.{get_platform()}-{vv}'
        self.suffix = get_config_vars().get('EXT_SUFFIX')

    def finalize_options(self):
        pass

    def run(self):
        src = f'{self.bdir}/{ext.name}{self.suffix}'
        cmd = f'ln -s {src} {ext.name}.so'
        if os.path.isfile(src):
            out = os.popen(cmd).read()
            if out != '':
                raise Exception(f'SymLink failed: {out}')


class BuildExtLocal(build_ext):

    def run(self):
        build_ext.run(self)
        self.run_command('symlink_ext')


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pyobs",
    version=version(),
    author="Mattia Bruno",
    author_email="mattia.bruno@cern.ch",
    description="A Python library to analyse (auto) correlated data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mbruno46/pyobs.git",
    packages=['pyobs', 'pyobs/core', 'pyobs/tensor', 'pyobs/optimize',
        'pyobs/qft', 'pyobs/qft/finite_volume', 'pyobs/misc', 'pyobs/IO'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv2",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.18",
        "sympy>=1.5",
        "scipy>=1.4",
        "bison @ git+https://github.com/mbruno46/bison.git@main#egg=bison"
    ],
    ext_modules=[ext],
    cmdclass={
        'symlink_ext': SymLinkExt,
        'build_ext': BuildExtLocal,
        },
)
