from setuptools import setup, Extension, find_packages
import os

VERSION = (1, 5, 2)

def get_numpy_include():
    import numpy
    return numpy.get_include()
    
def version():
    v = ".".join(str(v) for v in VERSION)
    cnt = f'__version__ = "{v}"\n__version_full__ = __version__\n'
    with open('pyobs/version.py', 'w') as f:
        f.write(cnt)
    with open('pyobs/VERSION','w') as f:
        f.write(f'v{v}')
    return v

ext = Extension('pyobs.core.mftools', ['pyobs/core/mftools.cc'],
                include_dirs=[get_numpy_include()])

setup(
    name='pyobs',
    version=version(),
    include_package_data=True,
    packages=find_packages(include=['pyobs*']),
    ext_modules=[ext],
)
