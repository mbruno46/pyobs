First Steps with PyObs
======================

Getting PyObs
-------------

You can download the latest stable release 
from the git repository `pyobs <https://github.com/mbruno46/pyobs>`_.

Install PyObs
-------------

Usage w/o installation PyObs
----------------------------

PyObs can be used without installing
it in the python library, by following these steps

 1. git clone the *master branch*

    $ cd /home/analysis
    $ git clone git@github.com:mbruno46/pyobs.git -b master 

 2. run `maker.sh` to compile a fast c library of core functions used in the package; 
    this script requires one argument which is the location of the C compiler::
     
    $ cd pyobs/pyobs/core
    $ sh maker.sh /usr/bin/gcc

 3. to use the package in a python script simply add the path with::

    >>> import sys
    >>> sys.path.insert(0,'/full/path/to/pyobs')
    >>> from pyobs import * 

.. warning::
   Make sure to add the path that points to the directory that
   contains the subfolder named ``pyobs``, which is the true package 
   loaded by ``import``. In our example above do not use `pyobs/pyobs`

