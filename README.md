# pyobs

A Python library to analyze lattice observables.

   * entirely based on the `numpy` library, with some additional modules written in c
   * objected-oriented with wide basis of overloaded operations
   * flexible calculation of the error with the `Gamma method`
   * analytic propagation of the error for all built-in functions, including fitting routines
  
## Usage

PyObs can be used without installing
it in the python library, by following these steps
   
   1. git clone the *master branch*
      ```bash
      $ git clone git@github.com:mbruno46/pyobs.git -b master
      ```
   2. run `maker.sh` to compile a fast c library of core functions used in the package;
    this script requires one argument which is the location of the C compiler
      
      ```bash
      $ cd pyobs/pyobs/core
      $ sh maker.sh /usr/bin/gcc
      ```
   
   3. to use the package in a python script simply add the path
   
      ```python
      >>> import sys
      >>> sys.path.insert(0,'/full/path/to/pyobs')
      >>> from pyobs import *
      ```
