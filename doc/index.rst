.. pyobs documentation master file, created by
   sphinx-quickstart on Sat Apr 11 18:35:15 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |logo| image:: ./pyobs-logo.png
    :width: 60pt

pyobs |release|
================

A Python3 library to analyse data generated
from (Monte Carlo) Markov chains.           

.. The current stable version is |release|

The software is hosted on `GitHub <https://github.com/mbruno46/pyobs>`__.

Installation
------------

`pyobs` can be installed in three different ways.
After installation it can be imported like any other package

.. code-block:: python

   >>> import pyobs
   >>> help(pyobs.observable)

* **pip install from remote repository**

.. code-block:: bash
   
   $ pip install git+https://github.com/mbruno46/pyobs.git@master#egg=pyobs 
   $ # for upgrading
   $ pip install -U git+https://github.com/mbruno46/pyobs.git@master#egg=pyobs

* **pip install from local repository**


.. code-block:: bash

   $ git clone git+https://github.com/mbruno46/pyobs.git
   $ pip install ./pyobs

* **developer mode**

.. code-block:: bash

   $ pip install -e ./pyobs

The library can alternatively be used without installation
by adding the appropriate path to `sys`.
In this case it is recommended to recompile the C++ extensions
manually with `python setup.py build_ext`. 

.. code-block:: python

   >>> import sys
   >>> sys.path.append('/path/to/pyobs/directory/')
   >>> import pyobs

Note that recompiling the C++ extensions might be necessary
also after pulling the latest commits, in the developer mode. 


.. Documentation
.. -------------
.. 
.. .. toctree::
..    :maxdepth: 2
..    
..    intro/index
..    pyobs/index

Features
--------

* object-oriented for easy manipulation of data
* FFT acceleration for calculation of autocorrelations
* minimal memory footprint for large data sets and memory monitoring
  system
* support for master-field setup, ie autocorrelations in more
  than one dimension
* support for systematic errors and external input data
* flexible interface for fits
* support for linear algebra operations (e.g. matrix inverse, eigenvalue problem, etc..)

Authors
-------

Copyright (C) 2020, Mattia Bruno

Changelog
---------

.. include:: ../CHANGELOG

.. Indices and tables
   ==================
   
   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
