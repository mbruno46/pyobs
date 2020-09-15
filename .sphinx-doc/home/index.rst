PyObs
=====

A Python library to analyze lattice observables. 

   * entirely based on the `numpy` library, with some additional modules written in c
   * objected-oriented with wide basis of overloaded operations
   * flexible calculation of the error with the `Gamma method`
   * analytic propagation of the error for all built-in functions, including fitting routines
   * support for :math:`\chi^2_\mathrm{exp}`

Current stable version is |release|

Introduction
------------

.. toctree::
   :maxdepth: 2

   intro
   first

TODO
----

.. todo::

   * add chi2exp
   * add bias cancellation for mean value
   * add support for writing MATLAB dobs .xml.gz
   * add support for prod
   * add support for eigenvalues and eigenvectors
   * cythonize other core functions beyond drho
   * add support for complex numbers (?)

.. todo::

   * test file formats: json, binary and xml

.. note::

   * uwerr tested for a single replica against original MATLAB implementation
   * plotter of autocorrelations and fluctuations tested
   * maker.sh tested on Mac
   * fitting routines tested (see quark-mass project)
