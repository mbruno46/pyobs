Built-in functions
==================


The class `observa` is defined with built-in 
overloaded operators, that let the user quickly 
obtain derived observables. Below we show some
examples based on the following objects

* `mat1` matrix observable with size 3x4
* `mat2` matrix observable with size 4x3
* `mat3` matrix observable with size 3x4, different from `mat1`
* `c` scalar observable (size 1x1)


.. code-block:: python

   >>> mat1 + mat3
   >>> -mat1
   >>> mat1 - mat3
   >>> mat1**2 # element-wise power 

Multiplications and divisions::

   >>> mat1 * mat3 # element-wise
   >>> mat1.dot(mat2)  # dot product
   >>> mat1.reciprocal() # element-wise
   >>> 1./mat1 # identical to reciprocal
   >>> mat1 * mat3.reciprocal() # element-wise

The only operation that permits the combination 
of observables with different sizes is the product 
between a vector/matrix with a scalar::

   >>> mat1 * c

Reductions::

   >>> mat1.sum(0) # returns 1x4 vector
   >>> mat1.sum(1) # return 3x1 vector

.. warning::
   If we ware interested in the product/sum/difference of an observable with 
   a `numpy.array` object we must defined the binary operation with the 
   `observa` as the first element. If we do the opposite an error is thrown
   because the routine `__add__` from the `numpy` library is picked
   first, instead of `__radd__` of the library `pyobs`::

   >>> obs * a # this works only if a.shape coincides with obs.dims
   >>> a * obs # this fails

scalar functions
----------------

The following functions can be imported using::

   >>> from pyobs import *
   >>> from pyobs import log, exp, ...

.. automodule:: pyobs.observa
   :members: log, exp, sin, cos, arccos, arcsin

matrix functions
----------------

The following functions can be imported using::

   >>> from pyobs import *
   >>> from pyobs import det, inv

.. automodule:: pyobs.observa
   :members: det, trace, inv
