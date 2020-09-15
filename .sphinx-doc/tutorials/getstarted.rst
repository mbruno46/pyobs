Getting Started
===============

In this tutorial we demonstrate how to create primary 
observables from raw measurements and compute their errors.

Creating the primary observables
################################

Let us suppose to have measured a pion correlator 
on a lattice of size :math:`16^3 \times 32`
called *E0* on a 1000 configurations::

   >>> import numpy
   >>> data = numpy.loadtxt('correlator_T32_ncnfg1000_E0_R0')

``data`` is now a 1D numpy.array object with 32 x 1000 entries, 
such that for every 32 consecutive elements we have the correlator
on a given configuration. We also assume that those measurements
have been performed on subsequent configurations::

   >>> idx = range(1000)

We can now create an observable using the class ``observa``::

   >>> from pyobs import observa
   >>> tmax = 32
   >>> corr = observa()
   >>> corr.primary_observable(0,'E0',[0],['R0'],[idx],[data],(1,tmax))

The input parameters of ``primary_observable``:

1. an integer which uniquely indentifies the ensemble
2. a string that labels the ensemble. This is an 
   auxiliary field, but now crucial in the identification of the ensemble
3. a list with the indices that uniquely identify the replicas
4. a list of strings with the names of each replica. As before this is not 
   used to identify the replica
5. a list of integers with the indices of the configurations corresponding to the
   measurements contained in ``data``
6. a list of floats with the measurements configuration by configuration
7. a 2D tuple [optional] with the dimensions of the observable. Default is (1,1).
   Note that for vectors the formats (1,x) or (x,1) can be used

Multiple replicas
#################

If the correlator has been measured on more than one replica 
the ``observa`` must be re-created::

   >>> data = []
   >>> data.append( numpy.loadtxt('correlator_T32_ncnfg1000_E0_R0') ) # first replica
   >>> idx.append( range(1000) )
   >>> data.append( numpy.loadtxt('correlator_T32_ncnfg600_E0_R1') )
   >>> idx.append( range(600) )
   >>> corr = observa()
   >>> corr.primary_observable(0,'E0',[0,1],['R0','R1'],idx,data,(1,tmax))

Errors (quick)
##############

The error (for each element of the observable) can be calculated
using the function ``vwerr()`` as shown here::

   >>> [c, dc] = corr.vwerr()
   >>> print c
   [ 1.50881841  1.09511523  0.8530834 ... 1.0948965 ]
   >>> print dc
   [ 0.0012057   0.00058748  0.00064241 ... 0.0005396 ]

Alternatively, the user can *print* the observable in a 
nice format with the error displayed in parenthesis::

   >>> print corr
       1.5088(12)    1.09512(59)    0.85308(64)    ...    1.09490(54)

For a complete tutorial on error calculation read :doc:`Computing the error <error>`

Accessing the internal structure
--------------------------------

The mean values of an observable can be extracted at any time; 
note that regardeless of the original size of the observable 
it is always a 2D numpy.array::

   >>> print corr.mean
   [[ 1.50881841,  1.09511523,  0.8530834 , ...,  1.0948965 ]])
   >>> print corr.mean[0,1:4]
   [ 1.09511523  0.8530834   0.69602007]
   >>> print corr.mean[0,0]
   1.50881841113

The full observable can be sliced exactly in the same way 
(more details can be found in `Manipulation of observables <mathfuncs.html#>`_),
using square brackets::

   >>> print corr[0,1:4]
       1.09512(59)    0.85308(64)    0.69602(49)
   >>> print corr[0,0]
       1.5088(12)

.. warning::
   The ``observa`` class is defined such that each observable is 
   always a 2D matrix: therefore the slicing or accessing of elements
   must always take this into account using ``[0,0]`` for scalar 
   observables and ``[0,:]`` or ``[:,0]`` for vectors.

Finally, another useful function is *peek* that allows 
the user to check on which ensembles and replicas an 
observable is defined

.. code-block:: python

    >>> corr.peek()
    [[ 1.50881841  1.09511523  0.8530834 ... 1.0948965 ]]
    dimensions  (1, 32)
    --- ensemble  0   E0
        --- replica  0   R0
	        --- ncnfg  1000
        --- replica  1   R1
	        --- ncnfg  600

Save/Load
---------

The observables defined with the class `observa`
can be saved and loaded to disk using the simple commands::

   >>> obs.save('filename')
   >>> obs.load('filename.pyobs.gz')

When saving the observable to disk the extension `pyobs.gz` can be omitted and
it will be added automatically. The file format is very simple: 
the class is dumped into a single string using the json python library
and then compressed in the final file. This allows human-readability::

   $ gzip -d test-observable.pyobs.gz
   $ less test-observable.pyobs

Other file formats are supported as well and can be used 
by specifing the appropriate extension
in the name of the file. For example the observable can be saved in binary format 
or in MATLAB format::

   >>> obs.save('test-observable.pyobs.dat') 
   >>> obs.save('test-observable.xml.gz') 

When loading an observable the full file name must be specified including
the extension.

External observables
####################

Often in analysis of lattice observables, external parameters or
quantities have to be included: for example PDG values of hadron 
masses or calculations of quantities by different collaborations 
for which the full statistical sample is not available. In all these
cases the statistical information is contained in covariance matrices.
The ``pyobs`` library allows the user to incorporate these quantities
and properly propagate their error throughout the entire analysis.

For simplicity let's consider an experiment that measures a 
certain quantity :math:`A_i` for several energy bins labeled by
:math:`i`. If we apply a function :math:`f` its error is determined
from

.. math::
   \sigma_f^2 = \frac{\partial f}{\partial A_i} \mathrm{cov}^{ij} \frac{\partial f}{\partial A_j}

To create such an observable, one starts as usual from the class
`observa`

.. code-block:: python

   >>> val = [0.00026333, 0.048661, 0.0352]
   >>> cov = [[1.18122806e-07, 3.03071340e-08, 2.01081569e-08],
   >>>        [3.03071340e-08, 2.08975942e-04, 7.08967804e-06],
   >>>        [2.01081569e-08, 7.08967804e-06, 4.54329935e-05]]
   >>> exp = observa()
   >>> exp.c_observable(0,'experiment',val,cov)
   >>> print exp
   	0.00026(34)	0.049(14)	0.0352(67)
   >>> (exp[0,0] * exp[0,2]).vwerr()
   [9.269216e-06, 1.2242634568573964e-05]


The input parameters of ``c_observa`` are:

   1. an integer which uniquely indentifies the quantity
	
   2. a string that labels the quantity. This is an auxiliary field, but now crucial

   3. a list of floats with the values of the observable

   4. a list or ``numpy.array`` with the covariance matrix. 

.. warning::
   Note that the integer used to identify the `c_observable` is completely
   independent from the integers used to identify the ensembles.

Additional possible cases and allowed input

.. code-block:: python

   >>> val = [0.00026333, 0.048661, 0.0352]
   >>> cov = [1.18122806e-07, 2.08975942e-04, 4.54329935e-05]
   >>> exp = observa()
   >>> exp.c_observable(0,'experiment',val,cov)
   >>> print exp
   	0.00026(34)	0.049(14)	0.0352(67)
   >>> (exp[0,0] * exp[0,2]).vwerr()
   [9.269216e-06, 1.222740068372286e-05]

.. warning::
   If the observable is 2D the expected covariance matrix is a 4D object 
   (possibly a ``numpy.array``)


