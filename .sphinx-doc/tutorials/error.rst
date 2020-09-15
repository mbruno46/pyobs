Computing the error
===================

In this tutorial we show the various methods
that can be used to compute and study 
the error of an observable.


The error (for each element of the observable) can be calculated
using the function `vwerr` as shown here::

   >>> [c, dc] = corr.vwerr()
   >>> print c
   [ 1.50881841  1.09511523  0.8530834 ... 1.0948965 ]
   >>> print dc
   [ 0.0012057   0.00058748  0.00064241 ... 0.0005396 ]


The method used for the estimate of the error is the
`Gamma method`. The summation window is defined through 
an automatic procedure, explained in the `introduction <intro.html#>`_
which depends on some parameters. 
The user can freely modify those, ensemble by ensemble, through the
class `errinfo`.


ErrInfo class
-------------

We provide here an example where we show how
to tweak the parameter :math:`S_\tau` for the
ensemble with id 0::

   >>> einfo = errinfo()
   >>> einfo.addEnsemble(0,Stau=3.0)
   >>> obs.vwerr(errinfo=einfo) # uses the new Stau
   >>> obs.vwerr() # standard method with Stau=1.5


Note that the automatic window can be bypassed
by imposing a fixed summation window with 
the argument ``W`` of the class `errinfo`::

   >>> einfo = errinfo()
   >>> einfo.addEnsemble(0,W=30) 
   >>> obs.vwerr(errinfo=einfo) # gamma summed up to W
   >>> obs.vwerr() # standard method with automatic window


The `errinfo` class can be quickly inspected using 
the `print` command::

   >>> print einfo
   Ensemble id 0 
        Stau = 1.5 ; Texp = 10.00 ; Nsigma = 2 



Standard UWerr approach
-----------------------

The standard approach described in the 
`introduction <intro.html#>`_
is usually called `UWerr`. The automatic
window procedure depends on :math:`S_\tau`
alone. The examples above show how to changed
this parameter ensemble by ensemble.

.. warning::
   To make sure that the `UWerr` method is enforced
   on a given ensemble, check that 
   the `errinfo` class **Texp=0**. 


Attaching Exponential Tail
--------------------------

In order to use the approach `VWerr` 
on a given ensemble, the user must define 
the `errinfo` class with non-zero ``Texp``
field::

   >>> einfo = errinfo()
   >>> einfo.addEnsemble(0,Texp=10.)
   >>> obs.vwerr(errinfo=einfo)


Plotting the autocorrelation function
-------------------------------------

The plotting of the autocorrelation function :math:`\rho`
is performed with the library `matplotlib` and
can be activated with the flag ``plot`` in `vwerr`::

   >>> obs.vwerr(True) # to simply plot
   >>> obs.vwerr(plot=True) # equivalent as line above

.. image:: ../figs/example_rho_E0_00.pdf


.. sidebar:: Explaining the plot

   The title contains the ensemble name and the integrated autocorrelation
   time. The vertical line shows the optimal window found by the
   automatic procedure (U.Wolff).

The plots can be saved to a pdf file by using the ``pfile`` field
in `vwerr`. Note that the argument of ``pfile`` must be a base name
for the file, without the extension, e.g. ``/home/users/tmp/test_observable``:
the program automatically attaches to it the ensemble name and the observable
components and saves a file like ``/home/users/tmp/test_observable_E0_00.pdf``.
If multiple ensembles or if the observable is a vector or a matrix, 
then multiple plots and files will be generated with the 
appropriate names ending with ``_ensemblename_ij.pdf``::

   >>> obs.vwerr(True, '/path/to/file' ) # to plot and save the figure to disk
   >>> obs.vwerr(pfile='/path/to/file', plot=True) # same as above
   >>> obs.vwerr(False, '/path/to/file') # raises an error

If the exponential tails are defined in a given `errinfo` class
then an exponential function is displayed in the plot as well::

   >>> einfo = errinfo()
   >>> einfo.addEnsemble(0,Texp=10.)
   >>> obs.vwerr(plot=True, errinfo=einfo)

.. image:: ../figs/example_rho_tail_E0_00.pdf


If the observable contains the fluctuations of 
many ensembles, an additional pie chart is plotted automatically
displaying which ensemble contributes to the final error and in
what percentage

.. image:: ../figs/example_pie.pdf


.. warning::
   The function ``vwerr`` generates one plot for every element of the observable and
   for every ensemble. In this example, it is better to avoid 
   plotting the entire correlator with ``corr.vwerr(True)``.

Additionally it is also possible to visualize the fluctuations of a given
observable by calling the function `plotter`::

   >>> obs.plotter()

.. warning::
   Also in this case one plot per ensemble and component will be generated

(Pseudo)-jackknife error
------------------------

As an alternative it is also possible to estimate the error using the 
jackknife method. In general this method is extremely powerful as it
allows to automatically propagate the error by applying the wanted 
function to each jackknife bin, thus also including the derivative 
of the function. In ``pyobs`` instead the derivative is computed at each
step therefore the jackknife error estimation is applied to the 
fluctuations :math:`\pi_\alpha^{i,r}`.
In essence, the function `jkerr` generates jackknife bins of the 
fluctuations of the observables and computes the error from those,
by simply specifying the bin size. 

Similarly to the `errinfo` class, the bin size can be 
specified for each ensemble separately through a dictionary. 
If not specified a bin size of 1 is assumed for all ensembles.
The dictionary is in the form a:b with a the ensemble id and b the
bin size:::

   >>> [v, e] = obs.jkerr() # default bin size of 1 for all ensembles
   >>> bsize = {0:2, 1:1, 2:4}
   >>> # ensemble 0 has bin size of 2, ensemble 1 has bin size of 1, etc...
   >>> [v, e] = obs.jkerr(jkinfo = bsize)
   >>> obs.jkerr(True) # to plot the piechart



