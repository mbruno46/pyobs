Fitting
=======

The library PyObs provides the class `mfit` to 
perform fits to observables. The class is pretty generic
allowing also for combined/global fits. 
The minimization algorithm is the Levenberg-Marquardt
and the gradient and hessian of the fitted function w.r.t.
the parameters are computed analytically: they are used in the
LM algorithm and also in the propagation of the error from
input (y) observables, which is also analytical.
Let us illustrate a few possible ways to use this class.

Fitting a correlator
--------------------

The first example concerns the fitting of observables defined
on a unique ensembles, such as a correlation function measured 
on all time slices.
Assuming that the `coor` observable is defined to be a vector 
observable with size `(1,T)`, we want to perform a single exponential
fit.

The free parameters of the fit are always defined using the notation
`p[i]`, with `i` an integer number. 
For the single exponential fit

.. math::
   
   f(x) = A \exp (-m x)

we first defined a new `mfit` class::
   
   >>> fit = mfit('p[0] * exp(-p[1]*x)')

Note that the function is passed as a string!
The minimization of an un-correlated :math:`\chi^2`
is obtained simply by::

   >>> xdata = numpy.arange(T)
   >>> fit.run(xdata, corr)
   >>> print fit
   >>> print fit.pars

where the first `print` command prints out some useful information
on the minimization procedure, such as the final value of the
:math:`\chi^2`, the number of iterations of the minimizator, etc...

Plotting the fitted function
----------------------------

The results of the fit, the parameters, are stored inside
the class `fit` and can be accessed by typing `fit.pars`.
However we provide a useful routine to evaluate the fit function
at some given points using the results of the fit::

   >>> xax = numpy.arange(0,T,0.1)
   >>> [yax, dyax] = fit.eval(xax)
   >>> errorbar(xax, yax[0,0], dyax[0,0])

Note the presence of the `[0,0]` in both `yax` and `dyax`. This point will 
be clarified in the next examples.

Fitting two correlators simulateneously
---------------------------------------

Let us assume to have two (2-point) correlators
with different operators at the source and sink,
`corr1` and `corr2`.

In order to fit them simulateneously the first step is to
combine them in a single observable::

   >>> corr3 = corr1.clone(True)
   >>> corr3.addrow( corr2 )

.. warning:: 
   The observables must be merged row by row: this is crucial
   because the programs automatically assumes that the columns
   correspond to the (y) data points of the fit.
   Note also that his operation is meaningful only if 
   both `corr1` and `corr2` are defined on the same ensemble.
   
At this point we have an observable with dimensions `(2,T)`. Therefore
we need to specify two functions::

   >>> fit = mfit(['p[0]*exp(-p[2]*x)','p[1]*exp(-p[2]*x)']) # p[2] is common parameter
   >>> xdata = numpy.arange(T) # 1D fit
   >>> fit.run(xdata, corr3, [1., 1., 0.1])
   >>> print fit
   >>> print 'mass = ', fit.pars[0,2]

Fitting observables from different ensembles
--------------------------------------------

Good examples for fits of (derived) observables from
different ensembles are the chiral and continuum extrapolations.

Let us take the latter for simplicity::

   >>> asq = numpy.array([0.1, 0.08, 0.06])**2 # lat spacing in fm
   >>> obs = [obs1, obs2, obs3] # a generic scalar observable defined on the 3 ensembles

In this case the observables measured on each ensemble are patched together
in a *list*. We assume to take the continuum limit in :math:`a^2`::

   >>> fit = mfit('p[0] + p[1]*x')
   >>> fit.run(asq, obs)
   >>> print fit

.. note::
   Note the importance of patching together the observables in a list

Fitting observables from different ensembles - II
-------------------------------------------------

Let us now suppose to have measured two different quantities 
:math:`\phi_1` and :math:`\phi_2` both with a well defined and unique 
continuum limit: in this case we want to perform a constrained extrapolation.
As before, we first patch together the observables ensemble by ensemble
(without the constraint of having to use `rows`)::

   >>> phi1 = [phi1_E0, phi1_E1, phi1_E2]
   >>> phi2 = [phi2_E0, ... ]
   >>> phi3 = []
   >>> for i in range(3):
   >>>    tmp = phi1[i].clone(True)
   >>>    tmp.addcol(phi2[i]) # or equivalently tmp.addrow(phi2[i])
   >>>	  phi3.append(tmp)

Then we can perform the fit by defining the correct list of functions::

   >>> fit = mfit([['p[0]+p[1]*x','p[0]+p[2]*x']]) # if we used addcol
   >>> fit = mfit([['p[0]+p[1]*x'],['p[0]+p[2]*x']]) # if we used addrow
   >>> fit.run(asq, phi3)
   >>> print fit
   >>> print fit.pars

We can also imagine the situation where we have computed :math:`\phi_1` 
and :math:`\phi_2` on 6 different ensembles::

   >>> phi1 = [phi1_E0, phi1_E1, phi1_E2, phi1_E3]
   >>> asq1 = numpy.array([0.1, 0.08, 0.06, 0.05])**2
   >>> phi2 = [phi2_E4, phi2_E5]
   >>> asq2 = numpy.array([0.09, 0.065])**2

In this case the constrained fit can be implemented by performing
a bi-dimensional fit, where the slope parameter now explicitly depends
on the second coordinate `c`::

   >>> phi3 = phi1 + phi2 # simply join the two lists
   >>> xdata = []
   >>> for i in range(4):
   >>>     xdata.append([asq1[i], 1, 0])
   >>> for i in range(2):
   >>>     xdata.append([asq2[i], 0, 1])
   >>> xdata = numpy.array(xdata)
   >>> fit = mfit('p[0]+p[1]*x*c1 + p[2]*x*c2', var='x, c1, c2') # c1 and c2 select the function
   >>> fit.run(xdata, phi3)

Most generic fit
----------------

The most generic case that the `mfit` function can handle 
is defined from the minimization 
of a :math:`\chi^2` where 
the functions can be matrices
with indices :math:`(\alpha \beta)` 
and the coordinates :math:`x` can have multiple dimensions :math:`x^i_\mu`

.. math::
   \chi^2 = \sum_\alpha \sum_{\beta \rho} \sum_{ij} 
   (f_{\alpha\beta}(x^i_\mu) - y_{\alpha \beta}^i) 
   W_{\beta \rho}^{ij} (f_{\rho \alpha}(x^j_\mu) - y_{\rho \alpha}^j) 

If the weight matrix is block-diagonal w.r.t. the :math:`\beta \rho`
indices, the :math:`\chi^2` defined above simply reduces to
a sum of :math:`\chi^2`, each defined by the function :math:`f_{\alpha\beta}`

.. math::
   \chi^2 = \sum_{\alpha \beta} \sum_{ij} (f_{\alpha\beta}(x^i_\mu) - y_{\alpha \beta}^i) 
   W_\beta^{ij} 
   (f_{\beta \alpha}(x^j_\mu) - y_{\beta \alpha}^j)

Let's examine again the case where we fit 
two correlators simulatenously, which we 
label :math:`C_\alpha(i)`, with :math:`alpha=1,2`
labels the correlator and the index :math:`i` 
the time slice.
In this case all data is correlated and 
from the covariance matrix the weight matrix 
:math:`W` can be defined as

.. math::
   \mathrm{Cov}_{\alpha \beta}^{ij} = \langle \delta C_\alpha(i) \delta C_\beta(j) \rangle
   \quad \to \quad W \equiv \mathrm{Cov}^{-1}

:math:`W` in this case is a dense matrix, with all entries being non-zero.
If instead we fit together two observables 
defined on different ensembles, such as 
:math:`\phi_1` and :math:`\phi_2` introduced above,
we can use a weight matrix that takes
into account correlations among the two 
for a fixed ensemble. In this case
:math:`\alpha=1,2` refers to :math:`\phi_1, \phi_2`
while the index :math:`i` refers to the 
ensemble (which is the data point). 
Therefore the covariance matrix
is block-diagonal in the :math:`ij` indices
and still dense in :math:`\alpha \beta`.


If the user decides to manually pass the matrix :math:`W`
the following structure must be followed::

   >>> W = numpy.zeros(Nalpha*Npt, Nalpha*Npt) 
   >>> # Nalpha number of observables
   >>> # Npt number of data points
   >>> for a in range(Nalpha):
   >>>    for b in range(Nalpha):
   >>>       for i in range(Npt):
   >>>          for j in range(Npt):
   >>>             W[a*Npt+i, b*Npt+j] = ... #


Polynomial fits
---------------


