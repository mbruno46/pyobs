.. pyobs documentation master file, created by
   sphinx-quickstart on Sat Apr 11 18:35:15 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The Gamma Method
================

Here we skect the so-called Gamma method; 
more details can be found in Ref [1]_ which has been the base for
many of the routines implemented in this library.

Autocorrelated data
-------------------

Let us imagine to have a generic set of primary observables
:math:`\{A_\alpha\}` measured on :math:`R` replicas of the same ensemble,
each having :math:`N_r` configurations with index :math:`i`.

We denote with :math:`N` the total number of configurations per ensemble
:math:`N=\sum_r N_r`.
The measurements are labeled as :math:`a_{\alpha}^{i,r}` and their
mean value and fluctuations are defined as

.. math::
   \bar a_\alpha^r = \frac{1}{N_r} \sum_{i=1}^{N_r} a_\alpha^{i,r} \quad
   \bar{\bar a}_\alpha  = \frac{1}{N} \sum_{r=1}^R N_r \bar a_\alpha^r =
   \frac{1}{N} \sum_{r=1}^R \sum_{i=1}^{N_r} a_\alpha^{i,r} \quad
   \quad
   \delta a_\alpha^{i,r} = (a_\alpha^{i,r} - A_\alpha)


The autocorrelation function is defined as

.. math::
   \langle \delta a_\alpha^{i,r} \ \delta a_\beta^{j,s} \rangle =
   \delta_{rs} \Gamma_{\alpha \beta} (j-i)

At the practical level the autocorrelation function is computed as

.. math::
   \Gamma_{\alpha \beta} (t) = \frac{1}{N-Rt} \sum_{r=1}^R \sum_{i=1}^{N_r -t}
   (a_\alpha^{i,r} - \bar{\bar a}_\alpha)(a_\beta^{i+t,r} - \bar{\bar a}_\beta)  + O(1/N)


To be as general as possible we also need to account for situations where the measurements
are taken on irregular Monte-Carlo series, e.g. with holes and missing configurations.
For this reason it is convenient to consider the normalized fluctuations

.. math::
   \pi_\alpha^{i,r} = \left\lbrace \begin{array}{ll}
   0 & \mbox{if } a_\alpha^{i,r} \mbox{ on config } i \mbox{ is missing} \\
   (a_\alpha^{i,r} - \bar{\bar a}_\alpha) & \mbox{otherwise}
   \end{array} \right. \quad

such that the time series given by :math:`\pi_\alpha^{i,r}` is now contiguous 
over :math:`N_r` configurations, 
including the zero fluctuations 
(:math:`N_m` is the number of measurements at disposal). 

To take into account the holes in the time series we must change the 
weigth factor :math:`1/(N-Rt)` accordingly. To do so we consider the 
following quantity

.. math::
   \hat \pi_\alpha^{i,r} = \left\lbrace \begin{array}{ll}
   0 & \mbox{if } a_\alpha^{i,r} \mbox{ on config } i \mbox{ is missing} \\
   1 & \mbox{otherwise}
   \end{array} \right. \quad

and we proceed to the calculation of the autocorrelation function
.. The calculation of the autocorrelation function therefore becomes

.. math::
   \Gamma_{\alpha \beta} (t) = \frac{ \sum_{r=1}^R \sum_{i=1}^{N_r -t}
   \pi_\alpha^{i,r} \pi_\beta^{i+t,r} }{
   \sum_{r=1}^R \sum_{i=1}^{N_r -t} \hat \pi_\alpha^{i,r} \hat \pi_\beta^{i+t,r}
   }

and its integral defines the covariance matrix

.. math::
   C_{\alpha \beta} = \sum_{t=-\infty}^\infty \Gamma_{\alpha \beta} (t)

The error of a given observable can be read off from the diagonal
elements of the covariance matrix, which can be re-expressed in terms
of the integrated autocorrelation time

.. math::
   \sigma^2 (a_\alpha) = \frac{C_{\alpha \alpha}}{N} = \frac{ 2 \tau_\mathrm{int} }{N}
   \Gamma_{\alpha \alpha}(0)

.. math::
   \tau_\mathrm{int\,, \alpha} = \frac{1}{2 \Gamma_{\alpha \alpha}(0)}
   \sum_{t=-\infty}^\infty \Gamma_{\alpha \alpha} (t)

It is often convenient to think in terms of the normalized autocorrelation function
:math:`\rho(t) = \Gamma(t) / \Gamma(0)`, whose error can be computed
according to [2]_ (for the case :math:`\alpha=\beta`)

.. math::
   \mathrm{var} \rho (t) \simeq \frac{1}{N} \sum_{k=1}^\infty
   \big[ \rho(k+t) + \rho(k-t) - 2 \rho(k) \rho(t) \big]^2

Note that for a derived observable, :math:`F(\{a\})`, the same equations
hold with the only difference being in the fluctuations

.. math::
   \pi_F^{i,r} \equiv \sum_\alpha
   \frac{\partial F(\{a\})}{\partial a_\alpha} \bigg|_{a=\bar{\bar a}}
   \pi_\alpha^{i,r}

Automatic Window
----------------

In practice with a finite number of configurations
it is not possible to sum the autocorrelation function up to
infinity.
Moreover the noise of :math:`\Gamma(t)` grows
with :math:`t`. As a compromise an optimal summation window
:math:`W` is chosen such that the error estimate is reliable.

In fact, it is possible to demonstrate that :math:`\Gamma` is
expected to be a sum of exponentials whose coefficients
are observable dependent and whose arguments depend
instead on the Markov Transition Matrix.

The automatic windowing procedure introduced in [1]_
defines a criterion for the optimal window :math:`W`,
which is the first value for which :math:`g(W)` is negative


.. math::
   g(W) = \exp [-W / \tau_W] -  \tau_W /\sqrt{W N}

.. math::
   \tau_W = S_\tau \bigg[ \ln \bigg( \frac{\sum_W \rho(W) + 1}{\sum_W \rho(W)-1} \bigg)
   \bigg]^{-1}

A standard choice for the parameter :math:`S_\tau` is 1.5


Exponential tails
-----------------

To be implemented soon, from Ref [3]_.

References
----------

.. [1] U. Wolff "Monte Carlo errors with less errors"
       `Link to article <https://arxiv.org/abs/hep-lat/0306017>`__

.. [2] M. Luscher "Schwarz-preconditioned HMC algorithm for two-flavour lattice QCD"
       `Link to article <https://arxiv.org/abs/hep-lat/0409106>`__

.. [3] S. Schaefer, R. Sommer and F. Virotta "Investigating the critical slowing down of QCD simulations"
       `Link to article <https://arxiv.org/abs/0910.1465>`__
