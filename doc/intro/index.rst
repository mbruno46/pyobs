Introduction
============

Below we give a brief introduction on the notion of autocorrelations in
Markov Chains, with Lattice QCD simulations and observables in mind, 
and we set the notation for the rest of the documentation accordingly.

Definitions
-----------

:Lattice: a discretized four-dimensional torus of size :math:`L^3 \times T`
:Field Configurations: value taken by field on every position :math:`x`
:Replica: a stream of (gauge) field configurations generated from a single
  Markov Chain,
:Ensemble: a set of replicas generated from Markov Chains differing only in the
  random seed
:Correlator: the product of several fields (properly contracted) located
  at different positions
:Expectation value: the average over a single ensemble
:Observables: quantities that are extracted from expectation values of 
  n-point functions, aka correlators.


Methods
-------

.. toctree::
   :maxdepth: 1

   gamma
   jknife   
   

