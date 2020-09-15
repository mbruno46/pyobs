Good and Bad practices
======================

To maximize the efficiency of the library it is very 
important to `pack` the observables together in the
right way.


Let's assume to have two observables :math:`A` and
:math:`B` defined on ensemble E0 and E1 respectively.
Performing a merging operation where we define a new 
observable :math:`C_i` such that

.. math::
   C_0 = A \quad C_1 = B

would be very unproductive since the fluctuations of 
:math:`C_0` would be :math:`(\delta A, 0)` for ensemble
E0 and viceversa for E1. Thus a lot of zeros which would
slow down the analysis and occupy disk space. For this 
reason ``addcol`` and ``addrow`` can *only* be used
with identical ensemble content.

