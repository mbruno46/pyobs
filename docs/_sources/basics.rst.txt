
Basics
======


In this tutorial we show on observables can be manipulated 
and combined into new ones.

Effective mass
##############

Let's assume to have a correlator measured on 32 time slices 
with periodic boundary conditions, which has already been saved 
as an `observa` class::

   >>> tmax=32
   >>> corr = observa()
   >>> corr.load('correlator.pyobs.gz')

The effective mass is defined as 

.. math::

   m_\mathrm{eff} (t) = \mathrm{arccosh} \bigg( \frac{ C(t-1) + C(t+1) }{2 C(t) } \bigg)

This library has been designed such that the previous formula can be mapped
into a single line of code that reads exactly the same::

   >>> meff = arccosh( (corr[0,0:tmax-2]+corr[0,2:tmax])/corr[0,1:tmax-1]*0.5 )

At this point, the user defines a procedure to exclude excited states
and define a plateau where the effective mass can be averaged::

   >>> plateau = [10,20]
   >>> norm = 1.0/10.
   >>> mass = meff[0,plateau[0]:plateau[1]].sum(1) * norm
   >>> print 'mass = ', mass
   mass =      0.15044(88)
   >>> mass.save('pion_mass')

Combining observables
#####################

Now that we have computed the pion mass in lattice units :math:`a m_\pi` 
we need a scale. Let suppose that the user has already compute the 
flow parameter :math:`t_0/a^2` (see Ref [1]_ for more details)
and that he is interested in the dimensionless quantity

.. math::
   \phi = 8 t_0 m_\pi^2

The observable :math:`\phi` can be computed again in a single line::

   >>> # t0 and mpi already loaded/defined
   >>> print 't0 = ', t0
   t0 =      5.1652(44)
   >>> print 'mpi = ', mpi
   mpi =      0.1343(19)
   >>> phi = 8. * t0 * (mpi**2)
   >>> print 'phi = ', phi
   phi =      0.746(21)

.. [1] M. Luscher "..." 
