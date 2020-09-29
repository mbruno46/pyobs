#################################################################################
#
# random.py: routines for the generation of synthetic autocorrelated data
# Copyright (C) 2020 Mattia Bruno
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
#################################################################################

import numpy

__all__ = ['acrand','acrandn']

def acrand(mu,sigma,tau,N):
    """
    Create synthetic autocorrelated Monte Carlo data

    Parameters:
       mu (int or float): the central value
       sigma (float): the square root of the variance of 
          the observable in absence of autocorrelations
       tau (float): the integrated autocorrelation time
       N (int): the number of configurations/measurements

    Returns:
       list : the synthetic data

    Note:
       The expected error from a proper autocorrelation analysis
       is the product of `sigma` with the square root of `tau` 
       divided by `N`.

    Examples:
       >>> data = pyobs.random.acrand(0.1234,0.0001,4.0,1000)
       >>> obs = pyobs.observable(desc='test-acrand')
       >>> obs.create('A',data)
       >>> print(obs)
       0.12341(26)

    """
    r=numpy.random.normal(0.,1.,N)
    if tau>0.0:
        f=numpy.exp(-1./tau)
    else:
        f=0.0
    ff=numpy.sqrt(1.-f*f)
    rn=numpy.zeros((N,))
    rn[0]=ff*r[0]
    for i in range(1,N):
        rn[i]=ff*r[i] + f*rn[i-1]
    return list(mu + sigma*rn)

def acrandn(mu,cov,tau,N):
    """
    Create synthetic correlated Monte Carlo 1-D data

    Parameters:
       mu (list of array): the central values of corresponding observable;
          a 1-D array is expected
       cov (array): the covariance matrix of the observable (in absence of 
          autocorrelations); if a 1-D array is passed, a diagonal covariance
          matrix is assumed
       tau (float): the integrated autocorrelation time
       N (int): the number of configurations/measurements

    Returns:
       numpy.ndarray : 2-D array with the synthetic data, such that each row 
          corresponds to a configuration
    """
    if len(mu)!=numpy.shape(cov)[0]: # pragma: no cover
        raise ValueError
    nf=len(mu)
    if tau>0:
        f=numpy.exp(-1./tau)
    else:
        f=0.0
    ff=numpy.sqrt(1.-f*f)
    
    r=numpy.reshape( numpy.random.normal(0.0,1.0,N*nf), (N,nf) )
    rn=numpy.zeros((N,nf))
    rn[0,:] = ff*r[0,:]
    for i in range(N):
        rn[i,:] = ff*r[i,:] + f*rn[i-1,:]
    
    if numpy.ndim(cov)==1:
        cov=numpy.diag(cov)
    [w,v] = numpy.linalg.eig(cov)

    Q = numpy.diag(numpy.sqrt(w)) @ v.T
    for i in range(N):
        rn[i,:] = rn[i,:] @ Q + numpy.array(mu)
    return rn
