#################################################################################
#
# linalg.py: definitions of linear algebra methods
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
import pyobs

__all__ = ['inv','eig','eigLR','matrix_power']

def inv(x):
    """
    Compute the inverse of a square matrix

    Parameters:
       x (obs): Matrix to be inverted
    
    Returns:
       obs: (Multiplicative) inverse of `x`
    
    Examples:
       >>> from pyobs.linalg import inv
       >>> a = pyobs.observable()
       >>> a.create('A',data,shape=(2,2))
       >>> ainv = pyobs.inv(a)
    
    Notes:
       If the number of dimensions is bigger than 2, 
       `x` is treated as a stack of matrices residing 
       in the last two indexes and broadcast accordingly.
    """
    if (x.shape[-2]!=x.shape[-1]): # pragma: no cover
        raise pyobs.PyobsError(f'Unexpected matrix for inverse with shape={x.shape}')
    mean=numpy.linalg.inv(x.mean)
    # V Vinv = 1,   dV Vinv + V dVinv = 0 ,  dVinv = - Vinv dV Vinv
    g=pyobs.gradient( lambda x: - mean @ x @ mean, x.shape)
    return pyobs.derobs([x],mean,[g])

def eig(x):
    """
    Computes the eigenvalues and eigenvectors of a square matrix observable.
    The central values are computed using the `numpy.linalg.eig` routine.

    Parameters:
       x (obs): a symmetric square matrix (observable) with dimensions `NxN`

    Returns:
       list of obs: a vector observable with the eigenvalues and a matrix observable whose columns correspond to the eigenvectors

    Notes:
       The error on the eigenvectors is based on the assumption that the input
       matrix is symmetric. If this not respected, the returned eigenvectors will
       have under or over-estimated errors.
    
    Examples:
       >>> [w,v] = pyobs.linalg.eig(mat)
       >>> for i in range(N):
       >>>     # check eigenvalue equation  
       >>>     print(mat @ v[:,i] - v[:,i] * w[i])
    """
    if len(x.shape)>2: # pragma: no cover
        raise pyobs.PyobsError(f'Unexpected matrix with shape {x.shape}; only 2-D arrays are supported')
    if numpy.any(numpy.fabs(x.mean/x.mean.T-1.0)>1e-10): # pragma: no cover
        raise pyobs.PyobsError(f'Unexpected non-symmetric matrix: user eigLR')
    
    [w,v] = numpy.linalg.eig(x.mean)
    
    # d l_n = (v_n, dA v_n)
    gw=pyobs.gradient( lambda x: numpy.diag(v.T @ x @ v), x.shape)

    # d v_n = sum_{m \neq n} (w_m, dA v_n) / (l_n - l_m) w_m
    def gradv(y):
        tmp = v.T @ y @ v
        gv = numpy.zeros(x.shape)
        for n in range(x.shape[0]):
            for m in range(x.shape[1]):
                if n!=m:
                    gv[:,n] += tmp[m,n]/(w[n]-w[m])*v[:,m]
        return gv

    gv=pyobs.gradient(gradv, x.shape)
    return [pyobs.derobs([x],w,[gw]), pyobs.derobs([x],v,[gv])]

def eigLR(x):
    """
    Computes the eigenvalues and the left and right eigenvectors of a 
    square matrix observable. The central values are computed using 
    the `numpy.linalg.eig` routine.

    Parameters:
       x (obs): a square matrix (observable) with dimensions `NxN`; 

    Returns:
       list of obs: a vector observable with the eigenvalues and two 
       matrix observables whose columns correspond to the right and 
       left eigenvectors respectively.

    Notes:
       This input matrix is not expected to be symmetric. If it is the 
       usage of `eig` is recommended for better performance.
    
    Examples:
       >>> [l,v,w] = pyobs.linalg.eigLR(mat)
       >>> for i in range(N):
       >>>     # check eigenvalue equation  
       >>>     print(mat @ v[:,i] - v[:,i] * l[i])
       >>>     print(w[:,i] @ mat - w[:,i] * l[i])
    """
    if len(x.shape)>2: # pragma: no cover
        raise pyobs.PyobsError(f'Unexpected matrix with shape {x.shape}; only 2-D arrays are supported')
   
    # left and right eigenvectors
    [l,v] = numpy.linalg.eig(x.mean)
    [l,w] = numpy.linalg.eig(x.mean.T)
    
    # d l_n = (w_n, dA v_n) / (w_n, v_n)
    gl=pyobs.gradient( lambda x: numpy.diag(w.T @ x @ v)/numpy.diag(w.T @ v), x.shape)

    # d v_n = sum_{m \neq n} (w_m, dA v_n) / (l_n - l_m) w_m
    def gradv(y):
        tmp = w.T @ y @ v
        gv = numpy.zeros(x.shape)
        for n in range(x.shape[0]):
            for m in range(x.shape[1]):
                if n!=m:
                    gv[:,n] += tmp[m,n]/(l[n]-l[m])*w[:,m]
        return gv
    gv=pyobs.gradient( gradv, x.shape)
    
    # d w_n = sum_{m \neq n} (v_m, dA^T w_n) / (l_n - l_m) v_m
    def gradw(y):
        tmp = v.T @ y.T @ w
        gw = numpy.zeros(x.shape)
        for n in range(x.shape[0]):
            for m in range(x.shape[1]):
                if n!=m:
                    gw[:,n] += tmp[m,n]/(l[n]-l[m])*v[:,m]
        return gw
    
    gw=pyobs.gradient( gradw, x.shape)
    return [pyobs.derobs([x],l,[gl]), pyobs.derobs([x],v,[gv]), pyobs.derobs([x],w,[gw])]

def matrix_power(x,a):
    """
    Raises a square symmetric matrix to any non-integer power `a`.

    Parameters:
       x (obs): a symmetric square matrix (observable) with dimensions `NxN`
       a (float): the power
       
    Returns:
       obs: the matrix raised to the power `a`

    Notes:
       The calculation is based on the eigenvalue decomposition of the
       matrix.
    
    Examples:
       >>> matsq = pyobs.linalg.matrix_power(mat, 2) # identical to mat @ mat
       >>> matsqrt = pyobs.linalg.matrix_power(mat, -0.5)
       >>> matsqrt @ mat @ matsqrt # return the identity
    """
    [w,v] = eig(x)
    return v @ pyobs.diag(w**a) @ pyobs.transpose(v)
    