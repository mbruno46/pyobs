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
from pyobs.core.utils import error_msg
from pyobs.core.derobs import derobs
from pyobs.tensor.unary import unary_grad
from pyobs.tensor.manipulate import diag, transpose

def inv(x):
    """
    Compute the inverse of a square matrix

    Parameters:
       x (obs): Matrix to be inverted
    
    Returns:
       obs: (Multiplicative) inverse of `x`
    
    Examples:
       >>> from pyobs.linalg import inv
       >>> a = pyobs.obs()
       >>> a.create('A',data,dims=(2,2))
       >>> ainv = pyobs.inv(a)
    
    Notes:
       If the number of dimensions is bigger than 2, 
       `x` is treated as a stack of matrices residing 
       in the last two indexes and broadcast accordingly.
    """
    if (x.dims[-2]!=x.dims[-1]):
        error_msg(f'Unexpected matrix for inverse with dims={x.dims}')
    mean=numpy.linalg.inv(x.mean)
    # V Vinv = 1,   dV Vinv + V dVinv = 0 ,  dVinv = - Vinv dV Vinv
    g=unary_grad(x.mean, lambda x: - mean @ x @ mean)
    return derobs([x],mean,[g])

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
    if len(x.dims)>2:
        error_msg(f'Unexpected matrix with dims {x.dims}; only 2-D arrays are supported')
    if numpy.any(numpy.fabs(x.mean/x.mean.T-1.0)>1e-10):
        error_msg(f'Unexpected non-symmetric matrix: user eigLR')
    
    [w,v] = numpy.linalg.eig(x.mean)
    
    # d l_n = (v_n, dA v_n)
    gw=unary_grad(x.mean, lambda x: numpy.diag(v.T @ x @ v))

    # d v_n = sum_{m \neq n} (w_m, dA v_n) / (l_n - l_m) w_m
    def gradv(y):
        tmp = v.T @ y @ v
        gv = numpy.zeros(x.dims)
        for n in range(x.dims[0]):
            for m in range(x.dims[1]):
                if n!=m:
                    gv[:,n] += tmp[m,n]/(w[n]-w[m])*v[:,m]
        return gv

    gv=unary_grad(x.mean, gradv)
    return [derobs([x],w,[gw]), derobs([x],v,[gv])]

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
    if len(x.dims)>2:
        error_msg(f'Unexpected matrix with dims {x.dims}; only 2-D arrays are supported')
   
    # left and right eigenvectors
    [l,v] = numpy.linalg.eig(x.mean)
    [l,w] = numpy.linalg.eig(x.mean.T)
    
    # d l_n = (w_n, dA v_n) / (w_n, v_n)
    gl=unary_grad(x.mean, lambda x: numpy.diag(w.T @ x @ v)/numpy.diag(w.T @ v))

    # d v_n = sum_{m \neq n} (w_m, dA v_n) / (l_n - l_m) w_m
    def gradv(y):
        tmp = w.T @ y @ v
        gv = numpy.zeros(x.dims)
        for n in range(x.dims[0]):
            for m in range(x.dims[1]):
                if n!=m:
                    gv[:,n] += tmp[m,n]/(l[n]-l[m])*w[:,m]
        return gv
    gv=unary_grad(x.mean, gradv)
    
    # d w_n = sum_{m \neq n} (v_m, dA^T w_n) / (l_n - l_m) v_m
    def gradw(y):
        tmp = v.T @ y.T @ w
        gw = numpy.zeros(x.dims)
        for n in range(x.dims[0]):
            for m in range(x.dims[1]):
                if n!=m:
                    gw[:,n] += tmp[m,n]/(l[n]-l[m])*v[:,m]
        return gw
    
    gw=unary_grad(x.mean, gradw)
    return [derobs([x],l,[gl]), derobs([x],v,[gv]), derobs([x],w,[gw])]

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
    return v @ diag(w**a) @ transpose(v)
    
