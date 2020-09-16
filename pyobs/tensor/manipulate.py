#################################################################################
#
# manipulate.py: methods for the manipulation of the shape of observables
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
from pyobs.ndobs import obs
from pyobs.core.derobs import derobs
from pyobs.tensor.unary import unary_grad
from pyobs.core.utils import error_msg

__all__ = ['reshape','concatenate','transpose','sort','diag']

def reshape(x,new_shape):
    """
    Change the shape of the observable

    Parameters:
      x (obs) : observables to be reshaped
      new_shape (tuple): the new shape of the observable

    Returns:
      obs : reshaped observable

    Notes:
      This function acts exclusively on the mean
      value.
    """
    res = obs(x)
    res.shape = new_shape
    res.mean = numpy.reshape(x.mean, new_shape)
    return res

def concatenate(x,y,axis=0):
    """
    Join two arrays along an existing axis

    Parameters:
       x, y (obs): the two observable to concatenate
       axis (int, optional): the axis along which the 
             observables will be joined. Default is 0.
    
    Returns:
       obs : the concatenated observable
    
    Notes:
       If `x` and `y` contain information from separate
       ensembles, they are merged accordingly by keeping
       only the minimal amount of data in memory.
    """
    if x.size==0 and x.shape==[]:
        return obs(y)
    if y.size==0 and y.shape==[]:
        return obs(x)
    
    if len(x.shape)!=len(y.shape):
        error_msg(f'Incompatible dimensions between {x.shape} and {y.shape}')
    for d in range(len(x.shape)):
        if (d!=axis) and (x.shape[d]!=y.shape[d]):
            error_msg(f'Incompatible dimensions between {x.shape} and {y.shape} for axis={axis}')
    mean=numpy.concatenate((x.mean,y.mean),axis=axis)
    grads=[numpy.concatenate((numpy.eye(x.size),numpy.zeros((y.size,x.size))))]
    grads+=[numpy.concatenate((numpy.zeros((x.size,y.size)),numpy.eye(y.size)))]
    return derobs([x,y],mean,grads)

def transpose(x,axes=None):
    """
    Transpose a tensor along specific axes.
    For an array a with two axes, gives the matrix transpose.

    Parameters:
       x (obs): input observable
       axes (tuple or list of ints, optional): If specified, 
            it must be a tuple or list which contains a 
            permutation of [0,1,..,N-1] where N is the number of axes of `x`. 
            For more details read the documentation of `numpy.transpose`

    Returns:
       obs : the transposed observable
    """
    mean=numpy.transpose(x.mean,axes)
    grads=unary_grad(x.mean,lambda x:numpy.transpose(x,axes))
    return derobs([x],mean,[grads])

def sort(x,axis=-1):
    """
    Sort a tensor along a specific axis.
    
    Parameters:
       x (obs): input observable
       axis (int, optional): the axis which is sorted. Default is -1, the
       last axis.

    Returns:
       obs : the sorted observable
    """
    mean=numpy.sort(x.mean,axis)
    idx=numpy.argsort(x.mean,axis)
    grads=unary_grad(x.mean,lambda x: numpy.take_along_axis(x,idx,axis))
    return derobs([x],mean,[grads])

def diag(x):
    """
    Extract the diagonal of 2-D array or construct a diagonal matrix from a 1-D array
    
    Parameters:
       x (obs): input observable

    Returns:
       obs : the diagonally projected or extended observable
    """
    if len(x.shape)>2:
        error_msg(f'Unexpected matrix with shape {x.shape}; only 2-D arrays are supported')
    mean = numpy.diag(x.mean)
    grads = unary_grad(x.mean, lambda x:numpy.diag(x))
    return derobs([x],mean,[grads])
