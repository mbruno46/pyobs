#################################################################################
#
# unary.py: definitions of unary operations
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
import scipy
import pyobs

__all__ = ['sum','trace','log','exp',
           'cosh','sinh','arccosh','besselk']

def sum(x,axis=None):
    """
    Sum of array elements over a given axis.
    
    Parameters:
       x (obs): array with elements to sum
       axis (None or int or tuple of ints, optional): Axis or 
           axes along which a sum is performed. The default, axis=None, 
           will sum all elements of the input array.
    
    Returns:
       obs: sum along the axis
    
    Examples:
       >>> import pyobs
       >>> pyobs.sum(a)
       >>> pyobs.sum(a,axis=0)
    """
    if axis is None:
        f=lambda a: numpy.reshape(numpy.sum(a,axis=axis),(1,))
        t=f'sum all elements of {x.description}'
    else:
        f=lambda a: numpy.sum(a,axis=axis)
        t=f'sum over axis {axis} of {x.description}'
    g=x.gradient(f)
    return pyobs.derobs([x],f(x.mean),[g],description=t)

def trace(x, offset=0, axis1=0, axis2=1):
    """
    Return the sum along diagonals of the array.
    
    Parameters:
       x (obs): observable whose diagonal elements are taken
       offset (int, optional): offset of the diagonal from the main diagonal; 
           can be both positive and negative. Defaults to 0.
       axis1, axis2 (int, optional): axes to be used as the first and second 
           axis of the 2-D sub-arrays whose diagonals are taken; 
           defaults are the first two axes of `x`.
    
    Returns:
       obs : the sum of the diagonal elements
       
    Notes:
       If `x` is 2-D, the sum along its diagonal with the given offset is returned, 
       i.e., the sum of elements `x[i,i+offset]` for all i. If `x` has more than 
       two dimensions, then the axes specified by `axis1` and `axis2` are used to 
       determine the 2-D sub-arrays whose traces are returned. The shape of the 
       resulting array is the same as that of a with `axis1` and `axis2` removed.
    
    Examples:
       >>> tr = pyobs.trace(mat)
    """
    new_mean=numpy.trace(x.mean,offset,axis1,axis2)
    g=x.gradient(lambda x:numpy.trace(x,offset,axis1,axis2))
    return pyobs.derobs([x],new_mean,[g],description=f'trace for axes ({axis1,axis2}) of {x.description}')
    
#def sin(x):
#    g=x.gradient(lambda x:x*numpy.cos(x.mean))
#    return pyobs.derobs([self],numpy.sin(self.mean),[g])
#
#def cos(x):
#    g=unary(x.mean,lambda x:-x*numpy.sin(x.mean))
#    return pyobs.derobs([self],numpy.cos(self.mean),[g])
#


def log(x):
    """
    Return the Natural logarithm element-wise.
    
    Parameters:
       x (obs): input observable
    
    Returns:
       obs : the logarithm of the input observable, element-wise.
    
    Examples:
       >>> logA = pyobs.log(obsA)
    """
    new_mean = numpy.log(x.mean)
    aux = numpy.reciprocal(x.mean)
    g=x.gradient(lambda x: x*aux)
    return pyobs.derobs([x],new_mean,[g],description=f'log of {x.description}')


def exp(x):
    """
    Return the exponential element-wise.
    
    Parameters:
       x (obs): input observable
    
    Returns:
       obs : the exponential of the input observable, element-wise.
    
    Examples:
       >>> expA = pyobs.exp(obsA)
    """
    new_mean = numpy.exp(x.mean)
    g=x.gradient(lambda x: x*new_mean)
    return pyobs.derobs([x],new_mean,[g],description=f'exp of {x.description}')


def cosh(x):
    """
    Return the Hyperbolic cosine element-wise.
    
    Parameters:
       x (obs): input observable
    
    Returns:
       obs : the hyperbolic cosine of the input observable, element-wise.
    
    Examples:
       >>> B = pyobs.cosh(obsA)
    """
    new_mean = numpy.cosh(x.mean)
    aux = numpy.sinh(x.mean)
    g=x.gradient(lambda x: x*aux)
    return pyobs.derobs([x],new_mean,[g],description=f'cosh of {x.description}')


def sinh(x):
    """
    Return the Hyperbolic sine element-wise.
    
    Parameters:
       x (obs): input observable
    
    Returns:
       obs : the hyperbolic sine of the input observable, element-wise.
    
    Examples:
       >>> B = pyobs.sinh(obsA)
    """
    new_mean = numpy.sinh(x.mean)
    aux = numpy.cosh(x.mean)
    g=x.gradient(lambda x: x*aux)
    return pyobs.derobs([x],new_mean,[g],description=f'sinh of {x.description}')
    
def arccosh(x):
    """
    Return the inverse Hyperbolic cosine element-wise.
    
    Parameters:
       x (obs): input observable
    
    Returns:
       obs : the inverse hyperbolic cosine of the input observable, element-wise.
    
    Examples:
       >>> B = pyobs.arccosh(obsA)
    """
    new_mean = numpy.arccosh(x.mean) 
    aux = numpy.reciprocal(numpy.sqrt(x.mean**2-numpy.ones(x.shape)))  # 1/sqrt(x^2-1)
    g=x.gradient(lambda x: x*aux)
    return pyobs.derobs([x],new_mean,[g],description=f'arccosh of {x.description}')

def besselk(v, x):
    """
    Modified Bessel function of the second kind of real order `v`, element-wise.
    
    Parameters:
       v (float): order of the Bessel function
       x (obs): real observable where to evaluate the Bessel function
    
    Returns:
       obs : the modified bessel function computed for the input observable
    """
    new_mean = scipy.special.kv(v, x.mean)
    aux = scipy.special.kv(v-1,x.mean) + scipy.special.kv(v+1,x.mean)
    g=x.gradient(lambda x: -0.5*aux*x)
    return pyobs.derobs([x],new_mean,[g],description=f'BesselK[{v}] of {x.description}')
