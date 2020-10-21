#################################################################################
#
# einsumfunc.py: extension of numpy.einsum to observables
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

__all__ = ['einsum']

def einsum(subscripts, *operands):
    """
    Evaluates the Einstein summation convention on the input observables or arrays.
    
    Please check the documentation of `numpy.einsum`
    """
    inps = []
    means = []
    
    for o in operands:
        if isinstance(o,pyobs.observable):
            inps.append(o)
            means.append(o.mean)
        else:
            means.append(numpy.array(o))

    grads = []
    for i in range(len(operands)):
        if not isinstance(operands[i],pyobs.observable):
            continue

        f = lambda x: numpy.einsum(subscripts, *[means[j] for j in range(i)], x, *[means[j] for j in range(i+1,len(operands))])
        grads.append(pyobs.gradient(f, operands[i].mean)) # non-optimized for large number of observables
    
    new_mean = numpy.einsum(subscripts, *means)
    return pyobs.derobs(inps, new_mean, grads)