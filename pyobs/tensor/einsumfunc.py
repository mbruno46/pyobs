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

__all__ = ["einsum", "sum", "mean", "cumsum", "trace"]


def einsum(subscripts, *operands):
    """
    Evaluates the Einstein summation convention on the input observables or arrays.

    Please check the documentation of `numpy.einsum`
    """
    inps = []
    means = []

    for o in operands:
        if isinstance(o, pyobs.observable):
            inps.append(o)
            means.append(o.mean)
        else:
            means.append(numpy.array(o))

    grads = []
    for i in range(len(operands)):
        if not isinstance(operands[i], pyobs.observable):
            continue

        def f(x):
            return numpy.einsum(
                subscripts,
                *[means[j] for j in range(i)],
                x,
                *[means[j] for j in range(i + 1, len(operands))],
            )

        grads.append(
            pyobs.gradient(f, operands[i].mean)
        )  # non-optimized for large number of observables

    new_mean = numpy.einsum(subscripts, *means)
    return pyobs.derobs(inps, new_mean, grads)


def sum(x, axis=None):
    """
    Sum of observable elements over a given axis.

    Parameters:
       x (obs): observable with elements to sum
       axis (None or int or tuple of ints, optional): Axis or
           axes along which a sum is performed. The default, axis=None,
           will sum all elements of the input observable.

    Returns:
       obs: sum along the axis

    Examples:
       >>> import pyobs
       >>> pyobs.sum(a)
       >>> pyobs.sum(a,axis=0)
    """
    if axis is None:

        def f(a):
            return numpy.reshape(numpy.sum(a, axis=axis), (1,))

        t = f"sum all elements of {x.description}"
    else:

        def f(a):
            return numpy.sum(a, axis=axis)

        t = f"sum over axis {axis} of {x.description}"
    g = pyobs.gradient(f, x.mean)
    return pyobs.derobs([x], f(x.mean), [g], description=t)


def cumsum(x, axis=None):
    """
    Cumulative sum of the elements of an observable along a given axis.

    Parameters:
       x (obs): observable with elements to sum
       axis (int, optional): axis along which the cumulative
           sum is performed. The default, axis=None,
           will compute the cumulative sum over the flattened
           array.

    Returns:
       obs: cumulative sum along the axis

    Examples:
       >>> import pyobs
       >>> pyobs.sum(a)
       >>> pyobs.sum(a,axis=0)
    """

    def f(a):
        return numpy.cumsum(a, axis=axis)

    g = pyobs.gradient(f, x.mean)
    return pyobs.derobs([x], f(x.mean), [g])


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
    new_mean = numpy.trace(x.mean, offset, axis1, axis2)
    g = pyobs.gradient(lambda x: numpy.trace(x, offset, axis1, axis2), x.mean)
    return pyobs.derobs(
        [x],
        new_mean,
        [g],
        description=f"trace for axes ({axis1,axis2}) of {x.description}",
    )

def mean(x, axis=None):
    """
    Return the arithmetic mean along the specified axis.

    Parameters:
       x (obs): input observable
       axis (int or tuple of ints, optional): axis or axes along which the 
           means are computed

    Returns:
       obs: the mean of the observable along the given axis.

    Examples:
       >>> mean = pyobs.mean(obs)
    """
    norm = x.size if axis is None else numpy.prod(x.shape[axis])
    return pyobs.sum(x, axis=axis) / norm

