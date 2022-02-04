#################################################################################
#
# unary.py: definitions of unary operations
# Copyright (C) 2020-2021 Mattia Bruno
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

__all__ = [
    "sum",
    "cumsum",
    "trace",
    "log",
    "exp",
    "sin",
    "cos",
    "tan",
    "arctan",
    "cosh",
    "sinh",
    "arccosh",
    "besselk",
]


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


##############################################


def __unary(x, f, df):
    new_mean = f(x.mean)
    aux = df(x.mean)
    g = pyobs.gradient(lambda xx: xx * aux, x.mean, gtype="diag")
    return pyobs.derobs([x], new_mean, [g])


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
    return __unary(x, numpy.log, numpy.reciprocal)


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
    return __unary(x, numpy.exp, numpy.exp)


def sin(x):
    """
    Return the sine element-wise.

    Parameters:
       x (obs): input observable

    Returns:
       obs : the sine of the input observable, element-wise.

    Examples:
       >>> sinA = pyobs.sin(obsA)
    """
    return __unary(x, numpy.sin, numpy.cos)


def cos(x):
    """
    Return the cosine element-wise.

    Parameters:
       x (obs): input observable

    Returns:
       obs : the cosine of the input observable, element-wise.

    Examples:
       >>> cosA = pyobs.cos(obsA)
    """
    return __unary(x, numpy.cos, lambda x: -numpy.sin(x))


def tan(x):
    """
    Return the tangent element-wise.

    Parameters:
       x (obs): input observable

    Returns:
       obs : the tangent of the input observable, element-wise.

    Examples:
       >>> tanA = pyobs.tan(obsA)
    """
    return __unary(x, numpy.tan, lambda x: 1 / numpy.cos(x) ** 2)


def arctan(x):
    """
    Return the arctangent element-wise.

    Parameters:
       x (obs): input observable

    Returns:
       obs : the arctangent of the input observable, element-wise.

    Examples:
       >>> arctanA = pyobs.arctan(obsA)
    """
    return __unary(x, numpy.arctan, lambda x: 1 / (1 + x**2))


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
    return __unary(x, numpy.cosh, numpy.sinh)


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
    return __unary(x, numpy.sinh, numpy.cosh)


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
    return __unary(x, numpy.arccosh, lambda x: 1.0 / numpy.sqrt(x**2 - 1))


#     new_mean = numpy.arccosh(x.mean)
#     aux = numpy.reciprocal(
#         numpy.sqrt(x.mean ** 2 - numpy.ones(x.shape))
#     )  # 1/sqrt(x^2-1)
#     g = pyobs.gradient(lambda x: x * aux, x.mean, gtype="diag")
#     return pyobs.derobs([x], new_mean, [g], description=f"arccosh of {x.description}")


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
    aux = scipy.special.kv(v - 1, x.mean) + scipy.special.kv(v + 1, x.mean)
    g = pyobs.gradient(lambda x: -0.5 * aux * x, x.mean, gtype="diag")
    return pyobs.derobs(
        [x], new_mean, [g], description=f"BesselK[{v}] of {x.description}"
    )
