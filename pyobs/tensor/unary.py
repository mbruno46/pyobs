#################################################################################
#
# unary.py: definitions of unary operations
# Copyright (C) 2020-2023 Mattia Bruno
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
import pyobs
import numpy


def __unary(x, f, df):
    new_mean = f(x.mean)
    aux = df(x.mean)
    g = pyobs.gradient(lambda xx: xx * aux, x.mean, gtype="diag")
    return pyobs.derobs([x], new_mean, [g])


__all__ = [
    "log",
    "exp",
    "sin",
    "arcsin",
    "cos",
    "arccos",
    "tan",
    "arctan",
    "cosh",
    "arccosh",
    "sinh",
    "arcsinh",
]


def log(x):
    """
    Return the Natural logarithm element-wise.

    Parameters:
       x (obs): input observable

    Returns:
       obs : the logarithm of the input observable, element-wise.

    Examples:
       >>> y = pyobs.log(x)
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
       >>> y = pyobs.exp(x)
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
       >>> y = pyobs.sin(x)
    """
    return __unary(x, numpy.sin, numpy.cos)


def arcsin(x):
    """
    Return the Inverse sine element-wise.

    Parameters:
       x (obs): input observable

    Returns:
       obs : the arcsine of the input observable, element-wise.

    Examples:
       >>> y = pyobs.arcsin(x)
    """
    return __unary(x, numpy.arcsin, lambda x: (1 - x * x) ** (-0.5))


def cos(x):
    """
    Return the cosine element-wise.

    Parameters:
       x (obs): input observable

    Returns:
       obs : the cosine of the input observable, element-wise.

    Examples:
       >>> y = pyobs.cos(x)
    """
    return __unary(x, numpy.cos, lambda x: -numpy.sin(x))


def arccos(x):
    """
    Return the Inverse sine element-wise.

    Parameters:
       x (obs): input observable

    Returns:
       obs : the arcsine of the input observable, element-wise.

    Examples:
       >>> y = pyobs.arccos(x)
    """
    return __unary(x, numpy.arcsin, lambda x: -((1 - x * x) ** (-0.5)))


def tan(x):
    """
    Return the tangent element-wise.

    Parameters:
       x (obs): input observable

    Returns:
       obs : the tangent of the input observable, element-wise.

    Examples:
       >>> y = pyobs.tan(x)
    """
    return __unary(x, numpy.tan, lambda x: 1 / numpy.cos(x) ** 2)


def arctan(x):
    """
    Return the inverser tangent element-wise.

    Parameters:
       x (obs): input observable

    Returns:
       obs : the arctangent of the input observable, element-wise.

    Examples:
       >>> y = pyobs.arctan(x)
    """
    return __unary(x, numpy.arctan, lambda x: 1 / (1 + x * x))


def cosh(x):
    """
    Return the Hyperbolic cosine element-wise.

    Parameters:
       x (obs): input observable

    Returns:
       obs : the hyperbolic cosine of the input observable, element-wise.

    Examples:
       >>> y = pyobs.cosh(x)
    """
    return __unary(x, numpy.cosh, numpy.sinh)


def arccosh(x):
    """
    Return the inverse Hyperbolic cosine element-wise.

    Parameters:
       x (obs): input observable

    Returns:
       obs : the inverse hyperbolic cosine of the input observable, element-wise.

    Examples:
       >>> y = pyobs.arccosh(x)
    """
    return __unary(x, numpy.arccosh, lambda x: (x * x - 1) ** (-0.5))


def sinh(x):
    """
    Return the Hyperbolic sine element-wise.

    Parameters:
       x (obs): input observable

    Returns:
       obs : the hyperbolic sine of the input observable, element-wise.

    Examples:
       >>> y = pyobs.sinh(x)
    """
    return __unary(x, numpy.sinh, numpy.cosh)


def arcsinh(x):
    """
    Return the inverse Hyperbolic sine element-wise.

    Parameters:
       x (obs): input observable

    Returns:
       obs : the inverse hyperbolic sine of the input observable, element-wise.

    Examples:
       >>> y = pyobs.arcsinh(x)
    """
    return __unary(x, numpy.arcsinh, lambda x: (x * x + 1) ** (-0.5))
