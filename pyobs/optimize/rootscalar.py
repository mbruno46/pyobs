#################################################################################
#
# root_scalar.py: finds the roots of a scalar function
# Copyright (C) 2021 Mattia Bruno
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
from scipy.optimize import root_scalar as rs

import pyobs

__all__ = ["root_scalar"]

# f(x,a_i) = 0 @ x=x0
# df/dx dx + sum_i df/da_i da_i = 0
# delta x = sum_i [df/da_i] / [df/dx] * delta a_i evaluated at x=x0


def root_scalar(a, f, dfx, dfa, x0=None, method=None, bracket=None):
    """
    Find a root of a scalar function.

    Find a root of `f(x,a)` depending on a single coordinate `x`
    and on a set of observables `a_i`, e.g. `f(x,a) = a[0] + a[1]*x`. The program
    returns the root `x0` fulfilling `f(x0,a) = 0` as an observable.

    Parameters:
       a (obs): observable with the parameters `a_i`. It must be a vector.
       f (callable): the function to find a root of
       dfx (callable): the derivative of `f` w.r.t. `x`
       dfa (callable): a list with the derivatives of `f` w.r.t. the
           parameters `a_i`
       x0 (float, optional): initial guess
       method (str, optional): the type of solver. For more details check the
           documentation of `scipy.optimize.root_scalar`.
       bracket (list of 2 floats, optional): an interval bracketing a root.

    Returns:
       obs: observable with the root `x0`.

    Note:
       while `x0`, `method` and `bracket` are individually optional at least
       one of them must be supplied to tell the minimizer which method to use.

    Examples:
       >>> a = pyobs.observable()
       >>> a.create('A',data,shape=(1,2))
       >>> f = lambda x,a: a[0] + a[1]*x
       >>> dfx = lambda x,a: a[1]
       >>> dfa = lambda x,a: [1, x]
       >>> pyobs.optimize.root_scalar(f, dfx, dfa)
    """
    pyobs.assertion(
        len(a.shape) == 1,
        f"Unexpected observable with shape ${a.shape}; only vectors are supported",
    )

    res = rs(lambda x: f(x, a.mean), x0=x0, bracket=bracket, method=method)
    mean = numpy.reshape(res.root, (1,))
    _g = numpy.array(dfa(res.root, a.mean)) / dfx(res.root, a.mean)

    g = pyobs.gradient(lambda x: _g @ x, a.mean)
    return pyobs.derobs([a], mean, [g])
