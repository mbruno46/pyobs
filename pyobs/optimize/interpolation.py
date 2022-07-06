#################################################################################
#
# interpolation.py: interpolate a set of data points as observables
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
import pyobs

__all__ = ["interpolate"]


class interpolate:
    """
    Finds the polynomial with maximal degree passing exactly
    through the points given by `x` and `y`.

    Parameters:
       x (array): x coordinates of the points
       y (observable): y coordinates of the points

    When called on an array or float or observable, it evaluates
    the polynomial at the locations specified by the input parameters.

    Examples:
       >>> int0 = pyobs.interpolate(xax[0:5], yobs[0:5]) # interpolations with different degrees
       >>> int1 = pyobs.interpolate(xax[0:4], yobs[0:4])
       >>> int0([0.2, 0.3]) # evalutes the polynomial at those values
    """

    def __init__(self, x, y):
        """ """
        N = len(x)
        pyobs.assertion(
            len(y.shape) == 1,
            f"Unexpected observable with shape ${x.shape}; only vectors are supported",
        )
        pyobs.assertion(
            y.size == N,
            f"Unexpected observable with shape ${x.shape} not matching size of x",
        )
        M = numpy.array([[x[i] ** k for k in range(N)] for i in range(N)]).astype("f8")
        w = numpy.linalg.eig(M)[0]
        pyobs.assertion(
            abs(max(w) / min(w)) < 1e16, f"Singular matrix; decrease number of points"
        )
        Minv = numpy.linalg.inv(M)
        mean = Minv @ y.mean
        g = pyobs.gradient(lambda x: Minv @ x, y.mean)
        self.coeff = pyobs.derobs([y], mean, [g])
        self.k = N

    def __call__(self, x):
        """
        Evaluates the polynomial at the locations `x`.

        Parameters:
           x (array or float): location where the interpolated
               function should be evaluated

        Returns:
           observable: the evaluated function at `x`.
        """
        if type(x) is pyobs.observable:
            pyobs.assertion(
                len(x.shape) == 1,
                f"Unexpected observable with shape ${x.shape}; only vectors supported",
            )
            N = x.shape[0]
        else:
            N = len(x)
            x = numpy.array(x)
        res = pyobs.repeat(self.coeff[0], N)
        for i in range(1, self.k):
            res += pyobs.repeat(self.coeff[i], N) * (x**i)
        return res

    def solve(self, target, bracket):
        """
        Finds the location `x` where the polynomials equals a certain target value.
        """
        f = lambda x, a: numpy.sum([a[i] * x**i for i in range(self.k)]) - target
        dfx = lambda x, a: numpy.sum(
            [i * a[i] * x ** (i - 1) for i in range(1, self.k)]
        )
        dfa = lambda x, a: [x**i for i in range(self.k)]
        return pyobs.optimize.root_scalar(self.coeff, f, dfx, dfa, bracket=bracket)
