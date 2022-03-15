#################################################################################
#
# zeta00.py: fast implementation of Luescher zeta00 function
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
from scipy.integrate import quad
from scipy.optimize import root_scalar

nuk = numpy.loadtxt(__file__.replace("zeta00.py", "nuk.dat"))

__all__ = ["Zeta00"]


class Zeta00:
    def __init__(self, mu=None, D=3):
        if mu is None:
            self.mu = lambda x: 1 / (x + 1) ** (1.0 / x**0.25)
        else:
            self.mu = lambda x: mu
        self.eps = 1e-12
        self.N = 4
        self.Nmax = len(nuk)
        self.D = D
        self.kvals = []
        for k in range(self.Nmax):
            if nuk[k, D] != 0:
                self.kvals.append(k)

    def nth_state_exists(self, n):
        return n in self.kvals

    def I1(self, qsq, der):
        mu = self.mu(qsq)
        coeff = []
        if der == 0:
            coeff += [1]
        elif der == 1:
            coeff.extend([1, 1])
        elif der == 2:
            coeff.extend([2, 2, 1])

        def summand(qsq, k):
            h = k - qsq
            num = numpy.sum([c * (h * mu) ** i for i, c in enumerate(coeff)])
            return nuk[k, self.D] * numpy.exp(-mu * h) * num / h ** (der + 1)

        # print('ss ', summand(qsq,0), self.__sum(lambda k: summand(qsq,k), summand(qsq,1), 2))
        return self.__sum(lambda k: summand(qsq, k), 0) / numpy.sqrt(4 * numpy.pi)
        # return self.__sum(lambda k: summand(qsq,k), summand(qsq,1), 2) / numpy.sqrt(4*numpy.pi)
        # return summand(qsq, 0)

    def __sum(self, f, ik0):
        res = 0.0
        for k in self.kvals[ik0:]:
            delta = f(k)
            res += delta
            if abs(delta / res) < self.eps:
                return res
        print("Overflow encoutered")
        print(f"res = {res}, delta = {delta}")
        return res

    def I0(self, qsq, der):
        mu = self.mu(qsq)
        exp = der - self.D / 2
        a = quad(lambda t: t**exp * (numpy.exp(qsq * t) - 1), 0, mu)[0]

        def integral(qsq, k):
            return (
                nuk[k, self.D]
                * quad(
                    lambda t: t**exp * numpy.exp(qsq * t - numpy.pi**2 * k / t),
                    0,
                    mu,
                )[0]
            )

        b = self.__sum(lambda k: integral(qsq, k), 1)
        # print(a,b,self.D)
        exp = 2 * (der + 1) - self.D
        # return numpy.sqrt(numpy.pi)**(self.D-1)/2 * (a + b + 2.0*numpy.sqrt(mu)**exp/exp)
        # return numpy.sqrt(numpy.pi)**(self.D-1)/2 * b
        return (
            0.5
            * numpy.sqrt(numpy.pi) ** (self.D - 1)
            * (a + b + 2.0 * numpy.sqrt(mu) ** exp / exp)
        )

    def __call__(self, qsq, der=0):
        if float(qsq).is_integer():
            if nuk[int(qsq), self.D] != 0:
                return numpy.inf
        return self.I1(qsq, der) + self.I0(qsq, der)

    def solve(self, g, k0):
        for k in range(k0 + 1, self.Nmax):
            if nuk[k, self.D] != 0:
                bracket = [k0 + 1e-10, k - 1e-10]
                break
        res = root_scalar(
            lambda x: self(x) - g(x), bracket=bracket, x0=numpy.mean(bracket)
        )
        return res.root if res.converged else None
