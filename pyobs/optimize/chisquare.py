#################################################################################
#
# mfit.py: definition of mfit class and its properties
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

import pyobs
from .minimize import lm

__all__ = ["mfit"]


class chisquare:
    def __init__(self, x, W, f, df, v):
        self.v = v
        if numpy.ndim(x) == 2:
            (self.n, self.nx) = numpy.shape(x)
        elif numpy.ndim(x) == 1:
            self.n = len(x)
            self.nx = 1
        else:  # pragma: no cover
            raise pyobs.PyobsError("Unexpected x")
        pyobs.assertion(len(self.v) == self.nx, "Unexpected x")
        self.x = numpy.reshape(x, (self.n, self.nx))
        self.W = numpy.array(W)
        self.f = f
        self.df = df
        pyobs.assertion(
            f.__code__.co_varnames == df.__code__.co_varnames,
            "Unexpected f and df: varnames do not match",
        )
        self.pars = []
        for vn in f.__code__.co_varnames:
            if vn not in self.v:
                self.pars.append(vn)
        self.np = len(self.pars)
        self.e = numpy.zeros((self.n,), dtype=numpy.float64)
        self.de = numpy.zeros((self.n, self.np), dtype=numpy.float64)
        self.p = [0.0] * self.np

    def set_pars(self, pdict, p0):
        n = 0
        for pn in self.pars:
            self.p[n] = p0[pdict[pn]]
            n += 1

    def __call__(self, y):
        pyobs.assertion(
            len(y) == self.n,
            f"Unexpected length of observable {len(y)} w.r.t. x-axis {self.n}",
        )
        pyobs.assertion(
            numpy.shape(self.W)[0] == self.n,
            f"Unexpected size of W matrix {numpy.shape(self.W)} w.r.t. x-axis {self.n}",
        )
        for i in range(self.n):
            self.e[i] = self.f(*self.x[i, :], *self.p) - y[i]
        return self.e @ self.W @ self.e

    def csq(self):
        return self.e @ self.W @ self.e

    def grad(self, y, pdict):
        res = numpy.zeros((len(pdict),))
        for i in range(self.n):
            self.de[i, :] = numpy.array(
                self.df(*self.x[i, :], *self.p)
            )  # N x Na matrix
        tmp = 2.0 * (self.e @ self.W @ self.de).T
        for pn in self.pars:
            i = pdict[pn]
            j = self.pars.index(pn)
            res[i] = tmp[j]
        return res

    def hess(self, y, pdict):
        res = numpy.zeros((len(pdict), len(pdict)))
        tmp = 2.0 * (self.de.T @ self.W @ self.de)
        for pn0 in self.pars:
            a = pdict[pn0]
            i = self.pars.index(pn0)
            for pn1 in self.pars:
                b = pdict[pn1]
                j = self.pars.index(pn1)
                res[a, b] = tmp[i, j]
        return res

    def gvec(self, pdict, p0):
        res = numpy.zeros((len(pdict), self.n))

        self.set_pars(pdict, p0)
        g = numpy.array(
            [self.df(*self.x[i, :], *self.p) for i in range(self.n)]
        )  # N x Na matrix
        g = self.W @ g
        for pn in self.pars:
            a = pdict[pn]
            i = self.pars.index(pn)
            res[a, :] = g[:, i]
        return res

    def Hmat(self, pdict, p0):
        H = numpy.zeros((len(pdict), len(pdict)))

        self.set_pars(pdict, p0)
        g = numpy.array(
            [self.df(*self.x[i, :], *self.p) for i in range(self.n)]
        )  # N x Na matrix
        _H = g.T @ self.W @ g  # Na x Na matrix

        for pn1 in self.pars:
            a = pdict[pn1]
            i = self.pars.index(pn1)
            for pn2 in self.pars:
                b = pdict[pn2]
                j = self.pars.index(pn2)
                H[a, b] = _H[i, j]
        return H

    def eval(self, x, pdict, p0):
        if numpy.ndim(x) == 2:
            (n, nx) = numpy.shape(x)
        elif numpy.ndim(x) == 1:
            n = len(x)
            nx = 1
        elif numpy.ndim(x) == 0:
            n = 1
            nx = 1
        else:  # pragma: no cover
            raise pyobs.PyobsError("Unexpected x")
        x = numpy.reshape(x, (n, nx))
        res = numpy.zeros((n, len(pdict)))

        self.set_pars(pdict, p0)
        new_mean = numpy.array([self.f(*x[i, :], *self.p) for i in range(n)])
        g = numpy.array([self.df(*x[i, :], *self.p) for i in range(n)])  # N x Na
        for pn in self.pars:
            a = pdict[pn]
            i = self.pars.index(pn)
            res[:, a] = g[:, i]
        return [new_mean, res]


class mfit:
    r"""
    Class to perform fits to multiple observables, via the minimization
    of the :math:`\chi^2` function

    .. math:: \chi^2 = r^T W r \quad\,,\quad r_i = y_i - \phi(\{p\},\{x_i\})

    with :math:`\phi` the model function, :math:`p_{\\alpha}` the model
    parameters (:math:`\\alpha=1,\dots,N_{\\alpha}`) and :math:`x_i^\mu`
    the kinematic coordinates (:math:`i=1,\dots,N` corresponds to the
    i-th kinematical point, and :math:`\mu` labels the dimensions).
    The matrix :math:`W` defines the metric of the :math:`\chi^2` function.

    The ideal choice for the matrix :math:`W` is the inverse of the
    covariance matrix, which defines a correlated fit. Instead a more
    practical choice for cases where a reliable estimate of covariances
    is not available, is given by the inverse of the diagonal part of
    the covariance matrix, which instead defines an uncorrelated fit.

    Parameters:
       x (array): array with the values of the x-axis; 2-D arrays are
          accepted and the second dimension is intepreted with the index
          :math:`\mu`
       W (array): 2-D array with the matrix :math:`W`; if a 1-D array is
          passed, the program inteprets it as the inverse of the
          diagonal entries of the covariance matrix
       f (function): callable function or lambda function defining :math:`\phi`;
          the program assumes :math:`x_i^\mu` correspond to the first arguments
       df (function): callable function or lambda function returning an array
          that contains the gradient of :math:`\phi`, namely :math:`\partial
          \phi(\{p\},\{x_i\})/\partial p_\\alpha`
       v (str, optional): a string with the list of variables used in `f` as
          the kinematic coordinates. Default value corresponds to `x`, which
          implies that `f` must be defined using `x` as first and unique
          kinematic variable.

    Notes:
       Once created the class must be called with at least one argument given by
       the observables corresponding to the data points :math:`y_i`. See examples
       below.

    Examples:
       >>> xax=[1,2,3,4]
       >>> f=lambda x,p0,p1: p0 + p1*x
       >>> df=lambda x,p0,p1: [1, x]
       >>> [y,dy] = yobs1.error()
       >>> W=1./dy**2
       >>> fit1 = mfit(xax,W,f,df)
       >>> pars = fit1(yobs1)
       >>> print(pars)
       0.925(35)    2.050(19)
    """

    def __init__(self, x, W, f, df, v="x"):
        if numpy.ndim(W) == 1:
            r = len(W)
            W = numpy.diag(W)
        elif numpy.ndim(W) == 2:
            [r, c] = numpy.shape(W)
            pyobs.assertion(r == c, "Rectangular W matrix, must be square")
        else:  # pragma: no cover
            raise pyobs.PyobsError("Unexpected size of W matrix")
        tmp = [s.strip() for s in v.rsplit(",")]
        self.csq = {0: chisquare(x, W, f, df, tmp)}
        self.pdict = {}
        for i in range(len(self.csq[0].pars)):
            self.pdict[self.csq[0].pars[i]] = i

    def copy(self):
        c = self.csq[0]
        res = mfit(c.x, c.W, c.f, c.df, ",".join(c.v))
        for ic in self.csq:
            if ic > 0:
                c = self.csq[ic]
                res.csq[ic] = chisquare(c.x, c.W, c.f, c.df, c.v)
        res.pdict = {}
        for key in self.pdict:
            res.pdict[key] = self.pdict[key]
        return res

    def parameters(self):
        """
        Prints the list of parameters
        """
        print("Parameters : " + ", ".join([key for key in self.pdict.keys()]))

    def __add__(self, mf):
        res = self.copy()
        n = len(res.csq)
        for ic in mf.csq:
            c = mf.csq[ic]
            res.csq[n] = chisquare(c.x, c.W, c.f, c.df, c.v)
            n += 1
        n = len(res.pdict)
        for key in mf.pdict:
            if key not in res.pdict:
                res.pdict[key] = n
                n += 1
        return res

    @pyobs.log_timer("mfit")
    def __call__(self, yobs, p0=None, min_search=None):
        if len(self.csq) > 1:
            pyobs.check_type(yobs, "yobs", list)
        else:
            if isinstance(yobs, pyobs.observable):
                yobs = [yobs]
        pyobs.assertion(
            len(yobs) == len(self.csq),
            f"Unexpected number of observables for {len(self.csq)} fits",
        )
        if p0 is None:
            p0 = [1.0] * len(self.pdict)
        if min_search is None:
            min_search = lm

        def csq(p0):
            res = 0.0
            for i in range(len(yobs)):
                self.csq[i].set_pars(self.pdict, p0)
                res += self.csq[i](yobs[i].mean)
            return res

        def dcsq(x):
            return sum(
                [self.csq[i].grad(yobs[i].mean, self.pdict) for i in range(len(yobs))]
            )

        def ddcsq(x):
            return sum(
                [self.csq[i].hess(yobs[i].mean, self.pdict) for i in range(len(yobs))]
            )

        res = min_search(csq, p0, jac=dcsq, hess=ddcsq)

        # properly create gradients
        H = self.csq[0].Hmat(self.pdict, res.x)
        for i in range(1, len(yobs)):
            H += self.csq[i].Hmat(self.pdict, res.x)
        Hinv = numpy.linalg.inv(H)

        g = []
        for i in range(len(yobs)):
            tmp = self.csq[i].gvec(self.pdict, res.x)
            g.append(pyobs.gradient(Hinv @ tmp))

        if pyobs.is_verbose("mfit"):
            print(f"chisquare = {res.fun}")
            print(f"minimizer iterations = {res.nit}")
            print(f"minimizer status: {res.message}")
        return pyobs.derobs(yobs, res.x, g)

    def chisquared(self, pars):
        res = 0.0
        for i in range(len(self.csq)):
            self.csq[i].set_pars(self.pdict, pars.mean)
            res += self.csq[i].csq()
        return res

    def eval(self, xax, pars):
        """
        Evaluates the function on a list of coordinates using the parameters
        obtained from a :math:`\\chi^2` minimization.

        Parameters:
           xax (array,list of arrays) : the coordinates :math:`x_i^\\mu` where
              the function must be evaluated. For combined fits, a list of
              arrays must be passed, one for each fit.
           pars (obs) : the observable returned by calling this class

        Returns:
           list of obs : observables corresponding to the functions evaluated
           at the coordinates `xax`.

        Examples:
           >>> fit1 = mfit(xax,W,f,df)
           >>> pars = fit1(yobs1)
           >>> print(pars)
           0.925(35)    2.050(19)
           >>> xax = numpy.arange(0,10,0.2)
           >>> yeval = fit1.eval(xax, pars)
        """

        if not type(xax) is list:
            xax = [xax]
        pyobs.check_type(pars, "pars", pyobs.observable)
        N = len(xax)
        pyobs.assertion(
            N == len(self.csq),
            "Coordinates and Paramters do not match number of internal functions",
        )
        out = []
        for ic in self.csq:
            [m, g] = self.csq[ic].eval(xax[ic], self.pdict, pars.mean)
            out.append(pyobs.derobs([pars], m, [pyobs.gradient(g)]))
        if len(out) == 1:
            return out[0]
        return out
