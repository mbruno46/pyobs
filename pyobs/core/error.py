#################################################################################
#
# error.py: definition and properties of the error classes and functions
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
from scipy import special
import pyobs

try:  # pragma: no cover
    import matplotlib.pyplot as plt

    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False


def find_closest(arr, x0):
    return (numpy.abs(numpy.array(arr) - x0)).argmin()


class variance:
    def __init__(self, n, g, Stau, D, k, fold=False):
        self.D = D
        self.k = k
        self.Stau = Stau
        z = (D) * 0.5
        self.OmegaD = numpy.pi ** (z) / special.gamma(z) * 2

        (s, xmax) = numpy.shape(g)
        self.x = []
        for i in range(xmax):
            if numpy.any(n[:, i] > 1e-15):
                if D == 1:
                    self.x.append(i)
                else:
                    self.x.append(numpy.sqrt(i))

        gg = numpy.zeros((self.size, len(self.x)))

        idx = numpy.power(numpy.array(self.x), (1 if D == 1 else 2)).astype("i4")
        _n = numpy.array(n[:, idx], dtype=numpy.float)
        _n[(_n == 0)] = numpy.inf
        gg = g[:, idx] / _n

        if fold:
            _gg = numpy.array(gg)
            _gg[:, 1:] *= 2.0
            self.cvar = numpy.cumsum(_gg, axis=1)
        else:
            self.cvar = numpy.cumsum(gg, axis=1)
        self.N = n[:, 0]
        self.xopt = [self.x[-1]] * self.size
        self.var = numpy.zeros((self.size, 2))

    def g(self, i, a):
        cov = self.cvar[a, i] / self.cvar[a, 0]
        j = self.D - self.k
        xi = self.x[i]

        am = (special.gamma(j) * self.OmegaD / cov) ** (1.0 / j) / float(self.Stau)
        h = -numpy.exp(-am * xi) * xi ** (j - 1) * am ** (j) / special.gamma(j)
        return h + numpy.sqrt(
            self.D * self.OmegaD * 0.5 / self.N[a] * xi ** (self.D - 2)
        )

    def find_opt(self):
        for a in range(self.size):
            if self.cvar[a, 0] == 0.0:
                i = 0
            else:
                for i in range(1, len(self.x)):
                    if self.g(i, a) > 0:
                        break
            if self.cvar[a, i] < 0:
                i = 0
                print(f"Warning: automatic window failed for obs {a}, using {i}")

            self.xopt[a] = self.x[i]
            self.var[a, 0] = self.cvar[a, i]
            self.var[a, 1] = self.cvar[a, i] * self.stat_relerr(self.x[i], a)

            if self.xopt[a] == self.x[-1]:  # pragma: no cover
                print(
                    f"Warning: automatic window failed for obs {a}, using {self.xopt[a]}"
                )

    def set_opt(self, xopt):
        for a in range(self.size):
            if isinstance(xopt,(list, numpy.ndarray)):
                if xopt[a] in self.x:
                    i = self.x.index(xopt[a])
                else:
                    i = find_closest(self.x, xopt[a])
            else:
                if xopt in self.x:
                    i = self.x.index(xopt)
                else:
                    i = find_closest(self.x, xopt)

            self.xopt[a] = self.x[i]
            self.var[a, 0] = self.cvar[a, i]
            self.var[a, 1] = numpy.abs(self.cvar[a, i]) * self.stat_relerr(self.x[i], a)

    def stat_relerr(self, r, a):
        return numpy.sqrt(
            2.0 * self.OmegaD / (self.D * self.N[a]) * numpy.array(r) ** self.D
        )

    def correct_gamma_bias(self):
        for a in range(self.size):
            # f = self.OmegaD * (self.xopt[a] ** self.D) / (self.N[a] * self.D)
            # self.var[a] += self.var[a] * f
            # self.cvar[a, :] += self.cvar[a, :] * f
            # Gamma -> Gamma + C/N
            # cumsum(Gamma) -> cumsum(Gamma) + cumsum(C/N)
            self.cvar[a, :] += numpy.arange(1,len(self.cvar[a,:])+1)*self.var[a,0] / self.N[a]
        self.set_opt(self.xopt)
        
    def tauint(self):
        tau = numpy.zeros((self.full_size, 2))
        for a in self.mask:
            i = self.mask.index(a)
            if self.cvar[i, 0] > 0.0:
                tau[a, 0] = self.var[i, 0] / self.cvar[i, 0]
                tau[a, 1] = self.var[i, 1] / self.cvar[i, 0]
        return tau

    def sigma(self):
        out0 = numpy.zeros((self.full_size,))
        out1 = numpy.zeros((self.full_size,))
        for a in self.mask:
            i = self.mask.index(a)
            out0[a] = self.var[i, 0] / self.N[i]
            out1[a] = self.var[i, 1] / self.N[i]
        return [out0, out1]

    def plot(self, xlab, desc, ed, pfile):  # pragma: no cover
        if not MATPLOTLIB:
            pass

        for a in range(self.size):
            plt.figure()
            plt.title(f"{desc}; {ed}; {a}")
            plt.xlabel(xlab)
            plt.ylabel("covariance")

            y = self.cvar[a, :] / self.cvar[a, 0]
            err = self.stat_relerr(self.x, a)

            plt.plot(self.x, y, ".-", color="C0")
            plt.fill_between(
                self.x, y * (1.0 + err), y * (1.0 - err), alpha=0.3, color="C0"
            )
            plt.plot([0, self.x[-1]], [0, 0], "-k", lw=0.75)
            plt.plot(
                [self.xopt[a]] * 2,
                [0, self.var[a, 0] / self.cvar[a, 0]],
                "-",
                color="C1",
                label=f"opt={self.xopt[a]}",
            )
            plt.xlim(0, self.xopt[a] * 2)
            dy = self.var[a, 1] / self.cvar[a, 0]
            plt.ylim(1 - dy * 2, self.var[a, 0] / self.cvar[a, 0] + dy * 3)
            plt.legend(loc="upper right")
            plt.show()

    def cum_var(self):
        dy = numpy.array(self.cvar)
        for a in range(self.size):
            dy[a, :] *= self.stat_relerr(self.x, a)
        return [self.x, self.cvar, dy]


def plot_piechart(desc, errs, tot):  # pragma: no cover
    if not MATPLOTLIB:
        pass

    n = numpy.reciprocal(tot)
    s = numpy.size(tot)
    x = []
    for v in errs.values():
        x.append(numpy.reshape(v * n, (s,)))
    x = numpy.array(x)
    for a in range(s):
        plt.figure()
        plt.title(f"{desc}; {a}")
        plt.pie(x[:, a], labels=errs.keys(), autopct="%.0f%%", radius=1.0)
        plt.axis("equal")
        plt.show()


def init_var(x, name):
    keys = []
    ismf = False
    for rn in x.delta:
        if rn.split(":")[0] == name:
            keys.append(rn)
            if not x.delta[rn].lat is None:
                ismf = True
    if ismf:
        lat = x.delta[keys[0]].lat
        for ik in keys:
            if numpy.any(x.delta[ik].lat != lat):  # pragma: no cover
                raise pyobs.PyobsError(
                    "Lattice does not match among different replicas of same ensemble"
                )
        xmax = x.delta[
            keys[0]
        ].rrmax()  # int(min([x.mfdata[kk].rrmax() for kk in keys]))
        rescale = [1] * len(keys)
    else:
        rescale = numpy.zeros((len(keys),), dtype=numpy.int)
        wmax = numpy.zeros((len(keys),), dtype=numpy.int)
        for ik in range(len(keys)):
            d = x.delta[keys[ik]]
            rescale[ik] = numpy.min(numpy.diff(d.idx))
            wmax[ik] = d.wmax() * rescale[ik]
        gcd = numpy.gcd.reduce(rescale)  # greatest common divisor
        rescale = rescale // gcd
        xmax = min(wmax) // gcd
        del wmax

    return [keys, ismf, xmax, rescale]


class var(variance):
    def __init__(self, x, name, Stau, k):
        [keys, self.ismf, xmax, rescale] = init_var(x, name)

        # union of masks from replicas
        mask = []
        for ik in keys:
            mask += x.delta[ik].mask
        self.mask = list(set(mask))
        self.size = len(self.mask)

        n = numpy.zeros((self.size, xmax), dtype=numpy.float)
        g = numpy.zeros((self.size, xmax), dtype=numpy.float)

        # with this code we cover the situation where 1 obs happens to be known on 1 replica
        # and another obs in another replica, which may have different dtrj
        for i in range(len(keys)):
            d = x.delta[keys[i]]
            _xmax = xmax // rescale[i]
            for j in range(d.size):
                a = self.mask.index(d.mask[j])
                res = d.gamma(_xmax, j)
                n[a, :: rescale[i]] += res[0]
                g[a, :: rescale[i]] += res[1]

        if self.ismf:
            D = len(x.delta[keys[0]].lat)
            variance.__init__(self, n, g, Stau, D, k)
        else:
            variance.__init__(self, n, g, Stau, 1, k, fold=True)
        self.full_size = x.size


class covar(variance):
    def __init__(self, x, name):
        [keys, self.ismf, xmax, rescale] = init_var(x, name)

        # union of masks from replicas
        mask = []
        for ik in keys:
            mask += x.delta[ik].mask
        self.mask = list(set(mask))
        self.size = len(self.mask)
        self.size *= self.size + 1
        self.size //= 2

        n = numpy.zeros((self.size, xmax), dtype=numpy.float)
        g = numpy.zeros((self.size, xmax), dtype=numpy.float)

        # with this code we cover the situation where 1 obs happens to be known on 1 replica
        # and another obs in another replica, which may have different dtrj
        for i in range(len(keys)):
            d = x.delta[keys[i]]
            _xmax = xmax // rescale[i]
            for u in range(d.size):
                a = self.mask.index(d.mask[u])
                for v in range(u, d.size):
                    res = d.gamma(_xmax, u, v)
                    b = self.mask.index(d.mask[v])

                    j = b + a * (len(self.mask) - 1) - a * (a - 1) // 2
                    n[j, :: rescale[i]] += res[0]
                    g[j, :: rescale[i]] += res[1]

        # Stau and k passed but not utilized
        Stau = 1.5
        k = 1
        if self.ismf:
            D = len(x.delta[keys[0]].lat)
            variance.__init__(self, n, g, Stau, D, k)
        else:
            variance.__init__(self, n, g, Stau, 1, k, fold=True)
        self.full_size = x.size

    def covar(self):
        out0 = numpy.zeros((self.full_size, self.full_size))
        out1 = numpy.zeros((self.full_size, self.full_size))
        c = 0
        for a in self.mask:
            for b in self.mask[self.mask.index(a) :]:
                out0[a, b] = self.var[c, 0] / self.N[c]
                out1[a, b] = self.var[c, 1] / self.N[c]

                out0[b, a] = out0[a, b]
                out1[b, a] = out1[a, b]
                c += 1
        return [out0, out1]


class errinfo:
    def __init__(self, Stau=1.5, k=0, W=None):
        self.Stau = Stau
        self.k = k
        self.W = W


def gamma_error(x, name, plot=False, pfile=None, einfo=None):
    if einfo is None:
        einfo = errinfo()

    v = var(x, name, einfo.Stau, einfo.k)
    if einfo.W is None:
        v.find_opt()
    else:
        v.set_opt(einfo.W)

    if not v.ismf:
        v.correct_gamma_bias()
        tau = v.tauint() * 0.5
    else:
        tau = v.tauint()

    if plot:  # pragma: no cover
        v.plot("|R|/a" if v.ismf else "icnfg", x.description, name, pfile)
    return v.sigma() + [tau] + v.cum_var()


def covariance(x, name, W):
    cov = covar(x, name)
    cov.set_opt(W)

    if not cov.ismf:
        cov.correct_gamma_bias()

    return cov.covar()
