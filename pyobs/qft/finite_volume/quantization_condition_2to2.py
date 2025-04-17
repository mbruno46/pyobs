#################################################################################
#
# quantization_condition_2to2.py: methods and functionalities
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
from .zeta00 import Zeta00

__all__ = ["moving_frame", "com_frame", "single_channel"]


def pick_backend(type, zeta00):
    def __numpy(x, der):
        return zeta00(x, der)

    def __pyobs(x, der):
        mean = zeta00(x.mean, der)
        derz = zeta00(x.mean, der + 1)
        g = pyobs.gradient(lambda x: x * derz, x.mean, gtype="full")
        return pyobs.derobs([x], mean, [g])

    return __numpy if type == "numpy" else __pyobs


def moving_frame(P):
    raise pyobs.PyobsError("not implemented")


def com_frame(D=3):
    return Zeta00(D=D)


class single_channel:
    def __init__(self, frame, L, m, backend="numpy"):
        self.L = L
        self.m = m
        self.D = frame.D
        self.arctan = numpy.arctan if backend == "numpy" else pyobs.arctan
        self.backend = backend
        self.zeta00 = pick_backend(backend, frame)
        self.frame = frame

    def pstar(self, E):
        return (E**2 / 4 - self.m**2) ** (0.5)

    def q(self, E):
        return self.pstar(E) * self.L / (2 * numpy.pi)

    def E(self, q):
        return 2 * ((2 * numpy.pi * q / self.L) ** 2 + self.m**2) ** 0.5

    # phi(E) + delta(E) = n Pi
    # q = (E^2/4 - m^2)^(1/2) L/(2*Pi)
    # tan[phi(E)] = - q * Pi^(3/2) / z00(q^2)
    def phi(self, E):
        q = self.q(E)
        z00 = self.zeta00(q * q, 0)
        return self.arctan(q * numpy.pi**1.5 / z00)

    def der_phi(self, E):
        q = self.q(E)
        qsq = q * q
        z00 = self.zeta00(qsq, 0)
        dz00 = self.zeta00(qsq, 1)
        dphi = numpy.pi**1.5 * (z00 - 2 * qsq * dz00) / (numpy.pi**3 * qsq + z00**2)
        return dphi * E * self.L / (8 * numpy.pi * self.pstar(E))

    # tan(phi(E)) = q* Pi^(3/2)/z00(q^2) = tan(delta(E(q)))
    # 2*[((2*Pi)/L *q)^2 + m^2 ]^(1/2) = E(q)
    def En(self, tandelta, n):
        if not self.frame.nth_state_exists(n):
            return None
        pref = numpy.pi ** (3 / 2)

        def g(qsq):
            q = numpy.sqrt(qsq)
            t = tandelta(self.E(q))
            if abs(t) < 1e-12:
                return numpy.inf
            return q * pref / t

        qsq = self.frame.solve(g, n)
        return 2 * numpy.sqrt(qsq * (2 * numpy.pi / self.L) ** 2 + self.m**2)

    def get_energy(self, tandelta, n0=0):
        n = n0
        while True:
            yield self.En(tandelta, n)
            n += 1
            while not self.frame.nth_state_exists(n):
                n += 1

    # | 2 pi, L > <L, 2pi |  = |2pi,out> R <out, 2pi|
    # R = q/(16 pi E) / (phi' + delta')
    def R(self, E, ddelta):
        q = self.qstar(E)
        pref = q / (16 * numpy.pi) / E
        return pref / (-self.der_phi(E) + ddelta(E))
