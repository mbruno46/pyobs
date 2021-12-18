import numpy
import pyobs

from .zeta import Zeta00

def backend(type, zeta00):
    def __numpy(x, der):
        return zeta00(x, der)
    
    def __pyobs(x, der):
        mean = self.zeta00(x.mean, der)
        g = pyobs.gradient(lambda x: zeta00(x, der+1), x.mean, gtype="full")
        return pyobs.derobs([x], mean, [g])

    return __numpy if type=='numpy' else __pyobs

class single_channel:
    def __init__(self, P, L, m, be='numpy'):
        self.P = P
        self.L = L
        self.m = m
        self.arctan = numpy.arctan if backend=='numpy' else pyobs.arctan
        self.backend = be
        self.gamma = 1.0
        self.zeta00 = backend(be, Zeta00(P, gamma=self.gamma))
#         self.zeta00 = Zeta00(P, gamma=self.gamma)
        
    def qstar(self, E):
        if self.m2 is None:
            return (E**2 / 4 - self.m**2)**(0.5)
        
    def phi(self, E):
        p = self.qstar(E) * self.L / (2*numpy.pi)
        z00 = self.backend.zeta00(p, 0)
        return self.backend.arctan(p * numpy.pi**1.5 / z00)

    def der_phi(self, E):
        p = self.qstar(E) * self.L / (2*numpy.pi)
        z00 = self.backend.zeta00(p, 0)
        dz00 = self.backend.zeta00(p, 1)
        dphi = numpy.pi**1.5 * (z00 - p * dz00)/(numpy.pi**3 * p**2 + z00**2)
        return dphi * E * self.L / (8 * numpy.pi * self.qstar(E))

    def En(self, tandelta, n):
        pref = numpy.pi**(3/2)
        def g(x):
            return x * pref / tandelta(2*numpy.pi*x/self.L)
        
        bracket = [numpy.sqrt(n)+self.eps, numpy.sqrt(n+1)-self.eps]
        res = root_scalar(lambda x: self.zeta00(x) - g(x), bracket=bracket)
        p = res.root if res.converged else None

        qstar = p * 2 *numpy.pi / self.L
        return numpy.sqrt(4*qstar**2 + 4*self.m**2)

    def LellouchLuescher(self, E):
        return None