import numpy
import pyobs
from scipy.optimize import root_scalar

from .zeta import Zeta00

__all__ = [
    'moving_frame',
    'com_frame',
    'single_channel'
]

def backend(type, zeta00):
    def __numpy(x, der):
        return zeta00(x, der)
    
    def __pyobs(x, der):
        mean = self.zeta00(x.mean, der)
        g = pyobs.gradient(lambda x: zeta00(x, der+1), x.mean, gtype="full")
        return pyobs.derobs([x], mean, [g])

    return __numpy if type=='numpy' else __pyobs

def moving_frame(P):
    gamma = 1
    return Zeta00(P, gamma)

def com_frame():
    return Zeta00()

class single_channel:
    def __init__(self, frame, L, m, be='numpy'):
        self.L = L
        self.m = m
        self.arctan = numpy.arctan if be=='numpy' else pyobs.arctan
        self.backend = be
        self.zeta00 = backend(be, frame)
        self.frame = frame
        
    def qstar(self, E):
        return (E**2 / 4 - self.m**2)**(0.5)
        
    def p(self, E):
        return self.qstar(E) * self.L / (2*numpy.pi)
    
    def E(self, p):
        return 2*((2*numpy.pi*p/self.L)**2 + self.m**2)**0.5
    
    # phi(E) + delta(E) = n Pi
    # p = (E^2/4 - m^2)^(1/2) L/(2*Pi)
    # tan[phi(E)] = - p * Pi^(3/2) / z00(p)
    def phi(self, E):
        p = self.p(E) #self.qstar(E) * self.L / (2*numpy.pi)
        z00 = self.zeta00(p, 0)
        return self.arctan(p * numpy.pi**1.5 / z00)

    def der_phi(self, E):
        p = self.p(E) #self.qstar(E) * self.L / (2*numpy.pi)
        z00 = self.zeta00(p, 0)
        dz00 = self.zeta00(p, 1)
        dphi = numpy.pi**1.5 * (z00 - p * dz00)/(numpy.pi**3 * p**2 + z00**2)
        return dphi * E * self.L / (8 * numpy.pi * self.qstar(E))

    def nvectors(self):
        return self.frame.vectors
    
    # tan(phi(E)) = p* Pi^(3/2)/z00(p) = tan(delta(E(p)))
    # 2*[((2*Pi)/L *p)^2 + m^2 ]^(1/2) = E(p)
    def En(self, tandelta, n):        
        pref = numpy.pi**(3/2)
        
        def g(x):
            t = tandelta(self.E(x))
            if abs(t)<1e-12:
                return numpy.inf
            return x * pref / t
        
        bracket = [numpy.sqrt(n)+1e-12, numpy.sqrt(n+1)-1e-14]
    
        res = root_scalar(lambda x: self.zeta00(x, 0) - g(x), bracket=bracket)
        p = res.root if res.converged else None

        qstar = p * 2 *numpy.pi / self.L
        return numpy.sqrt(4*qstar**2 + 4*self.m**2)

    # A_inf^2 = A_L^2 * [p phi' + q delta'] * (3 Pi E^2)/(2 q^5)
    # note that in our convetion phi -> -phi
    def LellouchLuescher(self, E, ddelta):
        q = self.qstar(E)
        p = self.p(E)
        pref = 3*numpy.pi*E**2 / (2*q**5)
        return (-p * self.der_phi(E) + q * ddelta(E))*pref
