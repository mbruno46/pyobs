import pyobs
Nd = 3 #pyobs.qft.finite_volume.Nd

import numpy
from scipy.integrate import quad
from scipy.optimize import root_scalar


def factorial(n):
    if n==0:
        return 1
    else:
        return n * factorial(n-1)
    
class Zeta00:
    def __init__(self, d=[0]*Nd, gamma=1, nmax=10, eps=1e-12):
        self.d = numpy.array(d)
        self.normd = self.d @ self.d
        self.gamma = gamma
        self.nmax = nmax
        self.eps = eps
        if Nd==3:
            vectors = numpy.array([numpy.array([x,y,z]) for x in range(-nmax,nmax+1) for y in range(-nmax,nmax+1) for z in range(-nmax, nmax+1)])
        else:
            raise pyobs.PyobsError('quantization condition in 1 and 2 dimensions not implemented')
        
        self.vectors = {}
        for n in range(nmax+1):
            self.vectors[n] = []
        for v in vectors:
            for n in reversed(range(nmax+1)):
                if numpy.isin(n,numpy.unique(numpy.abs(v))):
                    self.vectors[n] += [v]
                    break
        self.factorial = [factorial(n) for n in range(200)]
        
    # flag can be 0 or 1
    def norm(self, x, flag):
        r0 = x + self.d*0.5 * flag
        npar = numpy.zeros((len(x),)) if self.normd==0.0 else (r0@self.d)/self.normd * self.d
        nperp = r0 - npar
        r = nperp + npar * self.gamma * (1-flag) + flag * npar / self.gamma
        return r @ r

    
    def I0(self, m, n, der):
        out = 0
        for x in self.vectors[n]:
            r = self.norm(x,1)
            if der==0:
                out += numpy.exp(-(r-m))/(r-m)
            elif der==1:
                out += numpy.exp(-(r-m))*(1+r-m)/(r-m)**2
            elif der==2:
                out += numpy.exp(-(r-m))*(2+2*r+r*r-2*m-2*m*r+m*m)/(r-m)**3
        return out
        
        
    def I1(self, m, n, der):
        def g(t):
            return numpy.sum([(-1.0)**(x@self.d) * numpy.exp(-numpy.pi**2 * self.norm(x,0)/t) for x in self.vectors[n]])
        pow = 1.5 - der
        return quad(lambda t: numpy.exp(t*m) * numpy.pi**1.5/t**pow * g(t), 0.0, 1.0)[0]
    
    
    def I2(self, m, l, der):
        A = 1
        if der==1:
            A = l
        elif der==2:
            A = l*(l-1)
        return numpy.pi**1.5 / (l-0.5) * A * m**(l-der) / self.factorial[l]

    
    def __sum(self, func, init):
        n = 1
        delta = init+1
        res = init
        while (abs(delta) > self.eps * abs(res)):
            delta = func(n)
            res += delta
            n += 1
        return res
    
    
    def inner(self, qsq, der):
        res = self.__sum(lambda n: self.I0(qsq, n, der), self.I0(qsq, 0, der))
        I1 = self.__sum(lambda n: self.I1(qsq, n, der), 0.0)
        I2 = self.__sum(lambda n: self.I2(qsq, n, der), self.I2(qsq, 0, der))
        return (res + self.gamma * (I1 + I2)) / numpy.sqrt(4*numpy.pi)

        
    def __call__(self, q, der=0):
        qsq = q*q
        if float(qsq).is_integer():
            return numpy.inf
        
        if der==0:
            return self.inner(qsq, 0)
        elif der==1:
            return self.inner(qsq,1) * 2 * q
        elif der==2:
            return self.inner(qsq,2) * 2* q + self.inner(qsq,1)*2


    # def derivative(self, q):
    #     if isinstance(q, pyobs.observable):
    #         mean = self.__call_float(q.mean, 1)
    #         g = pyobs.gradient(lambda x: self.__call_float(x, 2), q.mean, gtype="full")
    #         return pyobs.derobs([qsq], mean, [g])
    #     else:
    #         return self.__call_float(q, 1)
    
#     def solve(self, f, n):
#         bracket = [numpy.sqrt(n)+self.eps, numpy.sqrt(n+1)-self.eps]
#         res = root_scalar(lambda x: self(x) - f(x), bracket=bracket)
#         if res.converged:
#             return res.root
#         return None
    