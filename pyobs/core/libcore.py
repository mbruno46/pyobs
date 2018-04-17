
from .utils import valerr

import numpy
import numpy.ctypeslib as npct
import ctypes #from ctypes import CDLL, c_int
import os

##################################################
### loading c-library

cptr = npct.ndpointer(dtype=numpy.double, ndim=1, flags='CONTIGUOUS')

cwd = os.path.dirname(os.path.abspath(__file__))

#_libcore = npct.load_library("_libcore.so",".")
_libcore = ctypes.CDLL(cwd + "/_libcore.so")

_libcore.compute_drho.restype = None
_libcore.compute_drho.argtypes = [cptr, cptr, ctypes.c_int, ctypes.c_int]

##################################################


def gamma(Wmax,data1,data2=None):
    N=len(data1)
    g = numpy.zeros(Wmax, dtype=numpy.double)

    if (data2 is not None):
        if (len(data2)!=N):
            raise
    else:
        data2 = data1
    
    for t in range(Wmax):
        g[t] = data1[0:N-t].dot(data2[t:N])
    
    return g

def normalize_gamma(gamma, Wmax, N, R):
    n = N-R*numpy.arange(0.,Wmax,1.)
    nn = 1.0/n
    return numpy.multiply(gamma, nn)

def correct_gamma_bias(gamma, W, N):
    Copt = gamma[0] + 2.*numpy.sum(gamma[1:W+1])
    return gamma + Copt/float(N)

def compute_drho(rho, N):
    tmax = len(rho)
    drho = numpy.zeros(len(rho), dtype=numpy.double)
    _libcore.compute_drho(drho, rho, tmax, N)
    return drho 

def compute_drho_py(rho,N):
    tmax = len(rho)
    drho = numpy.zeros(len(rho), dtype=numpy.double)
    for i in range(1,tmax/2):
        if (i==1):
            hh = rho[0:tmax-2]
        else:
            hh = numpy.r_[rho[i-1::-1], rho[1:tmax-2*i]]
        h = rho[i+1:tmax] + hh -2.0*rho[i]*rho[1:tmax-i]
        drho[i] = numpy.sqrt( numpy.sum( h**2 )/N )
    return drho
    
def tauint(gamma, W, N):
    sum_gamma = gamma[0] + 2.*numpy.sum(gamma[1:W+1])
    sigma = sum_gamma/float(N)
    tau = sum_gamma*(0.5/gamma[0])
    dtau = tau*2*numpy.sqrt((W-tau+0.5)/float(N));
    return [sigma, tau, dtau]


def find_window(rho, N, Stau, texp=None):
    Wmax = int(len(rho))
    rho_int = 0.
    flag=0
    for W in range(1,Wmax):
        rho_int = rho_int + rho[W]
        tauW = Stau/numpy.log((rho_int+1.)/rho_int)
        gW = numpy.exp(-W/tauW) - tauW/numpy.sqrt(W*N)
        if (gW<0):
            Wopt = W
            Wmax = min([Wmax,2*Wopt])
            flag=1
            break
    if (flag==0):
        print 'Warning: automatic window procedure failed'
        Wopt = Wmax
    return [Wopt, Wmax]

def find_upper_bound(rho, drho, Nsigma, Wsmall):
    Wmax = int(len(rho))
    for W in range(Wmax-1):
        if (rho[W] < Nsigma*drho[W]):
            break
    if (W==(Wmax-2)):
        print ('Warning: Could not meet Nsigma criterion; using t %d' % W)
    else:
        ff = lambda t : numpy.fabs(numpy.fabs(rho[t])-Nsigma*drho[t])
        if (ff(W-1)<ff(W)):
            W -= 1
    
    if Nsigma>1.5:
        cons_level = 2.0
    else:
        cons_level = 0.5
        
    if (W<Wsmall or rho[W]<0.):
        ConsRho = numpy.sum([ max(rho[i], cons_level*drho[i]) for i in range(W,W+Wsmall) ])
    else:
        ConsRho = numpy.sum([ rho[i] for i in range(W,W+Wsmall) ])
    return [W,  ConsRho]

