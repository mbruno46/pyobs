import numpy
import matplotlib.pyplot as plt

from .utils import valerr

import libcore

__all__ = ['uwerr']

def uwerr(data,data2,ncnfg,plot=False,opts=(),Stau=1.5,W=None):
	R=len(ncnfg)
	N = numpy.sum(ncnfg)
	Wmax = min(ncnfg)/2
	
	gg = numpy.zeros(Wmax, dtype=numpy.double)
	for ir in range(R):
        if data2 is None:
            gg = gg + gamma_fft(Wmax, data[ir])
        else:
            gg = gg + gamma_fft(Wmax, data[ir], data2[ir])
	gg2 = normalize_gamma(gg, Wmax, N, R)

	if (W==None):
		[Wopt, Wmax] = find_window(gg2/gg2[0], N, Stau)
	else:
		Wopt=W
		Wmax=2*W

	gg3 = correct_gamma_bias(gg2, Wopt, N)
	res = tauint(gg3,Wopt,N)

    if (plot==True):
		rho = gg3/gg3[0]
		drho = libcore.compute_drho(rho,N)

	if (plot==True):
        _rho_plotter(rho,drho,Wmax,Wopt,res[1:3],opts[:-1],opts[-1])

	return res

#def uwerr_texp(data,ncnfg,plot=False,opts=(),texp=0.,W=None):


def _rho_plotter(rho,drho,Wmax,Wopt,tau,opts,fname=None):
	plt.figure()
	plt.title('Ensemble ' + opts[0] + ' ; $\\tau_\mathrm{int}$ = ' + valerr(tau[0],tau[1]))
	plt.xlabel('$W$')
	plt.plot([0,Wmax],[0,0],'-k',lw=.75)
	plt.plot([Wopt,Wopt],[0,1],'--r',label='$W_\mathrm{opt} = '+str(Wopt)+'$')
	plt.errorbar(range(Wmax), rho[range(Wmax)], drho[range(Wmax)], fmt='.', label='$\\rho$')
	plt.legend(loc='upper right')
	plt.tight_layout()
	if (fname!=None):
		plt.savefig(fname+'_'+opts[0]+'_'+str(opts[1])+str(opts[2])+'.pdf')
	plt.show()


##########################################################


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

def compute_drho(rho,N):
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


