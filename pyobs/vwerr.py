
import numpy
import matplotlib.pyplot as plt

from .core.utils import valerr

from core import libcore

__all__ = ['uwerr','uwerr_texp','errinfo']


class errinfo:
    """ Class for error computation
    
    It contains the basic parameters for the calculation
    of the error: Stau for UWerr and Texp, Nsigma, Wsmall
    for UWerrTexp.
    If not given the defaults are Stau=1.5, Texp=0.0, Nsigma=2
    and Wsmall=0. If we the parameter W is passed the automatic
    window procedure is bypassed and W is used as summation window.
    """
    
    def __init__(self):
        self.eid = []
        self.Stau = []
        self.Texp = []
        self.Nsigma = []
        self.Wsmall = []
        self.W = []

    def addEnsemble(self, eid, Stau=1.5, Texp=0., Nsigma=2, Wsmall=0, W=None):
        """ Add non-defaults parameters for a given ensemble
        
        Parameters
        ----------
        eid: int
            ensemble identifier; it must match the id used to create the
            observables
        Stau: float, optional
        Nsigma: int, optional
        Wsmall: int, optional
        
        Examples
        --------
        Assuming to have an observable defined on two ensembles with 0 and 1,
        the following example shows how to compute the error with UWerr for 
        ensemble 0 and with UWerrTexp for ensemble 1:
        
        >>> einfo = errinfo()
        >>> einfo.addEnsemble(0,Stau=2.5)
        >>> einfo.addEnsemble(1,Texp=15.0)
        >>> obs.vwerr(errinfo=einfo)
        """
        
        self.eid.append( eid )
        self.Stau.append( Stau )
        self.Texp.append( Texp )
        self.Nsigma.append( Nsigma )
        self.Wsmall.append( Wsmall )
        self.W.append( W )

    def getUWerr(self, eid):
        try:
            ie = self.eid.index(eid)
            return (self.Stau[ie], self.W[ie])
        except:
            return (1.5, None)

    def getUWerrTexp(self, eid):
        ie = self.eid.index(eid)
        return (self.Stau[ie], self.Texp[ie], self.Nsigma[ie], self.Wsmall[ie], self.W[ie])

    def Tail(self, eid):
        try:
            ie = self.eid.index(eid)
            if (self.Texp[ie]>0.):
                return True
        except:
            return False

    def __str__(self):
        out = ''
        for ie in range(len(self.eid)):
            out = out + ('Ensemble id %d \n' % self.eid[ie])
            out = out + ('\t Stau = %.1f ; Texp = %.2f ; Nsigma = %d \n' % (self.Stau[ie],self.Texp[ie],self.Nsigma[ie]))
        return out


def _compute_gamma(data,ncnfg):
    R=len(ncnfg)
    N = numpy.sum(ncnfg)
    Wmax = min(ncnfg)/2
    
    gg = numpy.zeros(Wmax, dtype=numpy.double)
    for ir in range(R):
        gg = gg + libcore.gamma(Wmax, data[ir])
    return [libcore.normalize_gamma(gg, Wmax, N, R), N]

def uwerr(data,ncnfg,plot=False,plotopts=('',0.,0.,''),Stau=1.5,W=None):
    [gg, N] = _compute_gamma(data,ncnfg)
    if (gg[0]==0.0):
        return [0., 0., 0.]

    if (W==None):
        [Wopt, Wmax] = libcore.find_window(gg/gg[0], N, Stau)
    else:
        Wopt=W
        Wmax=2*W

    gg2 = libcore.correct_gamma_bias(gg, Wopt, N)
    res = libcore.tauint(gg2,Wopt,N)
    
    if (plot==True):
        rho = gg2/gg2[0]
        drho = libcore.compute_drho(rho,N)
        _rho_plotter(rho,drho,Wmax,Wopt,res[1:3],plotopts)

    return res

def _rho_plotter(rho,drho,Wmax,Wopt,tau,opts,texp=None):
    plt.figure()
    plt.title('Ensemble ' + opts[0] + ' ; $\\tau_\mathrm{int}$ = ' + valerr(tau[0],tau[1]))
    plt.xlabel('$W$')
    plt.plot([0,Wmax],[0,0],'-k',lw=.75)
    plt.plot([Wopt,Wopt],[0,1],'--r',label='$W_\mathrm{opt} = '+str(Wopt)+'$')
    plt.errorbar(range(Wmax), rho[range(Wmax)], drho[range(Wmax)], fmt='.', label='$\\rho$')

    if (texp!=None):
        xax=numpy.arange(Wopt,Wmax,(Wmax-Wopt-1)/50.)
        plt.plot(xax, rho[Wopt]*numpy.exp(-(xax-Wopt)/texp), '-g', label='$\\tau_\mathrm{exp}$')

    plt.legend(loc='upper right')
    plt.tight_layout()

    if (len(opts)==4 and isinstance(opts[3],str)):
        plt.savefig(opts[3]+'_'+opts[0]+'_'+str(opts[1])+str(opts[2])+'.pdf')
    plt.show()


def uwerr_texp(data,ncnfg,plot=False,plotopts=(),Stau=1.5,Texp=0.0,Nsigma=2,Wsmall=0,W=None):
    [gg, N] = _compute_gamma(data,ncnfg)
    if (gg[0]==0.0):
        return [0., 0., 0.]
    
    if (W==None):
        [Wopt, Wmax] = libcore.find_window(gg/gg[0], N, Stau)
    else:
        Wopt=W
        Wmax=2*W

    gg2 = libcore.correct_gamma_bias(gg, Wopt, N)
    
    rho = gg2/gg2[0]
    drho = libcore.compute_drho(rho,N)

    if (W==None):
        [Wup, amp] = libcore.find_upper_bound(rho, drho, Nsigma, Wsmall)
    else:
        Wup = W
        
    amp = rho[Wup+1]
    res1 = libcore.tauint(gg2,Wup,N)
    tau_upper = res1[1] + rho[Wup+1]*Texp
    dtau_upper = numpy.sqrt(res1[2]**2 + (drho[Wup+1]*Texp)**2)

    [sigma, tau, dtau] = libcore.tauint(gg2,Wopt,N)
        
    if (plot==True):
        if (tau_upper>tau):
            _rho_plotter(rho,drho,Wmax,Wup,[tau_upper,dtau_upper],plotopts, Texp)
        else:
            _rho_plotter(rho,drho,Wmax,Wopt,[tau,dtau],plotopts)

    if (tau_upper>tau):
        sigma = gg2[0]*2.*tau_upper/float(N)
        return [sigma, tau_upper, dtau_upper ]
    else:
        return [sigma, tau, dtau]


