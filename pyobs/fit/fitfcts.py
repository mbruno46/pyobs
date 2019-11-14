import numpy

#from .lm import lm, lm_2
#from .chi2exp import derfit
#from .utils import extract_data, reshape_gradient
from ..observa import observa, derobs

from scipy.special import gammainc

import sympy
from sympy.parsing.sympy_parser import parse_expr

__all__ = ['pfit','mfit']

# index = i + n*(b + d1*a)
def chisq(func, xdata, pars, ydata, Wmat):
    [d0,d1,n] = ydata.shape
    E = []
    for a in range(d0):
        for b in range(d1):
            E.extend([func[a*d1+b](pars, *xdata[i]) - ydata[a,b,i] for i in range(n)])
    E = numpy.array(E)
    return [E.dot(Wmat).dot(E.transpose()), E]


def chisq_derivatives(dfunc, ddfunc, xdata, pars, E, Wmat):
    Np = len(pars)
    [n,_] = numpy.shape(xdata)

    der = []
    EW = 2.*E.dot(Wmat)
    Hbar = numpy.zeros((Np,Np))
    for k in range(len(dfunc)):
        for i in range(n):
            der.append(dfunc[k](pars, *xdata[i]))

            if (ddfunc!=None):
                h = ddfunc[k](pars, *xdata[i])
                Hbar = Hbar + EW[k*n + i] * numpy.array(h)
    der = numpy.array(der)

    return [EW.dot(der), 2.*der.transpose().dot(Wmat).dot(der) + Hbar]


def derfit(xdata, ydata, W, pars, func, grad, hess, split):
    np = len(pars)
    nf = len(func)
    (d0,d1,n) = ydata.shape

    E = []
    for a in range(d0):
        for b in range(d1):
            E.extend([func[a*d1+b](pars, *xdata[i]) - ydata[a,b,i] for i in range(n)])
    E = numpy.array(E)
    EW = E.dot(W)

    dfunc = numpy.zeros((n*nf,np))
    for k in range(nf):
        for i in range(n):
            dfunc[k*n+i,:] = numpy.array(grad[k](pars,*xdata[i]))
    
    WG = W.dot(dfunc)
    Hmat = dfunc.transpose().dot(WG)

    if (hess!=None):
        ddfunc = numpy.zeros((np,np))
        for k in range(nf):
            for i in range(n):
                ddfunc = numpy.array(hess[k](pars,*xdata[i]))
                Hmat = Hmat + EW[k*n+i]*ddfunc
    
    Hinv = numpy.linalg.inv(Hmat)
    
    dpars = []
    tmp = WG.dot(Hinv)
    if (split==False):
        h = numpy.zeros((1,np,nf,n))
        for i in range(n):
            for a in range(d0):
                h[0,:,a,i] = tmp[i+n*a, :]
        dpars.append( h )
    else:
        for i in range(n):
            h = numpy.zeros((1,np,d0,d1))
            for a in range(d0):
                for b in range(d1):
                    h[0,:,a,b] = tmp[i+n*(b+d1*a),:]
            dpars.append( h )
    return dpars    


# Levenberg_Marquardt
def lm(func, xdata, initp, ydata, Wmat, dfunc, ddfunc=None):
    TolX=1e-8
    Tol=1e-6
    lam = 1e-4
    MAX_ITER=1024
    LAMBDA_MAX=1e8

    [old_chisq, resid] = chisq(func, xdata, initp, ydata, Wmat)
    [grad, hess] = chisq_derivatives(dfunc, ddfunc, xdata, initp, resid, Wmat)
    pars = [ p for p in initp ]

    it=0
    Np = len(initp)
    while it<MAX_ITER:
        step=10.
        
        hess_diag = numpy.diag(numpy.diag(hess))
        alpha = hess + lam * hess_diag
        [u,s,vh] = numpy.linalg.svd(alpha)
        alpha_inv = (vh.conj().T * (1./s)).dot(u.conj().T)

        beta=-grad
        delta_pars = beta.dot( alpha_inv ) 
        new_pars = [ pars[i]+delta_pars[i] for i in range(Np) ]
        
        for i in range(Np):
            if (numpy.fabs(new_pars[i] - pars[i]) < pars[i]*TolX):
                msg = 'Reached %.1e per-cent tolerance on parameter %i ' % (TolX,i)
                break;

        [new_chisq, new_resid] = chisq(func, xdata, new_pars, ydata, Wmat)
        
        if ((new_chisq>old_chisq) and (new_chisq-old_chisq)>(Tol*old_chisq)):
            if (lam < LAMBDA_MAX):
                lam = lam * step
            else:
                print 'Warning: Levenberg-Marquardt: lambda parameter too large: stuck in valley'
        else:
            if ((old_chisq-new_chisq)<old_chisq*Tol):
                msg = 'Reached %.1e per-cent tolerance on chisq' % Tol
                break;
            else:
                lam = lam/step
                pars = new_pars
                old_chisq = new_chisq
                [grad, hess] = chisq_derivatives(dfunc, ddfunc, xdata, pars, new_resid, Wmat)

        it = it + 1
    if (it==MAX_ITER):
        msg = 'Reached Maximum number of iterations %d' % MAX_ITER
    [grad, hess] = chisq_derivatives(dfunc, ddfunc, xdata, pars, new_resid, Wmat)
    return [pars, old_chisq, it, msg, grad]




class pfit:
    def __init__(self,fb, var='x'):

        self.fbasis = []
        for f in fb:
            self.fbasis.append( sympy.lambdify(var, f, 'numpy') )
        self.np = len(fb)

        self.expr = '(' + fb[0] + '*p[0])'
        for i in range(1,self.np):
            self.expr += ' + (' + fb[i] + ('*p[%d])' % i)
        self.func = [ sympy.lambdify('p, ' + var, self.expr, 'numpy') ]
        self.grad = [ sympy.lambdify('p, ' + var, fb, 'numpy') ]
        
    def __str__(self):
        out = 'fbasis = ' + self.expr + '\n'
        out = out + ('npoints = %d ; npars = %d ; dof = %d \n' % (self.n,self.np,self.n-self.np))
        out = out + ('chi2 = %f  pval = %f \n' % (self.chi2,self.pval))
        return out


    def run(self,xdata, obs, W=None, cuts=None, c=None):
        # xdata can be a simple list [4,5,6,...] and this fixes it
        if (numpy.ndim(xdata)==1):
            xdata = numpy.reshape(xdata, (len(xdata),1))

        if (cuts!=None):
            [ydata, wmat, _obs] = _extract_data(obs,W, cuts)
            xax = numpy.array([xdata[i,:] for i in cuts])
            self.n = len(cuts)    
        else:
            [self.n,_] = numpy.shape(xdata)
            [ydata, wmat, _obs] = _extract_data(obs,W, range(self.n))
            xax = numpy.array(xdata)
        
        if (ydata.shape!=(1,1,self.n)):
            raise ValueError('unexpected observables')

        # exact minimization of chi^2    
        xm = []
        for i in range(self.n):
            xm.append([f(*xax[i]) for f in self.fbasis])
        xmat = numpy.array(xm)

        alpha = xmat.transpose().dot(wmat).dot(xmat)
        [u,s,vh] = numpy.linalg.svd(alpha)
        alpha_inv = (vh.conj().T * (1./s)).dot(u.conj().T)
        pp = ydata[0,0].dot(wmat).dot(xmat).dot(alpha_inv)

        # chi2
        tmp = ydata[0,0]-xmat.dot(pp)
        self.chi2 = tmp.dot(wmat).dot(tmp)
        self.pval = gammainc(self.n, self.n-self.np)
    
        if isinstance(obs,observa):
            dfunc = derfit(xax, ydata, wmat, pp, self.func, self.grad, None, False)
        else:
            dfunc = derfit(xax, ydata, wmat, pp, self.func, self.grad, None, True)
        self.pars = derobs(_obs, numpy.array([pp]), dfunc)

    def eval(self,xax):
        """ evaluates the function at given points using the results of the fit

        This function is very useful for plotting the fitted function

        Parameters
        ----------
        xax: list/array or 2D array
            similarly to xdata xax is now the list of points where the function
            is evaluated. For multidimensional fits xax[i,mu] corresponds to the
            mu component of the ith point

        Returns
        -------
        yax: 1D array
            central values of the function evaluated on the points specified by xax. 
        dyax: 1D array
            error of the function evaluated at the points specified by xax

        Examples
        --------
        >>> fit.run(...)
        >>> xax = numpy.arange(0.,4.,0.1) 
        >>> [yax, dyax, _] = fit.eval(xax)
        >>> errorbar(xax, yax[0,0], yerr=dyax[0,0])
        """
        D = xax.ndim
        if (D==1):
            [N] = xax.shape
            xax = numpy.reshape(xax,(N,1))
        else:
            [N,_] = xax.shape
    
        tmp0 = numpy.reshape([self.func[0](self.pars.mean[0,:],*xax[i]) for i in range(N)],(1,N))
        tmp1 = numpy.zeros((1,N,1,self.np))
        for n in range(N):
            tmp1[0,n,0,:] = numpy.array( self.grad[0](self.pars.mean[0,:],*xax[n]) )

        yobs = derobs(self.pars, tmp0, [tmp1])
        [yax, dyax] = yobs.vwerr(simplify=True)
        return [yax, dyax]


class mfit:
    """ Class to fit a function to observables

    Parameters
    ----------
    func : list of list of str or str
        func can be a string or a list of strings containing the definition
        of the function to be fitted. The following syntax must be used in the definition
        of these functions
        
            * for the parameters of the fit use `p[0], p[1], ...`
            
            * for the variable of a 1D fit always use `x`
            
            * for a multi-dimensional fit the user can adopt arbitrary names for the variables such as `a` or `y` or `z`; in this case the argument var must be passed
            
        If the observables that are fitted are vectors or matrices then func must be a
        list of lists of strings, matching exactly the dimensions of the input observables.
        In this case for a pair of indices (i,j) corresponding to the (i,j) components
        of the input observables, func[i][j] is the associated function.
        Note that different components can share the same parameters thus allowing for
        combined fits.
    var : str, optional
        specifies the names of the variables of the fit. For a multi-dimensional fit
        such as func=`p[0]*x + p[1]*y`, the argument var must be `x, y`. Note the comma
        used to separate the variables. If not specified the fit is automatically assumed
        to be 1-dimensional and dependent on the variable `x`. The argument var
        can also be specified in 1D fit, e.g. func=`p[0] + p[1]*y` requires var=`y`

    Examples
    --------
    >>> fit = mfit('p[0] + p[1]*x') # linear 1D fit
    >>> fit = mfit('p[0]*x + p[1]*a', var='x, a') # 2D fit
    >>> fit = mfit('p[0]*x + p[1]*x*y*z', var='x, y, z') # 3D fit
    >>> fit = mfit([['p[0] + p[1]*x', 'p[2] + p[1]*x']]) # combined 1D fit with common slope
    
    """
    
    def __init__(self,func,var='x',debug=False):
        _p = sympy.IndexedBase('p')
        self.pid = []

        if isinstance(func, str):
            func = [[func]]
        if isinstance(func,list) and isinstance(func[0], str):
            func = [[f] for f in func]

        if not isinstance(func,(str,list)):
            raise ValueError

        self.dims_func = numpy.shape(func)
        expr = []
        for i in range(self.dims_func[0]):
            for j in range(self.dims_func[1]):
                tmp = self._parse(func[i][j])
                expr.append( parse_expr(tmp,local_dict={'p':_p}) )

        dexpr = []
        ddexpr = []
        for e in expr:
            dexpr.append( [sympy.diff(e,_p[a]) for a in self.pid] )
            ddexpr.append( [[sympy.diff(dexpr[-1][a],_p[b]) for b in self.pid] for a in self.pid ] )
            if debug:
                print e
                print dexpr[-1]
                print ddexpr[-1]

        self.np = len(self.pid)
        self.expr = func

        self.func = [sympy.lambdify('p, '+var, e, 'numpy')   for e in expr ] 
        self.grad = [sympy.lambdify('p, '+var, de, 'numpy')  for de in dexpr ] 
        self.hess = [sympy.lambdify('p, '+var, dde, 'numpy') for dde in ddexpr ] 

        self.pars = []
        self.chi2 = 0.
        self.pval = 0.
        self.iters = 0

    def _parse(self,func):
        # sanity check
        if (func.find('p[')<0):
            raise ValueError
        # finds parameters
        for i in range(1000):
            if (func.find('p['+str(i)+']')>=0):
                try:
                    self.pid.index(i)
                except:
                    self.pid.append(i)
        return func

    def run(self, xdata, obs, initp=None, W=None, cuts=None):
        """ Finds the minimum of the chi^2 and the parameters of the fit

        Parameters
        ----------
        xdata: list/array of 2D array
            specifies the value of the coordinates where the functions 
            are evaluated. For multidimensional fits the ith data point
            with mu coordinate is specified by xdata[i,mu]
        obs: an observa or a list of observa
            if obs is an observa then it must be a vector with dimensions (1,D);
            in this case the length of the vector specifies the number of points
            in the fit and func must be either a str or [[`function`]]. This feature
            is useful to fit observables defined on the same ensembles, such as correlation
            functions.
            Instead if obs is a list of observa it is assumed that the ith data point
            correspond to the ith element of the list. In this case func must be a 
            list of lists of strings whose dimensions should match the observables.
            The observables are fitted together and a unique chi^2 (which is the sum
            of the chi^2 of each component of the observables) is minimized.
        initp: list/array, optional
            initial values of the parameters
        W: 4D array, optional
            W is a tensor that specifies the weights in the chi^2 for each data point.
            For a pairs of data points (i,j) and for a component (a,b) of the fitted observable
            W[a,b,i,j] specifies the weight of (func[a][b](xdata[i])-obs[i][a,b]) multiplied
            by (func[a][b](xdata[j])-obs[j][a,b]).
            If not given then W is set to 1/sigma_i^2, with sigma_i being the error of the
            (a,b) component of the ith data point (practically an uncorrelated chi^2)
        cuts: list of int, optional
            specifies which data points are included in the fit from xdata and obs. 
            Allowed values go from 0 to the total length of obs/xdata - 1.

        Examples
        --------
        >>> xdata = [0,1,2]
        >>> obs = [obs1, obs2, obs3] # 3 scalar observables defined on 3 ensembles
        >>> fit = mfit('p[0]+p[1]*x')
        >>> fit.run(xdata, obs)
        >>> print fit
        >>> print fit.pars
        
        """

        if (initp==None):
            initp = [1.]*self.np
        else:
            if (len(initp)!=self.np):
                raise ValueError('function and initp do not match')
        
        # xdata can be a simple list [4,5,6,...] and this fixes it
        if (numpy.ndim(xdata)==1):
            xdata = numpy.reshape(xdata, (len(xdata),1))

        if (cuts!=None):
            [ydata, wmat, _obs] = _extract_data(obs,W, cuts)
            xax = numpy.array([xdata[i,:] for i in cuts])
            self.n = len(cuts)    
        else:
            [self.n,_] = numpy.shape(xdata)
            [ydata, wmat, _obs] = _extract_data(obs,W, range(self.n))
            xax = numpy.array(xdata)
        self.n *= numpy.prod(ydata.shape[0:2])

        [pp, self.chi2, self.iters, self.msg, self.final_grad] = lm(self.func, xax, initp, ydata, wmat, self.grad, self.hess)
        self.pval = gammainc(self.n, self.n-self.np)
    
        if isinstance(obs,observa):
            dfunc = derfit(xax, ydata, wmat, pp, self.func, self.grad, self.hess, False)
        else:
            dfunc = derfit(xax, ydata, wmat, pp, self.func, self.grad, self.hess, True)
        self.pars = derobs(_obs, numpy.array([pp]), dfunc)

    def __str__(self):
        out = 'function \n'
        for i in range(self.dims_func[0]):
            for j in range(self.dims_func[1]):
                out += ('(%d,%d) = %s \n' % (i,j,self.expr[i][j]))
        out += ('npoints = %d ; npars = %d ; dof = %d \n' % (self.n,self.np,self.n-self.np))
        out += ('Levenberg-Marquardt algorithm %d iterations \n' % self.iters)
        out += 'Levenberg-Marquardt : ' + self.msg + '\n'
        out += '|dchi2 / dp[i]| :' + str(numpy.fabs(self.final_grad)) + '\n'
        out += ('chi2 = %f  pval = %f \n' % (self.chi2,self.pval))
        return out

    def eval(self,xax):
        """ evaluates the function at given points using the results of the fit

        This function is very useful for plotting the fitted function

        Parameters
        ----------
        xax: list/array or 2D array
            similarly to xdata xax is now the list of points where the function
            is evaluated. For multidimensional fits xax[i,mu] corresponds to the
            mu component of the ith point

        Returns
        -------
        yax: 3D array
            central values of the function evaluated on the points specified by xax. The result is
            always a 3D fit where the first 2 indices correspond to the (a,b) components
            of the function and the third one to the ith point
        dyax: 3D array
            error of the function evaluated at the points specified by xax

        Examples
        --------
        >>> fit.run(...)
        >>> xax = numpy.arange(0.,4.,0.1) 
        >>> [yax, dyax, _] = fit.eval(xax)
        >>> errorbar(xax, yax[0,0], yerr=dyax[0,0])
        
        """
        
        D = xax.ndim
        if (D==1):
            [N] = xax.shape
            xax = numpy.reshape(xax,(N,1))
        else:
            [N,_] = xax.shape
    
        yobs = []
        yax = numpy.zeros(self.dims_func+(N,))
        dyax = numpy.zeros(self.dims_func+(N,))
        
        for n in range(N):
            tmp0 = numpy.zeros(self.dims_func)
            tmp1 = numpy.zeros(self.dims_func+(1,self.np))
            for i in range(self.dims_func[0]):
                for j in range(self.dims_func[1]):
                    fid = i*self.dims_func[1]+j
                    tmp0[i,j] = self.func[fid](self.pars.mean[0,:],*xax[n])
                    tmp1[i,j,0,:] = self.grad[fid](self.pars.mean[0,:],*xax[n])
            yobs.append( derobs(self.pars, tmp0, [tmp1]) )
            [v,e] = yobs[-1].vwerr(simplify=False)
            yax[:,:,n] = v
            dyax[:,:,n] = e
        
        return [yax, dyax]

    def eval_obs(self,xax,var):
        """ Computes the function at a given point using the results of the fit

        Parameters
        ---------
        xax: list
            contains the coordinates where the function is evaluated. If the mu
            coordinate is an observa the errors are propagated accordingly. In this
            case the observa must be a scalar
        var: list
            contains the name of the variables corresponding to elements of xax

        Examples
        --------
        >>> fit.mfit('p[0]*x + p[1]*b')
        >>> fit.run(...)
        >>> x0 = [obs0, 3.4] # the fluctuations of obs0 are propagated
        >>> y0 = fit.eval_obs(x0, ['x', 'b'])
        
        """
        
        xax2 = []
        var2 = []
        xax_obs = observa()
        nx = 0
        for x in xax:
            if isinstance(x,observa):
                if (x.dims!=(1,1)):
                    raise ValueError('unexpected observable')
                xax2.append(x.mean[:,:])
                var2.append( var[xax.index(x)] )
                xax_obs.addcol(x)
                nx += 1
            else:
                xax2.append(x)
        #if (nx==0):
        #    return None

        _var = 'p'
        for v in var:
            _var += ', ' + v

        _p = sympy.IndexedBase('p')
        val = numpy.zeros(self.dims_func)
        grad_pars = numpy.zeros(self.dims_func+(1,self.np))
        grad_xax = numpy.zeros(self.dims_func+(1,nx))
        for i in range(self.dims_func[0]):
            for j in range(self.dims_func[1]):
                fid = i*self.dims_func[1]+j
                val[i,j] = self.func[fid](self.pars.mean[0,:],*xax2)
                grad_pars[i,j,0,:] = self.grad[fid](self.pars.mean[0,:],*xax2)
                
                expr = parse_expr(self.expr[i][j],local_dict={'p':_p})
                for k in range(nx):
                    df = sympy.lambdify(_var, sympy.diff(expr, var2[k]), 'numpy')
                    grad_xax[i,j,0,k] = df(self.pars.mean[0,:], *xax2)

        if (nx==0):
            return derobs([self.pars], val, [grad_pars])

        return derobs([self.pars, xax_obs], val, [grad_pars, grad_xax])

# if obs is an observable then the rows are the functions
# and the columns the data points 
# and ydata[:,0,:] = obs.mean[:,:]
# and W[k*n+i,k*n+i] = diag(1./err[k,i]*2)
# if obs is a list then ydata[a,b,i] will correspond
# to obs[i].mean[a,b] and
# W[k,k] = 1/err[i][a,b] with k=i+n*(b+d1*a)
def _extract_data(obs,W,cuts):
    if isinstance(obs,observa):
        nf = obs.dims[0]
        np = len(cuts)
        ydata = numpy.zeros((nf,1,np))
        for i in range(np):
            ydata[:,0,i] = obs.mean[:,cuts[i]]
        _obs = obs[:,cuts[0]]
        for ic in range(1,np):
            _obs.addcol(obs[:,cuts[ic]])
        if (W==None):
            wmat = numpy.zeros((nf*np,nf*np))
            [_,e] = _obs.vwerr(simplify=False)
            for k in range(nf):
                for i in range(np):
                    wmat[k*np+i, k*np+i] = 1./e[k,i]**2
        else:
            wmat = W
    elif isinstance(obs,list):
        (d0, d1) = obs[0].dims
        np = len(cuts)
        ydata = numpy.zeros((d0,d1,np))
        wmat = numpy.zeros((d0*d1*np,d0*d1*np))
        _obs = [obs[cuts[0]]]
        for ic in range(1,np):
            _obs.append( obs[cuts[ic]] )
        for i in range(np):
            if (W==None):
                [_,e] = obs[cuts[i]].vwerr(simplify=False)
                for a in range(d0):
                    for b in range(d1):
                        wmat[i+np*(b+d1*a), i+np*(b+d1*a)] = 1./e[a,b]**2
            else:
                wmat[i+np*(b+d1*a), i+np*(b+d1*a)] = W[i+np*(b+d1*a), i+np*(b+d1*a)]
            ydata[:,:,i] = obs[cuts[i]].mean
    else:
        raise ValueError('unexpected input')
    
    return [ydata, wmat, _obs]


# if obs is an observable then it must be a vector
# and new_grad = grad
# if obs is a list then ydata[a,b,i] will correspond
# to new_grad[i][0,:,a,b]
def _reshape_gradient(obs, grad):
    if isinstance(obs,observa):
        return grad
    elif isinstance(obs,list):
        new_grad = []
        i0=0
        dims = grad.shape
        for o in obs:
            i1 = i0 + o.dims[1]
            new_grad.append( numpy.reshape(grad[0,:,0,slice(i0,i1)],dims[0:3]+(i1-i0,)) )
            i0 = i1
    return new_grad

