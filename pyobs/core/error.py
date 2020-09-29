#################################################################################
#
# error.py: definition and properties of the error classes and functions
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
from scipy import special

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB=True
except:
    MATPLOTLIB=False

def find_closest(arr,x0):
    dx=min(numpy.diff(numpy.array(arr)-x0))
    if isinstance(arr,list):
        return array.index(x0+dx)
    elif isinstance(arr,numpy.ndarray):
        return numpy.where(arr==dx+x0)[0][0]

class variance:
    def __init__(self,n,g,Stau,D,k,fold=False):
        self.D=D
        self.k=k
        self.Stau=Stau
        z=(D)*0.5
        self.OmegaD=numpy.pi**(z)/special.gamma(z)*2
        
        (s,xmax) = numpy.shape(g)
        self.x=[]
        for i in range(xmax):
            if n[i]>0:
                if D==1:
                    self.x.append(i)
                else:
                    self.x.append(numpy.sqrt(i))
        
        gg=numpy.zeros((s,len(self.x)))
        j=0
        for i in range(xmax):
            if n[i]>1e-15:                   
                gg[:,j] = g[:,i]/n[i]
                j+=1
        
        self.size = s
        if fold:
            _gg = numpy.array(gg)
            _gg[:,1:] *= 2.0
            self.cvar = numpy.cumsum(_gg,axis=1)
        else:
            self.cvar = numpy.cumsum(gg,axis=1)
        self.N = n[0]
        self.xopt = [self.x[-1]]*self.size
        self.var = numpy.zeros((self.size,2))

    def g(self,i,a):
        cov=self.cvar[a,i]/self.cvar[a,0]
        j=self.D-self.k
        xi=self.x[i]
        
        am=(special.gamma(j)*self.OmegaD/cov)**(1./j)/float(self.Stau)
        h = -numpy.exp(-am*xi) * xi**(j-1) * am**(j) / special.gamma(j)
        return h + numpy.sqrt(self.D*self.OmegaD*0.5/self.N * xi**(self.D-2))
    
    def find_opt(self):
        for a in range(self.size):
            for i in range(1,len(self.x)):
                if self.g(i,a)>0:
                    self.xopt[a] = self.x[i]
                    self.var[a,0] = self.cvar[a,i]
                    self.var[a,1] = self.cvar[a,i] * self.stat_relerr(self.x[i])
                    break
            if self.xopt[a]==self.x[-1]:
                self.var[a,0] = self.cvar[a,-1]
                self.var[a,1] = self.cvar[a,-1] * self.stat_relerr(self.x[-1])
                print(f'Warning: automatic window failed, using {self.xopt[a]}')

    def set_opt(self,xopt):
        for a in range(self.size):
            if isinstance(xopt,int): 
                if xopt in self.x:
                    i=self.x.index(xopt)
                else:
                    i=find_closest(self.x,xopt)
            else:
                if xopt[a] in self.x:
                    i=self.x.index(xopt[a])
                else:
                    i=find_closest(self.x,xopt[a])
                
            self.xopt[a] = self.x[i]
            self.var[a,0] = self.cvar[a,i]
            self.var[a,1] = self.cvar[a,i] * self.stat_relerr(self.x[i])

    def stat_relerr(self,r):
        return numpy.sqrt(2.*self.OmegaD/(self.D*self.N) * numpy.array(r)**self.D)
    
    def correct_gamma_bias(self):
        for a in range(self.size):
            f = self.OmegaD * (self.xopt[a]**self.D) / (self.N*self.D)
            self.var[a] += self.var[a] * f
            self.cvar[a,:] += self.cvar[a,:]*f

    def tauint(self,full_size,mask):
        tau = numpy.zeros((full_size,2))
        for a in mask:
            i=mask.index(a)
            tau[a,0] = self.var[i,0]/self.cvar[i,0]
            tau[a,1] = self.var[i,1]/self.cvar[i,0]
        return tau    
    
    def reshape(self,full_size,mask):
        out0 = numpy.zeros((full_size,))
        out1 = numpy.zeros((full_size,))
        for a in mask:
            out0[a] = self.var[mask.index(a),0] / self.N
            out1[a] = self.var[mask.index(a),1] / self.N
        return [out0, out1]
    
    def plot(self,xlab,desc,ed,pfile): # pragma: no cover
        if not MATPLOTLIB:
            pass

        err=self.stat_relerr(self.x)
        
        for a in range(self.size):
            plt.figure()
            plt.title(f'{desc}; {ed}; {a}')
            plt.xlabel(xlab)
            plt.ylabel('covariance')
            
            y = self.cvar[a,:]/self.cvar[a,0]
            plt.plot(self.x,y,'-',color='C0')
            plt.fill_between(self.x,y*(1.+err),y*(1.-err),alpha=.3,color='C0')
            plt.plot([0,self.x[-1]],[0,0],'-k',lw=.75)        
            plt.plot([self.xopt[a]]*2,[0,self.var[a,0]/self.cvar[a,0]],'-',color='C1',label=f'opt={self.xopt[a]}')
            plt.xlim(0,self.xopt[a]*2)
            dy = self.var[a,1]/self.cvar[a,0]
            plt.ylim(1-dy*2,self.var[a,0]/self.cvar[a,0]+dy*3)
            plt.legend(loc='upper right')
            plt.show()

def plot_piechart(desc,errs,tot): # pragma: no cover
    if not MATPLOTLIB:
        pass

    n = numpy.reciprocal(tot)
    s=numpy.size(tot)
    x = []
    for v in errs.values():
        x.append( numpy.reshape(v*n,(s,)) )
    x=numpy.array(x)
    for a in range(s):
        plt.figure()
        plt.title(f'{desc}; {a}')
        plt.pie(x[:,a], labels=errs.keys(), autopct='%.0f%%', radius=1.0)
        plt.axis('equal')
        plt.show()
        
class mfvar(variance):
    def __init__(self,x,mfd,k,Stau):
        keys=[]
        for rn in x.mfdata:
            if rn.split(':')[0]==mfd:
                keys.append(rn)
        rrmax=x.mfdata[keys[0]].rrmax() #int(min([x.mfdata[kk].rrmax() for kk in keys]))
        size=len(x.mfdata[keys[0]].mask) # we assume all replica have the same mask

        [n, g]=x.mfdata[keys[0]].gamma(rrmax)
        for i in range(1,len(keys)):
            res = x.mfdata[keys[i]].gamma(rrmax)
            n += res[0]
            g += res[1]

        D=len(x.mfdata[keys[0]].lat)
        variance.__init__(self,n,g,Stau,D,k)
        self.full_size = x.size
        self.mask = x.mfdata[keys[0]].mask
        
    def sigma(self):
        return self.reshape(self.full_size,self.mask)
    
    def tau(self):
        return self.tauint(self.full_size,self.mask)
    
class rvar(variance):
    def __init__(self,x,ed,Stau):
        keys=[]
        for rn in x.rdata:
            if rn.split(':')[0]==ed:
                keys.append(rn)
        wmax=int(min([x.rdata[k].wmax() for k in keys]))
        size=len(x.rdata[keys[0]].mask) # we assume all replica have the same mask

        [n, g]=x.rdata[keys[0]].gamma(wmax)
        for i in range(1,len(keys)):
            res = x.rdata[keys[i]].gamma(wmax)
            n += res[0]
            g += res[1]
        
        variance.__init__(self,n,g,Stau,1,0,fold=True)
        self.full_size = x.size
        self.mask = x.rdata[keys[0]].mask
    
    def sigma(self):
        return self.reshape(self.full_size,self.mask)
    
    def tau(self):
        return self.tauint(self.full_size,self.mask)

class errinfo:
    def __init__(self,Stau=1.5,k=0,W=None,R=None):
        self.Stau = Stau
        self.k = k
        self.W = W
        self.R = R
        
def uwerr(x,ed,plot=False,pfile=None,Stau=1.5,W=None):
    rv = rvar(x,ed,Stau)
    if W is None:
        rv.find_opt()
    else:
        rv.set_opt(W)
    rv.correct_gamma_bias()
    tau = rv.tau()*0.5
    if plot: # pragma: no cover
        rv.plot('icnfg',x.description,ed,pfile)
    return rv.sigma() + [tau]

def mferr(x,mfd,plot=False,pfile=None,k=1,Stau=1.5,R=None):
    mfv = mfvar(x,mfd,k,Stau)
    if R is None:
        mfv.find_opt()
    else:
        mfv.set_opt(R)
    #rv.correct_gamma_bias()
    tau = mfv.tau()
    if plot: # pragma: no cover
        mfv.plot('|R|/a',x.description,mfd,pfile)
    return mfv.sigma() + [tau]
