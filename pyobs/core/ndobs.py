################################################################################
#
# ndobs.py: definition and properties of the core class of the library
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
import copy
import gzip, json
import os, pwd, re
from time import time
import datetime

import pyobs

from .data import delta
from .cdata import cdata
from .error import gamma_error, plot_piechart

__all__ = ['observable']

class observable:
    """
    Class defining an observable

    Parameters:
       orig (observable, optional): creates a copy of orig
       description (str, optional): description of the observable
    
    Examples:
       >>> from pyobs import observable
       >>> a = observable(description='test')
    """
    
    def __init__(self,orig=None,description='unknown'):
        if orig is None:
            pyobs.check_type(description,'text',str)
            self.description = description
            self.www = [pwd.getpwuid(os.getuid())[0], os.uname()[1], datetime.datetime.now().strftime('%c')]
            self.shape = []
            self.size = 0
            self.mean = []
            self.ename = []
            self.delta = {}
            self.cdata = {}
        else:
            if isinstance(orig,observable):
                self.description = orig.description
                self.www = orig.www
                self.shape = orig.shape
                self.size = numpy.prod(self.shape)
                self.mean = numpy.array(orig.mean) # or orig.mean.copy()
                
                self.ename = [e for e in orig.ename]
                self.delta = {}
                for key in orig.delta:
                    self.delta[key] = orig.delta[key].copy()
                                
                self.cdata = {}
                for key in orig.cdata:
                    self.cdata[key] = cdata(orig.cdata[key].cov)
                pyobs.memory.add(self)
            else:
                raise pyobs.PyobsError('Unexpected orig argument')
        pyobs.memory.add(self)
        
    def create(self,ename,data,icnfg=None,rname=None,shape=(1,),lat=None):
        """
        Create an observable
        
        Parameters:
           ename (str): label of the ensemble
           data (array, list of arrays): the data generated from a single 
              or multiple replica
           icnfg (array of ints or list of arrays of ints, optional): indices 
              of the configurations corresponding to data; if not passed the 
              measurements are assumed to be contiguous
           rname (str or list of str, optional): identifier of the replica; if 
              not passed integers from 0 are automatically assigned
           shape (tuple, optional): shape of the observable, data must be passed accordingly
           lat (list of ints, optional): the size of each dimension of the master-field;
              if passed data is assumed to be obtained from observables measured at different
              sites and `icnfg` is re-interpreted as the index labeling the sites; if `icnfg`
              is not passed data is assumed to be contiguous on all sites.
              
        Note:
           For data and icnfg array can mean either a list or a 1-D numpy.array.
           If the observable has already been created, calling create again will add
           a new replica to the same ensemble.
           
        Examples:
           >>> data = [0.43, 0.42, ... ] # a scalar observable
           >>> a = pyobs.observable(description='test')
           >>> a.create('EnsembleA',data)

           >>> data0 = [0.43,0.42, ... ] # replica 0
           >>> data1 = [0.40,0.41, ... ] # replica 1
           >>> a = pyobs.observable(description='test')
           >>> a.create('EnsembleA',[data0,data1],rname=['r0','r1'])

           >>> data = [0.43, 0.42, 0.44, ... ]
           >>> icnfg= [  10,   11,   13, ... ]
           >>> a = pyobs.observable(description='test')
           >>> a.create('EnsembleA',data,icnfg=icnfg)

           >>> data = [1.0, 2.0, 3.0, 4.0, ... ]
           >>> a = pyobs.observable(description='matrix')
           >>> a.create('EnsembleA',data,shape=(2,2))
       
        Examples:
           >>> data = [0.43, 0.42, 0.44, ... ]
           >>> lat = [64,32,32,32]
           >>> a = pyobs.observable(description='test-mf')
           >>> a.create('EnsembleA',data,lat=lat)
           
           >>> data = [0.43, 0.42, 0.44, ... ]
           >>> idx = [0, 2, 4, 6, ...] # measurements on all even points of time-slice
           >>> lat = [32, 32, 32]
           >>> a = pyobs.observable(description='test-mf')
           >>> a.create('EnsembleA',data,lat=lat,icnfg=idx)           
        """
        t0=time()
        pyobs.check_type(ename,'ename',str)
        if ':' in ename:
            raise pyobs.PyobsError(f'Column symbol not allowed in ename {ename}')
        pyobs.check_type(shape,'shape',tuple)
        self.shape = shape
        self.size=numpy.prod(shape)
        mask=range(self.size)
        if not ename in self.ename:
            self.ename.append(ename)
        
        if isinstance(data[0],(list,numpy.ndarray)):
            R=len(data)
        elif isinstance(data[0],(int,float,numpy.float64,numpy.float32)):
            R=1
        else:
            raise pyobs.PyobsError(f'Unexpected data type')
            
        if R==1:
            pyobs.check_type(data,f'data',list,numpy.ndarray)
            nc=int(len(data)/self.size)
            if rname is None:
                rname=0
            else:
                pyobs.check_not_type(rname,'rname',list)
            if icnfg is None:
                icnfg=range(nc)
            else:
                pyobs.check_type(icnfg,'icnfg',list,range)
                pyobs.check_type(icnfg[0],'icnfg[:]',int,numpy.int32,numpy.int64)
                if len(icnfg)*self.size!=len(data):
                    raise pyobs.PyobsError(f'Incompatible icnfg and data, for shape={shape}')
            if numpy.size(self.mean)!=0:
                N0 = sum([self.delta[key].n for key in self.delta])
                mean0 = numpy.reshape(self.mean,(self.size,))
                mean1 = numpy.mean(numpy.reshape(data,(nc,self.size)),0)
                self.mean = (N0*mean0 + nc*mean1)/(N0+nc)
                shift = nc*(mean0-mean1)/(N0+nc)
                for key in self.delta:
                    self.delta[key].delta += shift[:,None]
            else:
                self.mean=numpy.mean(numpy.reshape(data,(nc,self.size)),0)
                
            key=f'{ename}:{rname}'
            self.delta[key] = delta(mask, icnfg, data, self.mean, lat)
        else:
            if numpy.size(self.mean)!=0:
                raise pyobs.PyobsError('Only a single replica can be added to existing observables')
            for ir in range(R):
                pyobs.check_type(data[ir],f'data[{ir}]',list,numpy.ndarray)
            self.mean=numpy.zeros((self.size,))
            nt=0
            for dd in data:
                nc=int(len(dd)/self.size)
                self.mean += numpy.sum(numpy.reshape(dd,(nc,self.size)),0)
                nt+=nc
            self.mean *= 1.0/float(nt)
            if rname is None:
                rname = range(R)
            else:
                pyobs.check_type(rname,'rname',list)
                if len(rname)!=R:
                    raise pyobs.PyobsError('Incompatible rname and data')
            if not icnfg is None:
                pyobs.check_type(icnfg,'icnfg',list)
            
            if icnfg is None:
                icnfg = []
                for ir in range(len(data)):
                    nc=int(len(data[ir])/self.size)
                    icnfg.append(range(nc))
            else:
                for ir in range(len(data)):
                    if len(icnfg[ir])*self.size!=len(data[ir]):
                        raise pyobs.PyobsError(f'Incompatible icnfg[{ir}] and data[{ir}], for shape={shape}')
            for ir in range(len(data)):
                key=f'{ename}:{rname[ir]}'
                self.delta[key] = delta(mask, icnfg[ir], data[ir], self.mean, lat)
        self.mean = numpy.reshape(self.mean, self.shape)
        pyobs.memory.update(self)
        if pyobs.is_verbose('create'):
            print(f'create executed in {time()-t0:g} secs')

        
    def create_from_cov(self,cname,value,covariance):
        """
        Create observables based on covariance matrices
        
        Parameters:
           cname (str): label that uniquely identifies the data set
           value (array): central value of the observable; only 1-D arrays are supported
           covariance (array): a 2-D covariance matrix; if `covariance` is a 1-D array of
              same length as `value`, a diagonal covariance matrix is assumed.
        
        Examples:
           >>> mpi = pyobs.observable(description='pion masses, charged and neutral')
           >>> mpi.create_cd('mpi-pdg18',[139.57061,134.9770],[0.00023**2,0.0005**2])
           >>> print(mpi)
           139.57061(23)    134.97700(50)
        """
        if isinstance(value,(int,float,numpy.float64,numpy.float32)):
            self.mean = numpy.reshape(value,(1,))
            cov = numpy.reshape(covariance,(1,))
        else:
            self.mean = numpy.array(value)
            cov = numpy.array(covariance)
        self.shape = numpy.shape(self.mean)
        if numpy.ndim(self.shape)!=1:
            raise pyobs.PyobsError(f'Unexpected value, only 1-D arrays are supported')
        self.size = numpy.prod(self.shape)
        if cov.shape!=(self.size,) and cov.shape!=(self.size,self.size):
            raise pyobs.PyobsError(f'Unexpected shape for covariance {cov.shape}')
        pyobs.check_type(cname,'cname',str)
        self.cdata[cname] = cdata(cov)
        pyobs.memory.update(self)
        
    def add_syst_err(self,name,err):
        """
        Add a systematic error to the observable
        
        Parameters:
           name (str): label that uniquely identifies the syst. error
           err (array): array with the same dimensions of the observable
              with the systematic error
        
        Examples:
           >>> data = [0.198638, 0.403983, 1.215960, 1.607684, 0.199049, ... ]
           >>> vec = pyobs.observable(description='vector')
           >>> vec.create('A',data,shape=(4,))
           >>> print(vec)
           0.201(13)    0.399(26)    1.199(24)    1.603(47)
           >>> vec.add_syst_err('syst.err',[0.05,0.05,0,0])
           >>> print(vec)
           0.201(52)    0.399(56)    1.199(24)    1.603(47)
           
        """
        pyobs.check_type(name,'name',str)
        if name in self.cdata:
            raise pyobs.PyobsError(f'Label {name} already used')
        if numpy.shape(err)!=self.shape:
            raise pyobs.PyobsError(f'Unexpected error, dimensions do not match {self.shape}')
        cov = numpy.reshape(numpy.array(err)**2, (self.size,))
        self.cdata[name] = cdata(cov)
        pyobs.memory.update(self)        
        
    def __del__(self):
        pyobs.memory.rm(self)
        
    ##################################
    # pretty string representations
        
    def peek(self):
        """
        Display a brief summary of the content of the observable, including 
        its memory footprint and requirements (for error computation), its 
        description and ensemble/replica content
        
        Example:
           >>> obs.peek()
           Observable with shape = (1, 4)
            - description: vector-test
            - size: 82 KB
            - mean: [[0.20007161 0.40085252 1.19902686 1.60184989]]
            - Ensemble A
               - Replica 0 with mask [0, 1, 2, 3] and ncnfg 500
                    temporary additional memory required 0.015 MB

        """  
        print(f'Observable with shape = {self.shape}')
        print(f' - description: {self.description}')
        print(f' - size: {pyobs.memory.get(self)}')
        print(f' - mean: {self.mean}')
        
        for name in self.ename:
            print(f' - Ensemble {name}')
            m=0
            for key in self.delta:
                rn=key.split(':')
                if rn[0]==name:
                    outstr = f'    - {"Replica" if self.delta[key].lat is None else "Master-field"} {rn[1]}'
                    outstr = f'{outstr} with {"ncnfg" if self.delta[key].lat is None else "sites"} {self.delta[key].n}'
                    print(outstr)
                    mm=self.delta[key].ncnfg()*8.*2. if self.delta[key].lat is None else (self.delta[key].vol()+1)*8.
                    m=(mm>m)*mm + (mm<=m)*m
            print(f'         temporary additional memory required {m/1024.**2:.2g} MB')
        
        for cd in self.cdata:
            print(f' - Data {cd} with cov. matrix {self.cdata[cd].cov.shape}')
        print('')
    
    def __str__(self):
        [v,e] = self.error()
        D=len(self.shape)
        out = ''
        if D==1:
            out += '\t'.join([pyobs.valerr(v[i],e[i]) for i in range(self.shape[0])])
            out += '\n'
        elif D==2:
            for i in range(self.shape[0]):
                out += '\t'.join([pyobs.valerr(v[i,j],e[i,j]) for j in range(self.shape[1])])
                out += '\n' 
        return out
    
    def __repr__(self): # pragma: no cover
        return self.__str__()
    
    ##################################
    # overloaded indicing and slicing

    def set_mean(self, mean):
        if isinstance(mean,(int,float,numpy.float32,numpy.float64)):
            self.mean = numpy.reshape(mean,(1,))
        else:
            self.mean = numpy.array(mean)
        self.shape = numpy.shape(mean)
        self.size = numpy.size(mean)
        
    def slice(self,*args):
        na=len(args)
        if na!=len(self.shape):
            raise pyobs.PyobsError('Unexpected argument')
        f = lambda x: pyobs.slice_ndarray(x, *args)
        g0 = pyobs.gradient(f, self.mean, gtype='slice')
        return pyobs.derobs([self], f(self.mean), [g0])
    
    def __getitem__(self,args):
        if isinstance(args,(int,numpy.int32,numpy.int64,slice,numpy.ndarray)):
            args=[args]
        na=len(args)
        if na!=len(self.shape):
            raise pyobs.PyobsError('Unexpected argument')
        if self.mean[tuple(args)].size==1:
            f = lambda x: numpy.reshape(x[tuple(args)],(1,))
        else:
            f = lambda x: x[tuple(args)]
        g0 = pyobs.gradient(f, self.mean, gtype='slice')
        return pyobs.derobs([self], f(self.mean), [g0])
    
    def __setitem__(self,args,yobs):
        if isinstance(args,(int,numpy.int32,numpy.int64,slice,numpy.ndarray)):
            args=[args]
        if (self.mean[tuple(args)].size==1):
            if yobs.size!=1:
                raise pyobs.PyobsError('set item : dimensions do not match')
        else:
            if self.mean[tuple(args)].shape != yobs.shape:
                raise pyobs.PyobsError('set item : dimensions do not match')
        self.mean[tuple(args)] = yobs.mean

        idx = numpy.arange(self.size).reshape(self.shape)[tuple(args)]
        submask = idx.flatten()
        
        for key in yobs.delta:
            if not key in self.delta:
                raise pyobs.PyobsError('Ensembles do not match; can not set item')
            self.delta[key].assign(submask,yobs.delta[key])

        for key in yobs.cdata:
            if not key in self.cdata:
                raise pyobs.PyobsError('Covariance data do not match; can not set item')
            self.cdata[key].assign(submask,yobs.cdata[key])
        
    ##################################
    # overloaded basic math operations

    def __addsub__(self,y,sign):
        g0 = pyobs.gradient(lambda x: x, self.mean, gtype='diag')
        if isinstance(y,observable):
            g1 = pyobs.gradient(lambda x: sign*x, y.mean, gtype='diag')
            return pyobs.derobs([self,y],self.mean+sign*y.mean,[g0,g1])
        else:
            return pyobs.derobs([self],self.mean+sign*y,[g0])
    
    def __add__(self,y):
        return self.__addsub__(y,+1)
    
    def __sub__(self,y):
        return self.__addsub__(y,-1)
    
    def __neg__(self):
        g0 = pyobs.gradient(lambda x: -x, self.mean, gtype='diag')
        return pyobs.derobs([self],-self.mean,[g0])
    
    def __mul__(self,y):
        if isinstance(y,observable):
            if self.shape==y.shape:
                g0 = pyobs.gradient(lambda x: x*y.mean, self.mean, gtype='diag')
                g1 = pyobs.gradient(lambda x: self.mean*x, y.mean, gtype='diag')
            elif self.shape==(1,):
                g0 = pyobs.gradient(lambda x: x*y.mean, self.mean, gtype='full')
                g1 = pyobs.gradient(lambda x: self.mean*x, y.mean, gtype='diag')
            elif y.shape==(1,):
                g0 = pyobs.gradient(lambda x: x*y.mean, self.mean, gtype='diag')
                g1 = pyobs.gradient(lambda x: self.mean*x, y.mean, gtype='full')
            else:
                raise pyobs.PyobsError('Shape mismatch, cannot multiply')
            return pyobs.derobs([self,y],self.mean*y.mean,[g0,g1])
        else:
            # if gradient below was 'full' it would allow scalar_obs * array([4,5,6])
            # which would create a vector obs. right now that generates an error
            # but is faster for large gradients
            g0 = pyobs.gradient(lambda x: x*y, self.mean, gtype='diag')
            return pyobs.derobs([self],self.mean*y,[g0])
    
    def __matmul__(self,y):
        if isinstance(y,observable):
            g0 = pyobs.gradient(lambda x: x @ y.mean, self.mean)
            g1 = pyobs.gradient(lambda x: self.mean @ x, y.mean)
            return pyobs.derobs([self,y],self.mean @ y.mean,[g0,g1])
        else:
            g0 = pyobs.gradient(lambda x: x @ y, self.mean)
            return pyobs.derobs([self],self.mean @ y,[g0])

    def reciprocal(self):
        new_mean = numpy.reciprocal(self.mean)
        g0 = pyobs.gradient(lambda x: -x*(new_mean**2), self.mean, gtype='diag')
        return pyobs.derobs([self], new_mean,[g0])
    
    def __truediv__(self,y):
        if isinstance(y,observable):
            return self * y.reciprocal()
        else:
            return self * (1 / y)
   
    #__array_priority__=1000
    __array_ufunc__ = None
    def __radd__(self,y):
        return self+y
    def __rsub__(self,y):
        return -self+y
    def __rmul__(self,y):
        return self*y
    def __rtruediv__(self,y):
        return self.reciprocal() * y
    
    def __pow__(self,a):
        new_mean = self.mean**a
        g0 = pyobs.gradient(lambda x: a * x*self.mean**(a-1), self.mean, gtype='diag')
        return pyobs.derobs([self], new_mean, [g0])
    
    # in-place operations
    def __iadd__(self,y):
        tmp = self + y
        del self
        self = pyobs.observable(tmp)
        del tmp
        return self
    
    def __isub__(self,y):
        tmp = self - y
        del self
        self = pyobs.observable(tmp)
        del tmp
        return self
    
    def __imul__(self,y):
        tmp = self * y
        del self
        self = pyobs.observable(tmp)
        del tmp
        return self
    
    def __itruediv__(self,y):
        tmp = self / y
        del self
        self = pyobs.observable(tmp)
        del tmp
        return self


    ##################################
    # Error functions
    
    def error_core(self,errinfo,plot,pfile):
        sigma_tot = numpy.zeros(self.shape)
        dsigma_tot = numpy.zeros(self.shape)
        sigma = {}
        for e in self.ename:
            if e in errinfo:
                res = gamma_error(self,e,plot,pfile,errinfo[e])
            else:
                res = gamma_error(self,e,plot,pfile)
            sigma[e] = numpy.reshape(res[0],self.shape)
            sigma_tot += sigma[e]
            dsigma_tot += numpy.reshape(res[1],self.shape)
                    
        for cd in self.cdata:
            sigma[cd] = numpy.reshape(self.cdata[cd].sigmasq(),self.shape)
            sigma_tot += sigma[cd]
        return [sigma, sigma_tot, dsigma_tot]
    
    def error(self,errinfo={},plot=False,pfile=None):
        """
        Estimate the error of the observable, by summing in quadrature
        the systematic errors with the statistical errors computed from 
        all ensembles and master fields.
        
        Parameters:
           errinfo (dict, optional): dictionary containg one instance of 
              the errinfo class for each ensemble/master-field. The errinfo
              class provides additional details for the automatic or manual
              windowing procedure in the Gamma method. If not passed default
              parameters are assumed.
           plot (bool, optional): if specified a plot is produced, for 
              every element of the observable, and for every ensemble/master-field 
              where the corresponding element has fluctuations. In addition 
              one piechart plot is produced for every element, showing the 
              contributions to the error from the various sources, only if there
              are multiple sources, ie several ensembles. It is recommended to use 
              the plotting function only for observables with small dimensions.
           pfile (str, optional): if specified all plots produced with the flag
              `plot` are saved to disk, using `pfile` as base name with an additional
              suffix.
        
        Returns:
           list of two arrays: the central value and error of the observable.
           
        Note:
           By default, the errors are computed with the Gamma method, with the `Stau` 
           parameter equal to 1.5. Additionally the jackknife method can be used 
           by passing the appropriate `errinfo` dictionary with argument `bs` set
           to a non-zero integer value. For master fields the error is computed 
           using the master-field approach and the automatic windowing procedure
           requires the additional argument `k` (see main documentation), which 
           by default is zero, but can be specified via the errinfo dictionary.
           Through the `errinfo` dictionary the user can treat every ensemble 
           differently, as explained in the examples below.
           
           
        Examples:
           >>> obsA = pyobs.observable('obsA')
           >>> obsA.create('A',dataA) # create the observable A from ensemble A
           >>> [v,e] = obsA.error() # collect central value and error in v,e
           >>> einfo = {'A': errinfo(Stau=3.0)} # specify non-standard Stau for ensemble A
           >>> [_,e1] = obsA.error(errinfo=einfo)
           >>> print(e,e1) # check difference in error estimation
        
           >>> obsB = pyobs.observable('obsB')
           >>> obsB.create('B',dataB) # create the observable B from ensemble B
           >>> obsC = obsA * obsB # derived observable with fluctuations from ensembles A,B
           >>> einfo = {'A': errinfo(Stau=3.0), 'B': errinfo(W=30)}
           >>> [v,e] = obsC.error(errinfo=einfo,plot=True)
        """
        t0=time()
        [sigma, sigma_tot, _] = self.error_core(errinfo,plot,pfile)
        
        if plot: # pragma: no cover
            h=[len(self.ename),len(self.cdata)]
            if sum(h)>1:
                plot_piechart(self.description, sigma, sigma_tot)
            
        if pyobs.is_verbose('error'):
            print(f'error executed in {time()-t0:g} secs')
        return [self.mean, numpy.sqrt(sigma_tot)]

    def error_of_error(self,errinfo={}):
        """
        Returns the error of the error based on the analytic 
        prediction obtained by U. Wolff.
        
        Parameters:
           errinfo (dict, optional): see the documentation of the `error`
              method.
        
        Returns:
           array: the error of the error
        """
        [_, _, dsigma_tot] = self.error_core(errinfo,False,None)
        return numpy.sqrt(dsigma_tot)
        
    
    def tauint(self):
        """
        Estimates the integrated autocorrelation time and its error for every
        ensemble, with the automatic windowing procedure.

        Notes:
           To be added in future versions: support for arbitrary values of Stau
        """
        # to be improved - add errinfo
        tau = {}
        for e in self.ename:
            res = gamma_error(self,e)
            tau[e] = [numpy.reshape(res[2][:,0],self.shape), numpy.reshape(res[2][:,1],self.shape)]
        
        return tau
    