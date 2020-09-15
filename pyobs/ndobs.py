#################################################################################
#
# obs.py: definition and properties of the core class of the library
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

from pyobs.core.utils import *
from pyobs.core.data import rdata, mfdata
from pyobs.core.cdata import cdata
from pyobs.core.derobs import derobs
from pyobs.core.error import uwerr, mferr, plot_piechart
from pyobs.core import memory
from pyobs.tensor.unary import unary_grad

class obs:
    """
    Class defining an observable

    Parameters:
       orig (obs, optional): creates a copy of orig
       desc (str, optional): description of the observable
    
    Examples:
       >>> from pyobs import obs
       >>> a = obs(desc='test')
    """
    
    def __init__(self,orig=None,desc='unknown'):
        if orig is None:
            check_type(desc,'text',str)
            self.description = desc
            self.www = [pwd.getpwuid(os.getuid())[0], os.uname()[1], datetime.datetime.now().strftime('%c')]
            self.dims = []
            self.size = 0
            self.mean = []
            self.edata = []
            self.rdata = {}
            self.mfname = []
            self.mfdata = {}
            self.cdata = {}
        else:
            if isinstance(orig,obs):
                self.description = orig.description
                self.www = orig.www
                self.dims = orig.dims
                self.size = numpy.prod(self.dims)
                self.mean = numpy.array(orig.mean) # or orig.mean.copy()
                
                self.edata = [e for e in orig.edata] #copy.deepcopy(orig.edata) # copy works only for primitive types, not lists
                self.rdata = {}
                for key in orig.rdata:
                    self.rdata[key] = orig.rdata[key].copy()
                    
                self.mfname = [n for n in orig.mfname]
                self.mfdata = {}
                for key in orig.mfdata:
                    self.mfdata[key] = orig.mfdata[key].copy()
                
                self.cdata = {}
                for key in orig.cdata:
                    self.cdata[key] = cdata(orig.cdata[key].grad,orig.cdata[key].cov)
                memory.add(self)
            else:
                error_msg('Unexpected orig argument')
    
    def create(self,ename,data,icnfg=None,rname=None,dims=(1,),lat=None):
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
           dims (tuple, optional): dimensions of the observable, data must 
              be passed accordingly
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
           >>> a = pyobs.obs(desc='test')
           >>> a.create('EnsembleA',data)

           >>> data0 = [0.43,0.42, ... ] # replica 0
           >>> data1 = [0.40,0.41, ... ] # replica 1
           >>> a = pyobs.obs(desc='test')
           >>> a.create('EnsembleA',[data0,data1],rname=['r0','r1'])

           >>> data = [0.43, 0.42, 0.44, ... ]
           >>> icnfg= [  10,   11,   13, ... ]
           >>> a = pyobs.obs(desc='test')
           >>> a.create('EnsembleA',data,icnfg=icnfg)

           >>> data = [1.0, 2.0, 3.0, 4.0, ... ]
           >>> a = pyobs.obs(desc='matrix')
           >>> a.create('EnsembleA',data,dims=(2,2))
       
        Examples:
           >>> data = [0.43, 0.42, 0.44, ... ]
           >>> lat = [64,32,32,32]
           >>> a = pyobs.obs(desc='test-mf')
           >>> a.create('EnsembleA',data,lat=lat)
           
           >>> data = [0.43, 0.42, 0.44, ... ]
           >>> idx = [0, 2, 4, 6, ...] # measurements on all even points of time-slice
           >>> lat = [32, 32, 32]
           >>> a = pyobs.obs(desc='test-mf')
           >>> a.create('EnsembleA',data,lat=lat,icnfg=idx)           
        """
        t0=time()
        check_type(ename,'ename',str)
        if ':' in ename:
            error_msg(f'Column symbol not allowed in ename {ename}')
        check_type(dims,'dims',tuple)
        self.dims = dims
        self.size=numpy.prod(dims)
        mask=range(self.size)
        if lat is None:
            if not ename in self.edata:
                self.edata.append(ename)
        else:
            if not ename in self.mfname:
                self.mfname.append(ename)
        
        if isinstance(data[0],(list,numpy.ndarray)):
            R=len(data)
        elif isinstance(data[0],(int,float,numpy.float64,numpy.float32)):
            R=1
        else:
            error_msg(f'Unexpected data type')
            
        if R==1:
            check_type(data,f'data',list,numpy.ndarray)
            nc=int(len(data)/self.size)
            if rname is None:
                rname=0
            else:
                check_not_type(rname,'rname',list)
            if icnfg is None:
                icnfg=range(nc)
            else:
                check_type(icnfg,'icnfg',list,range)
                check_type(icnfg[0],'icnfg[:]',int,numpy.int32,numpy.int64)
                if len(icnfg)*self.size!=len(data):
                    error_msg(f'Incompatible icnfg and data, for dims={dims}')
            if numpy.size(self.mean)!=0:
                N0 = sum([self.rdata[rd].n for rd in self.rdata])
                mean0 = numpy.reshape(self.mean,(self.size,))
                mean1 = numpy.mean(numpy.reshape(data,(nc,self.size)),0)
                self.mean = (N0*mean0 + nc*mean1)/(N0+nc)
                d = nc*(mean0-mean1)/(N0+nc)
                if lat is None:
                    for rd in self.rdata:
                        for i in range(self.rdata[rd].n):
                            self.rdata[rd].delta[:,i] += d
                else:
                    for mfd in self.mfdata:
                        for i in range(self.mfdata[mfd].n):
                            self.mfdata[mfd].delta[:,i] += d
            else:
                self.mean=numpy.mean(numpy.reshape(data,(nc,self.size)),0)
                
            key=f'{ename}:{rname}'
            if lat is None:
                self.rdata[key] = rdata(mask,icnfg,data,self.mean)
            else:
                self.mfdata[key] = mfdata(mask,icnfg,lat,data,self.mean)
        else:
            for ir in range(R):
                check_type(data[ir],f'data[{ir}]',list,numpy.ndarray)
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
                check_type(rname,'rname',list)
                if len(rname)!=R:
                    error_msg('Incompatible rname and data')
            if not icnfg is None:
                check_type(icnfg,'icnfg',list)
            
            if numpy.size(self.mean)!=0:
                Py3obsErr('Only a single replica can be added to existing observables')
            if icnfg is None:
                icnfg = []
                for ir in range(len(data)):
                    nc=int(len(data[ir])/self.size)
                    icnfg.append(range(nc))
            else:
                for ir in range(len(data)):
                    if len(icnfg[ir])*self.size!=len(data[ir]):
                        error_msg(f'Incompatible icnfg[{ir}] and data[{ir}], for dims={dims}')
            for ir in range(len(data)):
                key=f'{ename}:{rname[ir]}'
                if lat is None:
                    self.rdata[key] = rdata(mask,icnfg[ir],data[ir],self.mean)
                else:
                    self.mfdata[key] = mfdata(mask,icnfg[ir],lat,data[ir],self.mean)
        self.mean = numpy.reshape(self.mean, self.dims)
        memory.add(self)
        if is_verbose('obs.create'):
            print(f'obs.create executed in {time()-t0:g} secs')

        
    def create_cd(self,cname,value,covariance):
        """
        Create observables based on covariance matrices
        
        Parameters:
           cname (str): label that uniquely identifies the data set
           value (array): central value of the observable; only 1-D arrays are supported
           covariance (array): a 2-D covariance matrix; if `covariance` is a 1-D array of
              same length as `value`, a diagonal covariance matrix is assumed.
        
        Examples:
           >>> mpi = pyobs.obs(desc='pion masses, charged and neutral')
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
        self.dims = numpy.shape(self.mean)
        if numpy.ndim(self.dims)!=1:
            error_msg(f'Unexpected value, only 1-D arrays are supported')
        self.size = numpy.prod(self.dims)
        if cov.shape!=(self.size,) and cov.shape!=(self.size,self.size):
            error_msg(f'Unexpected shape for covariance {cov.shape}')
        check_type(cname,'cname',str)
        self.cdata[cname] = cdata(numpy.eye(self.size),cov)
        memory.add(self)
        
    def add_syst_err(self,name,err):
        """
        Add a systematic error to the observable
        
        Parameters:
           name (str): label that uniquely identifies the syst. error
           err (array): array with the same dimensions of the observable
              with the systematic error
        
        Examples:
           >>> data = [0.198638, 0.403983, 1.215960, 1.607684, 0.199049, ... ]
           >>> vec = pyobs.obs(desc='vector')
           >>> vec.create('A',data,dims=(4,))
           >>> print(vec)
           0.201(13)    0.399(26)    1.199(24)    1.603(47)
           >>> vec.add_syst_err('syst.err',[0.05,0.05,0,0])
           >>> print(vec)
           0.201(52)    0.399(56)    1.199(24)    1.603(47)
           
        """
        check_type(name,'name',str)
        if name in self.cdata:
            error_msg(f'Label {name} already used')
        if numpy.shape(err)!=self.dims:
            error_msg(f'Unexpected error, dimensions do not match {self.dims}')
        cov = numpy.reshape(numpy.array(err)**2, (self.size,))
        grad = numpy.diag(1.0*(numpy.array(err)!=0.0))
        self.cdata[name] = cdata(grad,cov)
        
        
    def __del__(self):
        memory.rm(self)
        
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
        print(f'Observable with shape = {self.dims}')
        print(f' - description: {self.description}')
        print(f' - size: {memory.get(self)}')
        print(f' - mean: {self.mean}')
        
        def core(n0,n1):
            for name in self.__dict__[n0]:
                print(f' - {"Ensemble" if n0 == "edata" else "Master-field"} {name}')
                m=0
                for key in self.__dict__[n1]:
                    rn=key.split(':')
                    if rn[0]==name:
                        print(f'    - Replica {rn[1]} with mask {self.__dict__[n1][key].mask} and {"ncnfg" if n0 == "edata" else "sites"} {self.__dict__[n1][key].n}')
                        mm=self.rdata[key].ncnfg()*8.*2. if n0 == 'edata' else (self.mfdata[key].vol()+1)*8.
                        m=(mm>m)*mm + (mm<=m)*m
                print(f'         temporary additional memory required {m/1024.**2:.2g} MB')
        core('edata','rdata')
        core('mfname','mfdata')
        
        for cd in self.cdata:
            print(f' - Data {cd} with gradient shape {self.cdata[cd].grad.shape} and cov. matrix {self.cdata[cd].cov.shape}')
        print('')
    
    def save(self,name):
        """
        Save the observable to disk in '.json.gz' format. Apart from
        the compression with gunzip, the file is a plain text file generated
        with json format, for easy human readability and compatibility with
        other programming languages (json format is widely supported).
        
        Parameters:
           name (str): string with the destination (path+filename)
        
        Examples:
           >>> obsA = pyobs.obs('obsA')
           >>> obsA.create('A',data)
           >>> obsA.save('~/analysis/obsA')
           >>> obsA.save('~/analysis/obsA.json')
           >>> obsA.save('~/analysis/obsA.json.gz')
        """
        if os.path.isfile(name):
            error_msg(f'File {name} already exists')
        if (name[-6:]=='.json'):
            fname = f'{name}.gz'
        elif (name[-9:]=='.json.gz'):
            fname = name
        else:
            fname = f'{name}.json.gz'
        
        self.www[2] = datetime.datetime.now().strftime('%c')
        with gzip.open(fname, 'wt') as f:
            tofile = json.dumps(self, indent=2, default=self.__encoder__ )
            f.write( tofile )
    
    def load(self,name):
        """
        Load the observable from disk.
        
        Parameters:
           name (str): string with the source file (path+filename)
        
        Examples:
           >>> obsA = pyobs.obs('obsA')
           >>> obsA.load('~/analysis/obsA.json.gz')
        """
        if not os.path.isfile(name):
            error_msg(f'File {name} not found')
            
        tmp = json.loads(gzip.open(name, 'r').read())
        self.description = tmp['description']
        if 'www' in tmp:
            self.www = list(tmp['www'])
        else:
            self.www = ['unknown','unknown','unknown']
        self.mean = numpy.array(tmp['mean'])
        self.dims = tuple(tmp['dims'])
        self.size=numpy.prod(self.dims)
        self.edata = list(tmp['edata'])
        for key in tmp['rdata']:
            if (type(tmp['rdata'][key]['idx']) is str):
                regex=re.compile('[(,)]')
                h = regex.split(tmp['rdata'][key]['idx'])
                if h[0]!='range':
                    error_msg('Unexpected idx')
                self.rdata[key] = rdata(tmp['rdata'][key]['mask'],range(int(h[1]),int(h[2]),int(h[3])))
            else:
                self.rdata[key] = rdata(tmp['rdata'][key]['mask'],tmp['rdata'][key]['idx'])
            self.rdata[key].delta = numpy.array(tmp['rdata'][key]['delta'])
            self.rdata[key].delta2 = numpy.array(tmp['rdata'][key]['delta2'], dtype=numpy.float32)
        self.mfname = list(tmp['mfname'])
        for key in tmp['mfdata']:
            if (type(tmp['mfdata'][key]['idx']) is str):
                regex=re.compile('[(,)]')
                h = regex.split(tmp['mfdata'][key]['idx'])
                if h[0]!='range':
                    error_msg('Unexpected idx')
                self.mfdata[key] = mfdata(tmp['mfdata'][key]['mask'],
                                         range(int(h[1]),int(h[2]),int(h[3])),tmp['mfdata'][key]['lat'])
            else:
                self.mfdata[key] = mfdata(tmp['mfdata'][key]['mask'],
                                         tmp['mfdata'][key]['idx'],tmp['mfdata'][key]['lat'])
            self.mfdata[key].delta = numpy.array(tmp['mfdata'][key]['delta'])
            self.mfdata[key].delta2 = numpy.array(tmp['mfdata'][key]['delta2'], dtype=numpy.float32)
        for key in tmp['cdata']:
            self.cdata[key] = cdata(tmp['cdata'][key]['mask'],tmp['cdata'][key]['cov'])
        memory.add(self)
        
    def __encoder__(self,obj):
        if isinstance(obj,numpy.integer):
            return int(obj)
        elif isinstance(obj,numpy.ndarray):
            return obj.tolist() #json.dumps(obj.tolist())
        elif isinstance(obj,range):
            return f'range({obj.start},{obj.stop},{obj.step})'
        return obj.__dict__
    
    def __str__(self):
        [v,e] = self.error()
        D=len(self.dims)
        if D==1:
            out = '\t'.join([valerr(v[i],e[i]) for i in range(self.dims[0])])
            out += '\n'
        elif D==2:
            out= ''
            for i in range(self.dims[0]):
                out += '\t'.join([valerr(v[i,j],e[i,j]) for j in range(self.dims[1])])
                out += '\n'
        return out
    
    def __repr__(self):
        return self.__str__()
    
    def __getitem__(self,args):
        if isinstance(args,(int,numpy.int32,numpy.int64,slice,numpy.ndarray)):
            args=[args]
        na=len(args)
        if na!=len(self.dims):
            error_msg('Unexpected argument')
        new_size=1
        for i in range(na):
            if isinstance(args[i],(slice,numpy.ndarray)):
                new_size *= numpy.size(numpy.arange(self.dims[i])[args[i]])
        grad=numpy.zeros((new_size,self.size))
        hess=numpy.zeros((new_size,self.size,self.size))
        idx = numpy.reshape(range(self.size),self.dims)[tuple(args)]
        a=0
        for b in idx.flatten():
            grad[a,b]=1.0
            a+=1
        return derobs([self],self.mean[tuple(args)],[grad])
    
    def __addsub__(self,y,sign):
        g0=numpy.eye(self.size)
        if isinstance(y,obs):
            g1=sign*numpy.eye(y.size)
            return derobs([self,y],self.mean+sign*y.mean,[g0,g1])
        else:
            return derobs([self],self.mean+sign*y,[g0])
    
    def __add__(self,y):
        return self.__addsub__(y,+1)
    
    def __sub__(self,y):
        return self.__addsub__(y,-1)
    
    def __neg__(self):
        return derobs([self],-self.mean,[-numpy.eye(self.size)])
    
    def __mul__(self,y):
        if isinstance(y,obs):
            g0=unary_grad(self.mean,lambda x:x*y.mean)
            g1=unary_grad(y.mean,lambda x:self.mean*x)
            return derobs([self,y],self.mean*y.mean,[g0,g1])
        else:
            g0=unary_grad(self.mean,lambda x:x*y)
            return derobs([self],self.mean*y,[g0])
    
    def __matmul__(self,y):
        if isinstance(y,obs):
            g0=unary_grad(self.mean,lambda x: x @ y.mean)
            g1=unary_grad(y.mean,lambda x:self.mean @ x)
            return derobs([self,y],self.mean @ y.mean,[g0,g1])
        else:
            g0=unary_grad(self.mean,lambda x: x @ y)
            return derobs([self],self.mean @ y,[g0])

    def reciprocal(self):
        new_mean = numpy.reciprocal(self.mean)
        g0=unary_grad(self.mean, lambda x:-x*(new_mean**2))
        return derobs([self], new_mean,[g0])
    
    def __truediv__(self,y):
        if isinstance(y,obs):
            return self * y.reciprocal()
        else:
            return self * numpy.reciprocal(y)
   
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
        g0=unary_grad(self.mean, lambda x: a * x*self.mean**(a-1))
        return derobs([self], new_mean, [g0])
    
    def error_core(self,errinfo,plot,pfile):
        sigma_tot = numpy.zeros(self.dims)
        dsigma_tot = numpy.zeros(self.dims)
        sigma = {}
        for ed in self.edata:
            if ed in errinfo:
                if errinfo[ed].bsize is None:
                    res = uwerr(self,ed,plot,pfile,errinfo[ed].Stau,errinfo[ed].W)
            else:
                res = uwerr(self,ed,plot,pfile)
            sigma[ed] = numpy.reshape(res[0],self.dims)
            sigma_tot += sigma[ed]
            dsigma_tot += numpy.reshape(res[1],self.dims)
            
        for mf in self.mfname:
            if mf in errinfo:
                res = mferr(self,mf,plot,pfile,errinfo[mf].k,errinfo[mf].Stau,errinfo[mf].R)
            else:
                res = mferr(self,mf,plot,pfile)
            sigma[mf] = numpy.reshape(res[0],self.dims)
            sigma_tot += sigma[mf]
            dsigma_tot += numpy.reshape(res[1],self.dims)
        
        for cd in self.cdata:
            sigma[cd] = numpy.reshape(self.cdata[cd].sigmasq(),self.dims)
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
           >>> obsA = pyobs.obs('obsA')
           >>> obsA.create('A',dataA) # create the observable A from ensemble A
           >>> [v,e] = obsA.error() # collect central value and error in v,e
           >>> einfo = {'A': errinfo(Stau=3.0)} # specify non-standard Stau for ensemble A
           >>> [_,e1] = obsA.error(errinfo=einfo)
           >>> print(e,e1) # check difference in error estimation
        
           >>> obsB = pyobs.obs('obsB')
           >>> obsB.create('B',dataB) # create the observable B from ensemble B
           >>> obsC = obsA * obsB # derived observable with fluctuations from ensembles A,B
           >>> einfo = {'A': errinfo(Stau=3.0), 'B': errinfo(W=30)}
           >>> [v,e] = obsC.error(errinfo=einfo,plot=True)
        """
        t0=time()
        [sigma, sigma_tot, _] = self.error_core(errinfo,plot,pfile)
        
        if plot:
            h=[len(self.edata),len(self.mfname),len(self.cdata)]
            if sum(h)>1:
                plot_piechart(self.description, sigma, sigma_tot)
            
        if is_verbose('obs.error'):
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
        """
        # to be improved - add errinfo
        tau = {}
        for ed in self.edata:
            res = uwerr(self,ed,False,None)
            tau[ed] = [numpy.reshape(res[2][:,0],self.dims), numpy.reshape(res[2][:,1],self.dims)]

        for mf in self.mfname:
            res = mferr(self,mf,False,None)
            tau[mf] = [numpy.reshape(res[2][:,0],self.dims), numpy.reshape(res[2][:,1],self.dims)]
        
        return tau
        

