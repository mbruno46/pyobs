import numpy
import json
import gzip

from os.path import isfile

from .ensdata import edata, rdata, cdata

from .mathfcts import *

from .core.utils import valerr, union, double_union, piechart
from .core.utils import pad_with_zeros
from .core.libxml import read_xml

__all__ = ['observa', 'derobs',
        'log', 'exp', 'sin', 'cos', 'arcsin', 'arccos',
        'sinh', 'cosh', 'arcsinh', 'arccosh', 
        'trace', 'det', 'inv']

class InputError(Exception):
    pass

class observa:
    """ Class defining an observable

    It contains the fluctuations of all ensembles on which
    the observable is defined. It allows the computation of the
    error using the `Gamma-method` and supports most operations
    required for an analysis, including other miscellanea functions.

    """

    def __init__(self):
        self.mean = numpy.array(0.0)
        self.dims = ()
        self.eid = []
        self.edata = []
        self.cid = []
        self.cdata = []

    def __str__(self):
        [v, e] = self.vwerr()
        # prints automatically mean value and error in nice format
        out = ''
        if (self.dims==(1,1)):
            out = out + '\t' + valerr(v,e)
        elif (self.dims==(1,self.dims[1])):
            for j in range(self.dims[1]):
                out = out + '\t' + valerr(v[j],e[j])
        else:
            for i in range(self.dims[0]):
                for j in range(self.dims[1]):
                    out = out + '\t' + valerr(v[i,j],e[i,j])
                out = out + '\n'
        out = out + '\n'
        return out

    def clone(self,full=True):
        """ Clone the observable 
        
        Parameters
        ----------
        full : bool, optional
               flag to decide whether mean and fluctuations are copied. 
        
        Returns
        -------
        observa
            a copy of the original observa 

        Examples
        --------
        >>> aclone = a.clone(True)
        """

        res = observa()
        res.mean = numpy.array(self.mean)
        res.dims = self.dims
        res.eid = list(self.eid)
        res.edata = [ ed.clone(full) for ed in self.edata ]
        res.cid = list(self.cid)
        res.cdata = [ cd.clone(full) for cd in self.cdata ]
        return res

    def find_edata(self, ed0):
        for ed in self.edata:
            if (ed.id==ed0.id):
                if (ed.name==ed0.name):
                    return ed
                else:
                    raise 
        return None
    
    def find_cdata(self, cd0):
        for cd in self.cdata:
            if (cd.id==cd0.id):
                if (cd.name==cd0.name):
                    return cd
                else:
                    raise 
        return None

    def primary_observable(self,eid,ename,rid,rname,idx,data,dims=(1,1)):
        """ Create a primary observable from raw measurements

        Parameters
        ----------
        eid : int
            index that uniquely identifies the ensemble
        ename : str
            label of the ensemble
        rid : list
            indices that uniquely indetify the replicas
        rname : list
            labels of the replicas
        idx : list
            configuration indices corresponding to the measurements 
            on a set of replicas
        data : list
            measurements per configuration performed on a set of replicas. 
            For vectors and matrices the data is interpreted as follows
            data[ir][k + dims[1]*(j + dims[0]*i)] with i config. id and
            (j,k) row and column indices and ir the replica id.
        dims : 2D tuple, optional
            size of observable

        """

        if not isinstance(dims,tuple):
            raise InputError("dims must be a tuple")
        if (len(dims)!=2):
            raise InputError("dims must be a 2D tuple")
        size=numpy.prod(dims)
        self.dims = dims
        
        if not isinstance(rid,list):
            raise InputError("rid must be a list")
        if not isinstance(rname,list):
            raise InputError("rname must be a list")
        if not isinstance(idx,list):
            raise InputError("idx must be a list")
        if not isinstance(data,list):
            raise InputError("data must be a list")
        
        if (len(rid)!=len(numpy.unique(rid))):
            raise InputError("Unexpected rid: repetitions found")
        R = len(rid)
        if (rname!=[]):
            if (len(rname)!=R):
                raise InputError("rname incompatible with rid")
        if (len(idx)!=R):
            raise InputError("idx incompatible with rid")
        if (len(data)!=R):
            raise InputError("data incompatible with rid")
        for ir in range(R):
            if (len(data[ir]) % size)!=0:
                raise InputError("data and dims are not compatible")
            if (len(data[ir])/size)!=len(idx[ir]):
                raise InputError("data, dims and idx are not compatible")
        if not isinstance(eid,int):
            raise InputError("eid must be an integer")
        self.eid.append( eid )
        if not isinstance(ename,str):
            raise InputError("ename must be a string")
        self.edata.append( edata() )
        self.mean = self.edata[0].create(eid,ename,rid,rname,idx,data,dims)

    def c_observable(self,cid,cname,data,cov):
        if not isinstance(data,list):
            raise ValueError('data must be a list')
        if not isinstance(cid,int):
            raise ValueError('cid must be an int')
        if not isinstance(cov,(list,numpy.ndarray)):
            raise ValueError('unexpeceted cov')
        
        if (numpy.ndim(data)==1):
            self.mean = numpy.array([data])
            if (numpy.shape(cov)==(len(data),len(data))):
                _cov = numpy.reshape(cov,(1,len(data),1,len(data)))
            elif (numpy.shape(cov)==(len(data),)):
                _cov = numpy.zeros((1,len(data),1,len(data)))
                for i in range(len(data)):
                    _cov[0,i,0,i] = cov[i]
            elif (numpy.shape(cov)==(1,len(data),1,len(data))):
                _cov = numpy.array(cov)
            else:
                raise ValueError('unexpeceted cov')
        elif (numpy.ndim(data)==2):
            self.mean = numpy.array(data)
            (d0,d1) = self.mean.shape
            if (numpy.shape(cov)!=(d0,d1,d0,d1)):
                raise ValueError('unexpeceted cov')
            else:
                _cov = numpy.array(cov)
        self.dims = self.mean.shape
        
        self.cid.append( cid )
        _grad = numpy.zeros(self.dims+self.dims)
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                _grad[i,j,i,j] = 1.0
        cd = cdata(cid, cname, self.dims)
        cd.create(_cov,_grad)
        self.cdata.append(cd)

    def vwerr(self,plot=False,pfile=None,errinfo=None,simplify=True):
        """ Compute the error of each element

        Parameters
        ----------
        plot : bool, optional
            tells the program to plot the autocorrelation
            function with the automatic summation window. One plot per element
            and ensemble is displayed.
        pfile : str, optional
            provides a filename to save the plots with the autocorrelation function.
        errinfo : errinfo, optional
            class to customize the automatic procedure to define the summation 
            window. It contains information for multiple ensembles. If one ensemble is
            present in the observable and not in errinfo the default uwerr routine 
            is used. If Texp is non-zero then the tails are attached automatically.
        simplify: bool, optional
            for a scalar observable returns floats, for a vector returns a 1-D array
            and for a matrix returns the full 2D array. If set to False it always
            returns objects in 2D

        Returns
        -------
        value : float or array
        error : float or array
        tau : float or array
        dtau : float or array

        Examples
        --------
        >>> [val, err] = a.vwerr()
        >>> [val, err] = a.vwerr(True) # to plot
        >>> a.vwerr(True,'/path/to/file') # to save the plot
        >>> einfo = errinfo()
        >>> einfo.addEnsemble(0,Stau=2.4)
        >>> [val, err] = a.vwerr(errinfo=einfo)
        """

        if (plot==False and pfile!=None):
            print 'Warning: plots ' + pfile + ' are not saved. Use plot=True'

        sigma = []
        sigma_tot = numpy.zeros(self.dims)
        for ed in self.edata:
            if (errinfo==None):
                res = ed.uwerr(plot, pfile)
            else:
                if (errinfo.Tail(ed.id)==True):
                    res = ed.uwerr_texp(plot, pfile, errinfo.getUWerrTexp(ed.id))
                else:
                    res = ed.uwerr(plot, pfile, errinfo.getUWerr(ed.id))
            sigma.append(res[0])
            sigma_tot += sigma[-1]
        
        # add cdata
        for cd in self.cdata:
            sigma.append( cd.sigma() )
            sigma_tot += sigma[-1]

        if (plot==True):
            if (len(self.eid)>1 or len(self.cid)>1):
                nn = [ed.name for ed in self.edata]+[cd.name for cd in self.cdata]
                for i in range(self.dims[0]):
                    for j in range(self.dims[1]):
                        piechart([sigma[k][i,j] for k in range(len(sigma))],nn,(i,j))

        error = numpy.sqrt(sigma_tot)
        if (simplify==False):
            return [self.mean, error]
        else:
            if (self.dims[0]==1):
                if (self.dims[1]==1):
                    return [self.mean[0,0], error[0,0]]
                else:
                    return [self.mean[0,:], error[0,:]]
            else:
                return [self.mean, error]


    def vwcov(self,plot=False,errinfo=None,simplify=True):
        covmat = numpy.zeros(self.dims+self.dims)

        for ed in self.edata:
            if (errinfo==None):
                res = ed.uwcov(plot)
            else:
                if (errinfo.Tail(ed.id)==True):
                    raise ValueError('texp not yet supported for vwcov');
                    #res = ed.uwerr_texp(plot, pfile, errinfo.getUWerrTexp(ed.id))
                else:
                    res = ed.uwcov(plot, errinfo.getUWerr(ed.id))
            covmat += res[0]

        if (simplify==False):
            return covmat
        else:
            if (self.dims[0]==1):
                return numpy.array(covmat[0,:,0,:])
            else:
                if (self.dims[1]==1):
                    return numpy.array(covmat[:,0,:,0])


    def tauint(self,eid=None,errinfo=None):
        """ Computes the autocorrelation time and its error 

        Parameters
        ----------
        eid: int, optional
            ensemble id; if not given a list of the tauints of all ensembles
            is returned
        errinfo: einfo, optional
            einfo class with additional information on the error calculation

        Returns
        -------
        tau: list of arrays with dimension equal to the observable
        dtau: list of arrays with dimension equal to the observable
        """

        tau = []
        dtau = []
        if isinstance(eid,int):
            try:
                ie = self.eid.index(eid)
            except:
                raise TypeError("ensemble id not found")

            ed = self.edata[eid]
            if (errinfo==None):
                res = ed.uwerr(False, None)
            else:
                if (errinfo.Tail(ed.id)==True):
                    res = ed.uwerr_texp(False, None, errinfo.getUWerrTexp(ed.id))
                else:
                    res = ed.uwerr(False, None, errinfo.getUWerr(ed.id))
            tau.append(res[1])
            dtau.append(res[2])
        else:
            for ed in self.edata:
                if (errinfo==None):
                    res = ed.uwerr(False, None)
                else:
                    if (errinfo.Tail(ed.id)==True):
                        res = ed.uwerr_texp(False, None, errinfo.getUWerrTexp(ed.id))
                    else:
                        res = ed.uwerr(False, None, errinfo.getUWerr(ed.id))
                tau.append(res[1])
                dtau.append(res[2])
        return [tau,dtau]

    def naive_err(self):
        """ Compute the standard deviation assuming zero autocorrelations

        Returns
        -------
        sigma : array
            The standard deviation as an array with the same dimensions of the observable
        """ 

        sigma_tot = numpy.zeros(self.dims)
        for ed in self.edata:
            sigma_tot = sigma_tot + ed.naive_err()
        for cd in self.cdata:
            sigma_tot += cd.sigma()

        return numpy.sqrt(sigma_tot)

    def jkerr(self,plot=False,jkinfo=None,simplify=True):
        """ Compute the error of each element

        Parameters
        ----------
        plot : bool, optional
            tells the program to plot a pie-chart with the contributions
            to the error from the various ensembles
        jkinfo : dictionary, optional
            dictionary with the bin size for each ensemble.
            It contains information for multiple ensembles. If one ensemble is
            present in the observable and not in jkinfo the default jkerr routine 
            assigns bin size of 1. 
        simplify: bool, optional
            for a scalar observable returns floats, for a vector returns a 1-D array
            and for a matrix returns the full 2D array. If set to False it always
            returns objects in 2D

        Returns
        -------
        value : float or array
        error : float or array

        Examples
        --------
        >>> [val, err] = a.jkerr()
        >>> [val, err] = a.jkerr(True) # to plot
        >>> [val, err] = a.jkerr(jkinfo={0: 2})
        """

        sigma = []
        sigma_tot = numpy.zeros(self.dims)
        for ed in self.edata:
            if (jkinfo==None):
                res = ed.jkerr(1,self.mean)
            else:
                if ed.id in jkinfo:
                    res = ed.jkerr(jkinfo[ed.id],self.mean)
                else:
                    res = ed.jkerr(1,self.mean)
            sigma.append(res[0])
            sigma_tot += sigma[-1]
        
        # add cdata
        for cd in self.cdata:
            sigma.append( cd.sigma() )
            sigma_tot += sigma[-1]

        if (plot==True):
            if (len(self.eid)>1 or len(self.cid)>1):
                nn = [ed.name for ed in self.edata]+[cd.name for cd in self.cdata]
                for i in range(self.dims[0]):
                    for j in range(self.dims[1]):
                        piechart([sigma[k][i,j] for k in range(len(sigma))],nn,(i,j))

        error = numpy.sqrt(sigma_tot)
        if (simplify==False):
            return [self.mean, error]
        else:
            if (self.dims[0]==1):
                if (self.dims[1]==1):
                    return [self.mean[0,0], error[0,0]]
                else:
                    return [self.mean[0,:], error[0,:]]
            else:
                return [self.mean, error]

    #########################################################
    
    def __getitem__(self,args):
        if isinstance(args[0],slice):
            new_dims = (len(range(self.dims[0])[args[0]]),)
        elif isinstance(args[0], int):
            new_dims = (1,)
        if isinstance(args[1],slice):
            new_dims = new_dims + (len(range(self.dims[1])[args[1]]),)
        elif isinstance(args[1],int):
            new_dims = new_dims + (1,)
        new_mean = numpy.reshape( self.mean[args], new_dims)
        res = self.clone(False)
        res.mean = new_mean
        res.dims = new_dims
        for ed in res.edata:
            ed.dims = new_dims
            ed1 = self.find_edata(ed)
            for rd in ed.rdata:
                rd1 = ed1.find_rdata(rd)
                rd.dims = new_dims
                rd.data = numpy.reshape( rd1.data[args[0],args[1],:], new_dims+(rd1.ncnfg,) )
        for cd in res.cdata:
            cd.dims = res.dims
            ic = res.cdata.index(cd)
            cd.cov_grad = numpy.reshape(self.cdata[ic].cov_grad[args[0],args[1]], cd.dims+cd.cov_dims[0:2])
        return res


    def addrow(self,y):
        if (self.dims==()):
            self.mean = numpy.array(y.mean)
            self.dims = y.dims
            self.eid = list(y.eid)
            self.edata = [ ed.clone(True) for ed in y.edata ]
            self.cid = list(y.cid)
            self.cdata = [ cd.clone(True) for cd in y.cdata ]
        else:
            # check same ncols
            if (self.dims[1]!=y.dims[1]):
                raise ValueError('number of columns do not match')
            if (self.eid!=y.eid):
                raise ValueError('unexpected ensemble list')
            self._concatenate(y,0)
    
    def addcol(self,y):
        if (self.dims==()):
            self.mean = numpy.array(y.mean)
            self.dims = y.dims
            self.eid = list(y.eid)
            self.edata = [ ed.clone(True) for ed in y.edata ]
            self.cid = list(y.cid)
            self.cdata = [ cd.clone(True) for cd in y.cdata ]
        else:
            # check same ncols
            if (self.dims[0]!=y.dims[0]):
                raise ValueError('number of columns do not match')
            if (self.eid!=y.eid):
                raise ValueError('unexpected ensemble list')
            self._concatenate(y,1)

    def _concatenate(self,y,axis):
        self.mean = numpy.concatenate((self.mean, y.mean), axis)
        self.dims = self.mean.shape
        for ed in self.edata:
            ed.dims = self.dims
            ed1 = y.find_edata(ed)
            if (ed1!=None):
                for rd in ed.rdata:
                    rd.dims = self.dims
                    rd1 = ed1.find_rdata(rd)
                    if (rd1!=None):
                        rd.data = numpy.concatenate((rd.data, rd1.data),axis)
        for cd in self.cdata:
            cd.dims = self.dims
            cd1 = y.find_cdata(cd)
            if (cd1!=None):
                cd.cov_grad = numpy.concatenate((cd.cov_grad, cd1.cov_grad),axis)
        return self

    #########################################################

    def peek(self):
        """ Inspects the observable

        mean value and dimensions are printed out, together with
        a list of the ensembles and replica containing fluctuations 
        of the current observable
        """

        print self.mean
        print 'dimensions ', self.dims
        for ed in self.edata:
            print '--- ensemble ', ed.id, ' ', ed.name
            for rd in ed.rdata:
                print '    --- replica ', rd.id, ' ', rd.name
                print '            --- ncnfg ', rd.ncnfg
        for cd in self.cdata:
            print '--- cdata ', cd.id, ' ', cd.name
        print ' '

    def __encoder(self,obj):
        if isinstance(obj,numpy.ndarray):
            return obj.tolist() #json.dumps(obj.tolist())
        return obj.__dict__

    def save(self,name):
        """ Saves the observable to disk

        Parameters
        ----------
        name : str
            name of the output file, including the path. The extension is optional
            and if it is not passed the observables are always 
            saved in the (compressed) json format with the `.pyobs.gz` extension.
            Alternatively it is possible to save the observable in binary format
            by specifying the extensions `.pyobs.dat`

        Examples
        --------
        >>> a.save("./test") # creates a file test.pyobs.gz with json format
        >>> a.save("./test.pyobs.dat") # creates a file test.pyobs.dat with binary format TO FIX
        """

        if (name[-6:]=='.pyobs'):
            fname = name + '.gz'
        elif (name[-9:]=='.pyobs.gz'):
            fname = name
        else:
            fname = name + '.pyobs.gz'
    
        if (fname[-9:]=='.pyobs.gz'):
            self._save_json(fname)
        elif (fname[-10:]=='.pyobs.dat'):
            self._save_dat(fname)
        else:
            raise ValueError

    def _save_json(self,fname):
        f = gzip.open(fname, 'wb')
        tofile = json.dumps(self, indent=2, default=self.__encoder )
        f.write( tofile )
        f.close()

    def _save_dat(self,fname):
        f = open(fname, 'wb')
        nd=numpy.prod(self.dims)
        f.write(struct.pack('ii',*self.dims))
        dat=struct.pack('f'*nd,*self.mean.flatten())
        f.write(dat)
        f.write(struct.pack('i'*(1+len(self.eid)),len(self.eid),*self.eid))
        for ed in self.edata:
            f.write(struct.pack('i'+'c'*len(ed.name),len(ed.name),*ed.name))
            f.write('i'*(ed.R+1),ed.R,*ed.rid)
            for rd in ed.rdata:
                f.write(struct.pack('i'+'c'*len(rd.name),len(rd.name),*rd.name))
                f.write(struct.pack('i'*(ncnfg+1),ncnfg,*rd.idx))
                f.write(struct.pack('f'*ncnfg*nd, *rd.data.flatten()))
        f.write(struct.pack('i'*(1+len(self.cid)),len(self.cid),*self.cid))
        for cd in self.cdata:
            f.write(struct.pack('i'+'c'*len(cd.name),len(cd.name),*cd.name))
        f.close()
    
    def load(self,name):
        """ Reads the observable from disk

        Parameters
        ----------
        name : str
               full path of the file to be loaded. Supported formats are
               the compressed json format `.pyobs.gz`, the binary format
               `.pyobs.dat` and the MATLAB xml format `.xml.gz`.
               The extension must be always specified

        Examples
        --------
        >>> a = observa()
        >>> a.load('./test.pyobs.gz')
        """

        if not isfile(name):
            raise ValueError('file %s not found!' % name)

        if (name[-7:]=='.xml.gz'):
            tmp = gzip.open(name, 'r').read()
            self._load_xml(tmp)
        elif (name[-4:]=='.xml'):
            self._load_xml(name)
        elif (name[-10:]=='.pyobs.dat'):
            self._load_dat(name)
        elif (name[-9:]=='.pyobs.gz'):
            self._load_json(name)
        
    def _load_json(self, name):
        tmp = json.loads(gzip.open(name, 'r').read())

        self.mean = numpy.array(tmp['mean'])
        self.dims = tuple(tmp['dims'])
        self.eid = list(tmp['eid'])
        for e in self.eid:
            ie = self.eid.index(e)
            ed = edata()
            ed.id = e
            ed.dims = self.dims
            ed.name = tmp['edata'][ie]['name']
            ed.rid = list(tmp['edata'][ie]['rid'])
            ed.R = tmp['edata'][ie]['R']
            for r in ed.rid:
                ir = ed.rid.index(r)
                rd = rdata(r, tmp['edata'][ie]['rdata'][ir]['name'], self.dims)
                rd.idx = list(tmp['edata'][ie]['rdata'][ir]['idx'])
                rd.data = numpy.array( tmp['edata'][ie]['rdata'][ir]['data'] )
                rd.ncnfg = len(rd.idx)
                ed.rdata.append( rd )
            self.edata.append( ed )
        self.cid = list(tmp['cid'])
        for c in self.cid:
            ic = self.cid.index(c)
            cd = cdata(c, tmp['cdata'][ic]['name'], self.dims)
            cd.create(tmp['cdata'][ic]['cov'], tmp['cdata'][ic]['cov_grad']) 
            self.cdata.append(cd)

    def _load_dat(self,fname):
        f = open(fname, 'rb')
        nd=numpy.prod(self.dims)
        self.dims = struct.unpack('ii',f.read(8))
        nd=numpy.prod(self.dims)
        tmp = struct.unpack('f'*nd,f.read(8*nd))
        self.mean = numpy.reshape(tmp, self.dims)
        n = struct.unpack('i',f.read(4))
        self.eid = list( struct.unpack('i'*n, f.read(4*n)) )
        for e in self.eid:
            ed = edata()
            ed.id = e
            ed.dims = self.dims
            n = struct.unpack('i',f.read(4))
            ed.name = struct.unpack('c'*n, f.read(n))
            ed.R = struct.unpack('i',f.read(4))
            ed.rid = list( struct.unpack('i'*ed.R, f.read(4*ed.R)) )
            for r in ed.rid:
                n = struct.unpack('i',f.read(4))

                rd = rdata(r, struct.unpack('c'*n,f.read(n)), self.dims)
                rd.ncnfg = struct.unpack('i',f.read(4))
                rd.idx = list( struct.unpack('i'*rd.ncnfg, f.read(4*rd.ncnfg)) )
                dat = struct.unpack('f'*nd*rd.ncnfg, f.read(8*ncnfg*nd))
                rd.data = numpy.reshape(dat, self.dims+(rd.ncnfg,))
                ed.rdata.append( rd )
            self.edata.append( ed )
        f.close()

    def _load_xml(self, name, edict=None):
        xml = read_xml(name)
        
        tmp = numpy.fromstring(xml['mean'],dtype=float, sep=' ')
        self.dims = (1,len(tmp))
        self.mean = numpy.reshape(tmp, self.dims)
        ne = int(xml['ne'])
        if (edict==None):
            self.eid = range(ne)
        else:
            self.eid = []
            for ie in range(ne):
                self.eid.append(edict[xml['edata'][ie]['enstag'].strip()])
        for e in self.eid:
            ie = self.eid.index(e)
            ed = edata()
            ed.id = e
            ed.dims = self.dims
            ed.name = xml['edata'][ie]['enstag'].strip()
            nr = int(xml['edata'][ie]['nr'])
            if (edict==None):
                ed.rid = range(nr)
            else:
                ed.rid = []
                for ir in range(nr):
                    ed.rid.append(edict[xml['edata'][ie]['rdata'][ir]['id'].strip()])
            ed.R = nr
            for r in ed.rid:
                ir = ed.rid.index(r)
                rd = rdata(r, xml['edata'][ie]['rdata'][ir]['id'].strip(), self.dims)
                dat = numpy.fromstring(xml['edata'][ie]['rdata'][ir]['data'],dtype=float,sep=' ')
                ncnfg = len(dat)/(1+numpy.prod(self.dims))
                dat2 = dat.reshape((ncnfg,1+numpy.prod(self.dims)))
                rd.idx = list(dat2[:,0])
                rd.data = numpy.reshape(dat2[:,1:].T,self.dims+(ncnfg,))
                rd.ncnfg = len(rd.idx)
                ed.rdata.append( rd )
            self.edata.append( ed )


    def plotter(self):
        """ Generates plots with the fluctuations of the observable on each ensemble
        """
        for ed in self.edata:
            ed.plotter()

    #########################################################
  
    # when numpy finds the following element defined in a class
    # it automally lets the class handle operations such as rmul or radd
    # note that using __array_ufunc__ = None would not solve the problem
    # because numpy would still try to perform the rmul or radd operation
    # element-by-element
    __array_priority__ = 1000

    def check_dims(self,y):
        if isinstance(y,observa):
            if (self.dims!=y.dims):
                raise InputError('incompatible dimensions between operands '+str(self.dims)+' , '+str(y.dims))
        elif isinstance(y,numpy.ndarray):
            if (self.dims!=numpy.shape(y)):
                raise InputError('incompatible dimensions between operands '+str(self.dims)+' , '+str(numpy.shape(y)))
        elif isinstance(y,(int,float)):
            if (self.dims!=(1,1)):
                raise InputError('incompatible dimensions between operands '+str(self.dims)+' , int/float')

    ### element wise operations

    def __add__(self,y):
        self.check_dims(y)
        if isinstance(y,observa):
            return fast_math_binary(self,y,1)
        else:
            return fast_math_scalar(self,31,y)

    def __radd__(self,y):
        return (self+y)

    def __mul__(self,y):
        if isinstance(y,observa):
            self.check_dims(y)
            return fast_math_binary(self,y,4)
        else:
            return fast_math_scalar(self,34,y)

    def __rmul__(self,y):
        return (self*y)

    def __sub__(self,y):
        self.check_dims(y)
        if isinstance(y,observa):
            return fast_math_binary(self,y,2)
        else:
            return fast_math_scalar(self,32,y)

    def __rsub__(self,y):
        self.check_dims(y)
        if isinstance(y,observa):
            return fast_math_binary(self,y,3)
        else:
            return fast_math_scalar(self,33,y)

    def __neg__(self):
        return fast_math_scalar(self,35) 

    def reciprocal(self):
        return fast_math_scalar(self,30)

    def __div__(self,y):
        self.check_dims(y)
        if isinstance(y,observa):
            return self* (y.reciprocal())
        else:
            return self* numpy.reciprocal(y)

    def __pow__(self,y):
        if isinstance(y,observa):
            raise
        else:
            return fast_math_scalar(self,2,y)

    # matrix operations
    def transpose(self):
        res = self.clone(True)
        res.mean = self.mean.transpose()
        res.dims = res.mean.shape
        for ed in res.edata:
            ed.dims = res.dims
            for rd in ed.rdata:
                rd.dims = res.dims
                rd.data = numpy.einsum('ijk->jik',rd.data)
        for cd in res.cdata:
            cd.dims = res.dims
            cd.cov_grad = numpy.einsum('ijkl->jikl',cd.cov_grad)
        return res

    def dot(self,y):
        if isinstance(y,observa):
            if (self.dims[1]!=y.dims[0]):
                raise InputError('incompatible operands for dot product '+str(self.dims)+' , '+str(y.dims))
            else:
                [f, df] = math_dot(self.mean, y.mean)
                return derobs([self,y], f, df)
        else:
            [f, df] = math_dot(self.mean, numpy.array(y))
            return derobs([self], f, [df[0]])

    def sum(self, axis=0):
        [f,df] = math_sum(self.mean, axis)
        return derobs([self],f,df)

    def cumsum(self, axis=0):
        if (axis==0):
            res = self[0,:]
            for i in range(1,self.dims[0]):
                res.addrow( self[0:i+1,:].sum(0) )
        elif (axis==1):
            res = self[:,0]
            for i in range(1,self.dims[1]):
                res.addcol( self[:,0:i+1].sum(1) )
        return res

def fast_math_scalar(inp,ifunc,a=None):
    res = inp.clone(True)
    if a is not None:
        if isinstance(a,(int,float)):
            aa = numpy.reshape(a, (1,1))
        elif isinstance(a,numpy.ndarray):
            if (a.ndim==1):
                aa = numpy.reshape(a, (1,)+a.shape)
            elif (a.ndim==2):
                aa = numpy.array(a)
        else:
            raise ValueError('Unexpected input type')
    else:
        aa = None
    [res.mean, grad] = math_scalar(inp.mean, ifunc, aa)
    for ed in res.edata:
        for rd in ed.rdata:
            df = numpy.array([ grad.T for _ in range(rd.ncnfg)]).T
            rd.data = numpy.multiply(df, rd.data)
    for cd in res.cdata:
        for a in range(cd.cov_dims[0]):
            for b in range(cd.cov_dims[1]):
                cd.cov_grad[:,:,a,b] = numpy.multiply(grad, cd.cov_grad[:,:,a,b])
    return res

def fast_math_binary(inp1,inp2,ifunc):
    inps = [inp1,inp2]
    res = merge_observa(inp1.dims, inps)
    [res.mean, grad] = math_binary(inp1.mean,inp2.mean,ifunc)
    
    for ed in res.edata:
        for i in range(2):
            edi = inps[i].find_edata(ed)
            if (edi!=None):
                for rd in ed.rdata:
                    rdi = edi.find_rdata(rd)
                    if (rdi!=None):
                        df = numpy.array([grad[i].T for _ in range(rd.ncnfg)]).T

                        d0 = rdi.idx[0]-rd.idx[0]
                        d1 = rd.idx[-1]-rdi.idx[-1]

                        if (d0==0 and d1==0):
                            tmp_data = rdi.data
                        else:
                            tmp_data = pad_with_zeros(rdi.data, d0, d1)
                        rd.data = rd.data + numpy.multiply(df, tmp_data)
    for cd in res.cdata:
        for i in range(2):
            cdi = inps[i].find_cdata(cd)
            if (cdi!=None):
                for a in range(cd.dims[0]):
                    for b in range(cd.dims[1]):
                        cd.cov_grad[a,b] += grad[i][a,b] * cdi.cov_grad[a,b]
    return res



def log(x):
    """
    Compute the logarithm of an observable x element-wise
    """
    return fast_math_scalar(x,0)

def exp(x):
    """
    Compute the exponential of an observable x element-wise
    """
    return fast_math_scalar(x,1)

def sin(x):
    """
    Compute the sine of an observable x element-wise
    """
    return fast_math_scalar(x,10)

def cos(x):
    """
    Compute the cosine of an observable x element-wise
    """
    return fast_math_scalar(x,11)

def tan(x):
    """
    """
    return fast_math_scalar(x,14)

def arctan(x):
    """
    """
    return fast_math_scalar(x,15)

def arcsin(x):
    """
    Compute the arcsine of an observable x element-wise
    """
    return fast_math_scalar(x,12)

def arccos(x):
    """
    Compute the arccosine of an observable x element-wise
    """
    return fast_math_scalar(x,13)

def sinh(x):
    """
    """
    return fast_math_scalar(x,20)

def cosh(x):
    """
    """
    return fast_math_scalar(x,21)

def arcsinh(x):
    """
    """
    return fast_math_scalar(x,22)

def arccosh(x):
    """
    """
    return fast_math_scalar(x,23)

#### matrix operations


def trace(x):
    """
    Compute the trace of an observable

    Parameters
    ----------
    x : observa
        must be a square matrix
    """
    
    if (x.dims[0]!=x.dims[1]):
        raise InputError('unsupported operation for non-square matrices')
    [f, df] = math_tr(x.mean)
    return derobs([x], f, df)

def det(x):
    """
    Compute the determinant of an observable

    Parameters
    ----------
    x : observa
        must be a square matrix
    """
    
    if (x.dims[0]!=x.dims[1]):
        raise InputError('unsupported operation for non-square matrices')
    [f, df] = math_det(x.mean)
    return derobs([x], f, df)
    
def inv(x):
    """
    Compute the inverse of an observable

    Parameters
    ----------
    x : observa
        must be a square matrix
    """
    
    if (x.dims[0]!=x.dims[1]):
        raise InputError('unsupported operation for non-square matrices')
    [f, df] = math_inv(x.mean)
    return derobs([x], f, df)


def derobs(inps,func,dfunc=None):
    """ Compute the derived observable from a given function. 
    It is highly recommended to avoid the usage of this function
    as much as possible

    Parameters
    ----------
    inps : observa or list of observa
    func : function
        callable function with one argument
    dfunc : list of arrays, optional
        every element of the list corresponds to the derivative of 
        func with respect to the input observables inp 
        evaluated at the central values of inps. They are expected
        to be 4D arrays, with outer dimensions corresponding 
        to the (i,j) element of func and inner dimensions corresponding
        to the (k,l) element of the input observable.
        If not passed the gradient is computed numerically. 

    Returns
    -------
    x : class observa
        an observa with central value func(x) and error propagated 
        from inps using the derivatives of func

    Examples
    --------
    >>> inps = [obs1, obs2] # obs1 and obs2 are scalar observables
    >>> func = lambda x: x[0]*x[1]
    >>> dfunc = [ numpy.array([[x[1].mean]]), numpy.array([[x[0].mean]]) ]
    >>> res = derobs(inps, func, dfunc)
    
    """

    if not isinstance(inps,list):
        inps = [inps]
    if not isinstance(inps[0],observa):
        raise InputError('Unexpected type of input')

    all_mean = [i.mean for i in inps]
    if callable(func)==True:
        new_mean = func(all_mean)
    else:
        if not isinstance(func,numpy.ndarray):
            raise InputError('Unexpected function')
        else:
            new_mean = numpy.array(func)

    # checks new_mean has right properties
    if numpy.ndim(new_mean)!=2:
        raise ValueError('Unexpected function (wrong format)')
    else:
        if not isinstance(new_mean,numpy.ndarray):
            raise ValueError('Unexpected function (wrong format)')
        else:
            if not isinstance(new_mean[0],numpy.ndarray):
                raise ValueError('Unexpected function (wrong format)')

    new_dims = numpy.shape(new_mean)
    res = merge_observa(new_dims,inps)
    res.mean = new_mean
    res.dims = new_dims

    grad = []
    for i in range(len(inps)):
        if dfunc is None:
            grad.append( numerical_derivative(all_mean,func,i,inps[i].naive_err()) )
        else:
            grad.append( numpy.array(dfunc[i]) )

    for ed in res.edata:        
        for i in range(len(inps)):

            try:
                ie1 = inps[i].eid.index(ed.id)
                for rd in ed.rdata:
                    try:
                        ir1 = inps[i].edata[ie1].rid.index(rd.id)

                        d0 = inps[i].edata[ie1].rdata[ir1].idx[0]-rd.idx[0]
                        d1 = rd.idx[-1]-inps[i].edata[ie1].rdata[ir1].idx[-1]

                        if (d0==0 and d1==0):
                            tmp_data = inps[i].edata[ie1].rdata[ir1].data
                        else:
                            tmp_data = pad_with_zeros(inps[i].edata[ie1].rdata[ir1].data, d0, d1)
                        rd.data = rd.data + numpy.einsum('lmij,ijk->lmk',grad[i],tmp_data)
                    except:
                        pass
            except:
                pass

    for cd in res.cdata:
        for i in range(len(inps)):
            cdi = inps[i].find_cdata(cd)
            if (cdi!=None):
                cd.cov_grad += numpy.einsum('ijkl,klnm->ijnm',grad[i], cdi.cov_grad)
    return res


def numerical_derivative(x,f,i,dxi):
    old_dims = numpy.shape(x[i])
    shift1 = [ numpy.array(xx) for xx in x ]
    shift2 = [ numpy.array(xx) for xx in x ]

    tmp_grad = []
    for k in range(old_dims[0]):
        h = []
        for l in range(old_dims[1]):            
            shift1[i][k,l] = shift1[i][k,l] + dxi[k,l]
            shift2[i][k,l] = shift2[i][k,l] - dxi[k,l]

            h.append( (f(shift1) - f(shift2))/(2.*dxi[k,l]) )
            
            shift1[i][k,l] = shift1[i][k,l] - dxi[k,l]
            shift2[i][k,l] = shift2[i][k,l] + dxi[k,l]

        tmp_grad.append( h )
    return numpy.einsum('klij->ijkl',tmp_grad)


def merge_observa(new_dims, inps):
    res = observa()
    res.dims = new_dims
    res.eid = list(inps[0].eid)
    res.cid = list(inps[0].cid)
    ens_names = [ed.name for ed in inps[0].edata]
    cdat_names = [cd.name for cd in inps[0].cdata]
    for i in inps:
        [res.eid, ens_names] = double_union(res.eid, i.eid, ens_names, [ed.name for ed in i.edata])
        [res.cid, cdat_names] = double_union(res.cid, i.cid, cdat_names, [cd.name for cd in i.cdata])
    # for each ensemble merges rid and rdata of replica, data is set zeros
    for e in res.eid:
        ed = edata()
        ed.id = e
        ed.dims = new_dims
        ed.name = ens_names[res.eid.index(e)]

        # computes the union of id's of all replica found in all observables for this ensemble
        for i in inps:
            edi = i.find_edata(ed)
            if (edi!=None):
                if not ed.rid:
                    ed.rid.extend( edi.rid )
                    rep_names = [rd.name for rd in edi.rdata]
                else:
                    [ed.rid, rep_names] = double_union(ed.rid, edi.rid, rep_names, [rd.name for rd in edi.rdata])
        ed.R = len(ed.rid)

        # creates replica fields 
        for r in ed.rid:
            ed.rdata.append( rdata(r, rep_names[ed.rid.index(r)], new_dims) )

        # scans all replica for this ensemble in all observable and merges idx fields and initializes data to zero
        for rd in ed.rdata:
            tmp_idx = []

            for i in inps:
                try:
                    ie1 = i.eid.index(e)
                    ir1 = i.edata[ie1].rid.index(rd.id)
                    if not tmp_idx:
                        tmp_idx.extend( i.edata[ie1].rdata[ir1].idx )
                    else:
                        tmp_idx = union(tmp_idx, i.edata[ie1].rdata[ir1].idx)
                except:
                    pass
            
            rd.fill(tmp_idx)

        res.edata.append( ed )
    
    # checks cdata in common and merges covariance matrices, with zero gradients
    for c in res.cid:
        cd = cdata(c, cdat_names[res.cid.index(c)], res.dims)
        flag=False
        for i in inps:
            cdi = i.find_cdata(cd)
            if (cdi!=None):
                if not flag:
                    cd.create(cdi.cov)
                    flag=True
                else:
                    if not numpy.array_equal(cd.cov,cdi.cov):
                        raise ValueError('incompatible cdata')
        res.cdata.append(cd)
    return res
