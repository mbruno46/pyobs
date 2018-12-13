import numpy
import matplotlib.pyplot as plt

from .vwerr import uwerr, uwerr_texp
from .core.utils import fill_holes, irregular_measurements
from .jerr import jackknife_error

class edata:
    def __init__(self):
        self.id = 0
        self.name = ''
        self.dims = ()
        self.R = 0
        self.rid = []
        self.rdata = []

    # all arguments are assumed to be lists
    def create(self,eid,ename,rid,rname,idx,data,dims):
        self.id = eid
        self.name = ename
        size=numpy.prod(dims)
        self.R = len(rid)
        self.dims = dims
        if (sorted(rid)!=rid):
            raise InputError('rid list must be sorted')
        else:
            self.rid = rid

        # computes the global mean and the mean per replica
        n=0
        mean = numpy.array(numpy.zeros(size), dtype=numpy.double)
        for ir in range(self.R):
            self.rdata.append( rdata(rid[ir],rname[ir],dims) )
            mean = mean + self.rdata[-1].mean(data[ir])
            n=n+len(idx[ir])
        mean = mean * (1.0/n)

        # fills rdata with idx of configs and fluctuations of data
        for ir in range(self.R):
            self.rdata[ir].create(idx[ir], data[ir], mean)

        return numpy.reshape(mean, self.dims)

    def clone(self,full):
        res = edata()
        res.id = self.id
        res.name = self.name
        res.dims = self.dims
        res.R = self.R
        res.rid = list(self.rid)
        for rd in self.rdata:
            res.rdata.append( rd.clone(full) )
        return res

    def find_rdata(self,rd0):
        for rd in self.rdata:
            if (rd.id==rd0.id):
                if (rd.name==rd0.name):
                    return rd
                else:
                    raise
        return None

    def uwerr(self,plot,pfile,pars=(1.5,None)):
        sigma = numpy.zeros(self.dims)
        tau = numpy.zeros(self.dims)
        dtau = numpy.zeros(self.dims)
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                res = uwerr([rd.data[i,j] for rd in self.rdata], [rd.ncnfg for rd in self.rdata], plot, (self.name,i,j,pfile), pars[0], pars[1])
                sigma[i,j] = res[0]
                tau[i,j] = res[1]
                dtau[i,j] = res[2]
        return [sigma, tau, dtau]

    def uwerr_texp(self,plot,pfile,pars=(1.5,0.0,2,0,None)):
        sigma = numpy.zeros(self.dims)
        tau = numpy.zeros(self.dims)
        dtau = numpy.zeros(self.dims)
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                res = uwerr_texp([rd.data[i,j] for rd in self.rdata], [rd.ncnfg for rd in self.rdata], plot, (self.name,i,j,pfile), pars[0], pars[1], pars[2], pars[3], pars[4])
                sigma[i,j] = res[0]
                tau[i,j] = res[1]
                dtau[i,j] = res[2]
        return [sigma, tau, dtau]

    def naive_err(self):
        sigma = numpy.zeros(self.dims)
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                res = uwerr([rd.data[i,j] for rd in self.rdata], [rd.ncnfg for rd in self.rdata], 
                            False, (self.name,i,j,''), 1.5, 0)
                sigma[i,j] = res[0]
        return sigma

    def jkerr(self,bsize,mean):
        return jackknife_error(mean,self.rdata[0].data,self.rdata[0].ncnfg,self.dims,bsize)

    def plotter(self):
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                plt.figure()
                plt.title('Ensemble ' + self.name + ' (i,j)=('+str(i)+','+str(j)+')')
                plt.xlabel('configuration index')
                for ir in range(self.R):
                    xax = numpy.array(self.rdata[ir].idx) - self.rdata[ir].idx[0]
                    plt.plot(xax, self.rdata[ir].data[i,j,:],label=self.rdata[ir].name)
                plt.legend()
                plt.show()


class rdata:
    def __init__(self,rid,rname,dims):
        self.dims = dims
        self.id = rid
        self.name = rname
        #self.ncnfg
        #self.idx
        #self.data

    def mean(self,data):
        # computes the mean
        size = numpy.prod(self.dims)
        n = len(data)/size
        datasum = numpy.sum(numpy.reshape(data, (n,size)), 0)
        self.mean = numpy.reshape(datasum, self.dims) * (1.0/n)
    
        return datasum

    def create(self,idx,data,mean):
        subtraction = numpy.array([mean for _ in range(len(idx))]).flatten()
        delta = data - subtraction

        # check if data is irregular and fix it    
        if (irregular_measurements(idx)==True):
            tmp = fill_holes(idx, delta)
        else:
            tmp = [idx, delta]

        self.ncnfg = len(idx)
        self.idx = list(tmp[0])
        self.data = numpy.zeros(self.dims+(self.ncnfg,))
        for k in range(self.ncnfg):
            for i in range(self.dims[0]):
                for j in range(self.dims[1]):
                    self.data[i,j,k] = tmp[1][ (k*self.dims[1] + i)*self.dims[0] + j]
        #self.data = numpy.reshape(tmp[1], (self.ncnfg,)+self.dims[::-1]).T

    def clone(self,full):
        res = rdata(self.id,self.name,self.dims)
        res.ncnfg = self.ncnfg
        res.idx = list(self.idx)
        if (full==True):
            res.data = numpy.array(self.data)
        else:
            res.data = numpy.zeros(self.dims + (self.ncnfg,))
        return res

    def fill(self,idx,data=None):
        self.ncnfg = len(idx)
        self.idx = list(idx)
        if (data==None):
            self.data = numpy.zeros(self.dims + (self.ncnfg,))
        else:
            self.data = numpy.array(data)


class cdata:
    def __init__(self,cid, cname, dims):
        self.id = cid
        self.name = cname
        self.dims = tuple(dims) # dimensions of observable
    
    def create(self,cov,grad=None): 
        self.cov = numpy.array(cov)
        self.cov_dims = self.cov.shape[0:2]
        if grad is None:
            self.cov_grad = numpy.zeros(self.dims+self.cov_dims)
        else:
            self.cov_grad = numpy.array(grad)

    def clone(self,full):
        cd = cdata(self.id, self.name, self.dims)
        if (full==True):
            cd.create(self.cov,self.cov_grad)
        else:
            cd.create(self.cov)
        return cd

    def sigma(self):
        sig = numpy.zeros(self.dims)
        for a in range(self.dims[0]):
            for b in range(self.dims[1]):
                sig[a,b] = numpy.einsum('ij,ijkl,kl',self.cov_grad[a,b], self.cov, self.cov_grad[a,b])
        return sig

