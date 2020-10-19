#################################################################################
#
# data.py: definition and properties of the class with the fluctuations
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
import pyobs

__all__ = ['rdata','mfdata']

def conv_1D(data,idx,N,wmax):
    if type(idx) is range:
        aux=numpy.array(data,dtype=numpy.float64)
    else:
        aux=numpy.zeros((N+1,),dtype=numpy.float64)
        for i in range(len(idx)):
            aux[idx[i]-idx[0]] = data[i]

    # in-place, even if it adds one element at the end
    aux=numpy.fft.rfft(aux,n=2*N)
    aux*=aux.conj()
    aux=numpy.fft.irfft(aux)
   
    g=numpy.array(aux[0:wmax])
    return g

def conv_1D_offdiag(data1,data2,idx,N,wmax):
    if type(idx) is range:
        aux1=numpy.array(data1,dtype=numpy.float64)
        aux2=numpy.array(data2,dtype=numpy.float64)
    else:
        aux1=numpy.zeros((N+1,),dtype=numpy.float64)
        aux2=numpy.zeros((N+1,),dtype=numpy.float64)
        for i in range(len(idx)):
            aux1[idx[i]-idx[0]] = data1[i]
            aux2[idx[i]-idx[0]] = data2[i]

    # in-place, even if it adds one element at the end
    aux1=numpy.fft.rfft(aux1,n=2*N)
    aux2=numpy.fft.rfft(aux2,n=2*N)
    aux1*=aux2.conj()
    aux2=numpy.fft.irfft(aux1.real)
   
    g=numpy.array(aux2[0:wmax])
    return g
    
def conv_ND(data,idx,lat,rrmax):
    D=len(lat)
    fft_ax=tuple(range(D))
    shape=tuple(lat)
    v=numpy.prod(lat)
    
    if type(idx) is range:
        aux=numpy.array(data,dtype=numpy.float64)
    else:
        aux=numpy.zeros((v,),dtype=numpy.float64)
        for i in range(len(idx)):
            aux[idx[i]] = data[i]
    aux=numpy.reshape(aux,shape)
    
    # in-place, even if it adds one element at the end
    aux=numpy.fft.rfftn(aux,s=shape,axes=fft_ax)
    aux*=aux.conj()
    aux=numpy.fft.irfftn(aux,s=shape,axes=fft_ax)
    
    aux=numpy.reshape(aux,(v,))
    return pyobs.core.mftools.intrsq(aux,lat,rrmax)

class delta:
    def __init__(self,mask,idx,data=None,mean=None):
        # idx is expected to be a list or range
        self.size = len(mask)
        self.mask = [m for m in mask]
        self.it=0
        
        if (type(idx) is list):
            dc = numpy.unique(numpy.diff(idx))
            if numpy.any(dc<0):
                raise pyobs.PyobsError(f'Unsorted idx')
            if len(dc)==1:
                self.idx = range(idx[0],idx[-1]+dc[0],dc[0])
            else:
                self.idx = list(idx)
        elif (type(idx) is range):
            self.idx = idx
        else:
            raise pyobs.PyobsError(f'Unexpected idx')
        self.n = len(self.idx)

        self.delta = numpy.zeros((self.size,self.n),dtype=numpy.float64)
        
        if not mean is None:
            self.delta = numpy.reshape(data,(self.n,self.size)).T - numpy.stack([mean]*self.n,axis=1)
        
    def ncnfg(self):
        if type(self.idx) is range:
            return self.n
        else:
            return int(self.idx[-1]-self.idx[0])
        
    def get_mask(self,a):
        if a in self.mask:
            return self.mask.index(a)
        else:
            return -1

    def start_idx(self):
        self.it=0
    
    def get_idx(self,index):
        if type(self.idx) is range:
            return self.idx.index(index)
        else:
            while (self.idx[self.it]!=index):
                self.it+=1
            return self.it
        
    def axpy(self,grad,d,hess=None):
        N=d.delta.shape[1]
        
        # prepare index list
        self.start_idx()
        d.start_idx()
        jlist=[]
        for i in range(N):
            k=d.idx[i]
            jlist.append(self.get_idx(k))
        jlist=numpy.array(jlist,dtype=numpy.int)
        
        # apply gradient
        self.delta[:,jlist] += grad.apply(d,self.mask)

    def assign(self,submask,rd):
        if len(submask)!=len(rd.mask):
            raise pyobs.PyobsError('Dimensions do not match in assignment')
        for i in range(len(submask)):
            j = self.mask.index(submask[i])
            self.delta[j,:] = rd.delta[i,:]
            
class mfdata(delta):
    def __init__(self,mask,idx,lat,data=None,mean=None):
        delta.__init__(self,mask,idx,data,mean)
        self.lat = numpy.array(lat,dtype=numpy.int32)
    
    def copy(self):
        res = mfdata(self.mask,self.idx,self.lat)
        res.delta = numpy.array(self.delta)
        return res
    
    def rrmax(self):
        return int(numpy.sum((self.lat/2)**2)+1)

    def vol(self):
        return numpy.prod(self.lat)
    
    def gamma(self,rrmax):
        g=numpy.zeros((self.size,rrmax))
        v=numpy.prod(self.lat)
        if v==len(self.idx):
            m = [v]*rrmax
        else:
            m = conv_ND(numpy.ones(self.n),self.idx,self.lat,rrmax)
            Sr = pyobs.core.mftools.intrsq(numpy.ones(v),self.lat,rrmax)
            Sr = Sr + 1*(Sr==0.0)
            m /= Sr
        for a in range(self.size):
            g[a,:] = conv_ND(self.delta[a,:],self.idx,self.lat,rrmax)
        return [m, g]        
        
    def gamma_norm(self,rrmax):
        g=numpy.zeros((self.size,rrmax))
        m = conv_ND(numpy.ones(self.n),self.idx,self.lat,rrmax)
        for a in range(self.size):
            g[a,:] = conv_ND(self.delta[a,:],self.idx,self.lat,rrmax)
        return [m, g]        
    
class rdata(delta):
    def wmax(self):
        return self.ncnfg()//2
    
    def copy(self):
        res = rdata(self.mask,self.idx)
        res.delta = numpy.array(self.delta)
        return res
    
    def gamma(self,wmax):
        g=numpy.zeros((self.size,wmax))
        m = conv_1D(numpy.ones(self.n),self.idx,self.ncnfg(),wmax)
        for a in range(self.size):
            g[a,:] = conv_1D(self.delta[a,:],self.idx,self.ncnfg(),wmax)
        return [m, g]
