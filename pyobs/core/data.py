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
from time import time
from .utils import error_msg, sort_data, is_verbose
import pyobs.core.mftools as mftools

#def intersect_idx(*idx):
#    # if all idx are ranges optimize
#    flag=0
#    for arg in idx:
#        if not type(arg) is range:
#            flag+=1
#    if flag==0:
#        idx0=max([arg.start for arg in idx])
#        idx1=min([arg.stop for arg in idx])
#        idx2=min([arg.step for arg in idx])
#        if idx1>idx0:
#            return range(idx0,idx1,idx2)
#        else:
#            return []
#    else:
#        out=set()
#        for arg in idx:
#            out=out.intersect(set(arg))
#        return list(out)
        
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
    return mftools.intrsq(aux,lat,rrmax)

class delta:
    def __init__(self,mask,idx,data=None,mean=None):
        # idx is expected to be a list or range
        self.size = len(mask)
        self.mask = [m for m in mask]
        self.it=0
        
        if (type(idx) is list):
            dc = numpy.unique(numpy.diff(idx))
            if numpy.any(dc<0):
                error_msg(f'Unsorted idx')
            if len(dc)==1:
                self.idx = range(idx[0],idx[-1]+dc[0],dc[0])
            else:
                self.idx = list(idx)
        elif (type(idx) is range):
            self.idx = idx
        else:
            error_msg(f'Unexpected idx')
        self.n = len(self.idx)

        self.delta = numpy.zeros((self.size,self.n),dtype=numpy.float64)
        #self.delta2 = numpy.zeros((self.size,self.n),dtype=numpy.float32)
        
        if not mean is None:
            data=numpy.reshape(data,(self.n,self.size))
            for i in range(self.n):
                for a in range(self.size):
                    self.delta[a,i] = data[i,a] - mean[a]
        
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
        dt0=dt1=0.0
        N=d.delta.shape[1]
        
        # prepare index list
        self.start_idx()
        d.start_idx()
        jlist=[]
        for i in range(N):
            k=d.idx[i]
            jlist.append(self.get_idx(k))
        jlist=numpy.array(jlist,dtype=numpy.int)
        
        for a in self.mask:
            ia=self.get_mask(a)
            
            t0=time()
            gvec = []
            ib=[]
            for b in d.mask:
                if abs(grad[a,b])>1e-14:
                    gvec.append(grad[a,b])
                    ib.append(d.get_mask(b))
            if ib:
                ib=numpy.array(ib,dtype=numpy.int)
                gvec=numpy.array(gvec,dtype=numpy.float64)
                self.delta[ia,jlist]  += gvec @ d.delta[ib,:]
            
            dt0=time()-t0

        if is_verbose('data.axpy'):
            print(f'data.axpy executed in {dt0:g} secs')

            
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
            Sr = mftools.intrsq(numpy.ones(v),self.lat,rrmax)
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
    
    
def merge_idx(idx1,idx2):
    if (type(idx1) is range) and (type(idx2) is range):
        id0=min([idx1.start,idx2.start])
        id1=max([idx1.stop, idx2.stop])
        id2=min([idx1.step, idx2.step])
        return range(id0,id1,id2)
    else:
        u=set(idx1)
        return list(sorted(u.union(idx2)))
