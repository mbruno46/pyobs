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

__all__ = ['delta']

def create_fft_data(data,idx,shape,fft_ax):
    v=numpy.prod(shape)
    tmp = None
    if type(idx) is range:
        tmp = numpy.array(data,dtype=numpy.float64)
    else:
        tmp = numpy.zeros((v,),dtype=numpy.float64)
        for j in range(len(idx)):
            tmp[idx[j]-idx[0]] = data[j]
            
    if len(fft_ax)>1:
        tmp = numpy.reshape(tmp,shape)
    # in-place, even if it adds one element at the end
    return numpy.fft.rfftn(tmp,s=shape,axes=fft_ax)

# NOTE: if lat is integer then Monte carlo assumed, ie open BC at the boundary
# of the markov chain. if lat is a list then periodic BC assumed in all dirs, 
# even if lat is 1D
def conv_ND(data,idx,lat,xmax,a=0,b=None):
    if isinstance(lat,(int,numpy.int)):
        shape=(2*lat,)
        lat=[lat]
        ismf = False
    else:
        shape=tuple(lat)
        ismf = True
    
    D=len(lat)
    fft_ax=tuple(range(D))
    v=numpy.prod(lat)    
    
    aux = []
    for index in [a,b]:
        if index is None:
            continue
        aux += [create_fft_data(data[index,:],idx,shape,fft_ax)]

    if len(aux)==1:
        aux[0]*=aux[0].conj()
        aux  += [numpy.fft.irfftn(aux[0],s=shape,axes=fft_ax)]
    else:
        aux[0]*=aux[1].conj()
        aux[1]=numpy.fft.irfftn(aux[0].real,s=shape,axes=fft_ax)      

    if ismf:
        aux[1]=numpy.reshape(aux[1],(v,))
        return pyobs.core.mftools.intrsq(aux[1],lat,xmax)
    
    g=numpy.array(aux[1][0:xmax])
    return numpy.around(g,decimals=15)

    
# note that data, idx, N must be lists with 1 or 2 elements
# if 2 elements then it's like off diagonal element of cov matrix
# def conv_1D(data,idx,N,wmax,a=0,b=None):
#     aux = []
        
#     if (M==1):
#         aux[0]*=aux[0].conj()
#         aux += [numpy.fft.irfft(aux[0])]
#     else:
#         aux[0]*=aux[1].conj()
#         aux[1]=numpy.fft.irfft(aux[0].real)
        
#     g=numpy.array(aux[1][0:wmax])
#     return g

# def conv_1D_offdiag(data,idx,wmax):
#     if type(idx1) is range:
#         aux1=numpy.array(data1,dtype=numpy.float64)
#         aux2=numpy.array(data2,dtype=numpy.float64)
#     else:
#         aux1=numpy.zeros((N+1,),dtype=numpy.float64)
#         aux2=numpy.zeros((N+1,),dtype=numpy.float64)
#         for i in range(len(idx)):
#             aux1[idx[i]-idx[0]] = data1[i]
#             aux2[idx[i]-idx[0]] = data2[i]

#     # in-place, even if it adds one element at the end
#     aux1=numpy.fft.rfft(aux1,n=2*N)
#     aux2=numpy.fft.rfft(aux2,n=2*N)
#     aux1*=aux2.conj()
#     aux2=numpy.fft.irfft(aux1.real)
   
#     g=numpy.array(aux2[0:wmax])
#     return g
    
# def conv_ND(data,idx,lat,rrmax):
#     D=len(lat)
#     fft_ax=tuple(range(D))
#     shape=tuple(lat)
#     v=numpy.prod(lat)
    
#     if (len(data)!=len(idx)):
#         raise pyobs.PyobsError('conv_ND: err0')
#     if (len(data)>2):
#         raise pyobs.PyobsError('conv_ND: err1')
        
#     M = len(data)
#     aux = []
    
#     for i in range(M):
#         tmp = None
#         if type(idx[i]) is range:
#             tmp = numpy.array(data[i],dtype=numpy.float64)
#         else:
#             tmp = numpy.zeros((v,),dtype=numpy.float64)
#             for j in range(len(idx[i])):
#                 tmp[idx[i][j]] = data[i][j]
#         tmp = numpy.reshape(tmp,shape)
        
#         # in-place, even if it adds one element at the end
#         aux += [numpy.fft.rfftn(tmp,s=shape,axes=fft_ax)]
#         del tmp
        
#     if (M==1):
#         aux[0]*=aux[0].conj()
#         aux += [numpy.fft.irfftn(aux[0],s=shape,axes=fft_ax)]
#     else:
#         aux[0]*=aux[1].conj()
#         aux[1]=numpy.fft.irfftn(aux[0].real,s=shape,axes=fft_ax)      
    
#     aux[1]=numpy.reshape(aux[1],(v,))
#     return pyobs.core.mftools.intrsq(aux[1],lat,rrmax)


class delta:
    def __init__(self,mask,idx,data=None,mean=None,lat=None):
        # idx is expected to be a list or range
        self.size = len(mask)
        self.mask = [m for m in mask]
        self.it=0
        if lat is None:
            self.lat = None
        else:
            self.lat = numpy.array(lat,dtype=numpy.int32)

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

    def copy(self):
        res = delta(self.mask,self.idx,lat=self.lat)
        res.delta = numpy.array(self.delta)
        return res
        
    def ncnfg(self):
        if type(self.idx) is range:
            return self.n
        else:
            return int(self.idx[-1]-self.idx[0])+1 #first and last config included!
            
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
        #self.delta[:,jlist] += grad.apply(d,self.mask)
        grad.apply(self.delta, self.mask, jlist, d.delta, d.mask)
        
    def assign(self,submask,rd):
        if len(submask)!=len(rd.mask):
            raise pyobs.PyobsError('Dimensions do not match in assignment')
        a = numpy.nonzero(numpy.in1d(self.mask,submask))[0]
        self.delta[a,:] = rd.delta

#     def gamma(self,xmax):
#         g=numpy.zeros((self.size,xmax))
#         ones = numpy.reshape(numpy.ones(self.n),(1,self.n))
#         isMC = self.lat is None

#         if isMC:
#             m = conv_ND(ones,self.idx,self.ncnfg(),xmax)
#         else:
#             rrmax = xmax
#             v=self.vol()
#             if v==len(self.idx):
#                 m = [v]*rrmax
#             else:
#                 m = conv_ND(ones,self.idx,self.lat,rrmax)
#                 Sr = pyobs.core.mftools.intrsq(numpy.ones(v),self.lat,rrmax)
#                 Sr = Sr + 1*(Sr==0.0)
#                 m /= Sr
        
#         for a in range(self.size):
#             g[a,:] = conv_ND(self.delta,self.idx,self.ncnfg() if isMC else self.lat,xmax,a)
#         return [m,g]
#         if self.lat is None:
#             wmax = xmax
#             m = conv_1D([numpy.ones(self.n)],[self.idx],[self.ncnfg()],wmax)
#             for a in range(self.size):
#                 g[a,:] = conv_1D([self.delta[a,:]],[self.idx],[self.ncnfg()],wmax)
#         else:
#             rrmax = xmax
#             v=self.vol()
#             if v==len(self.idx):
#                 m = [v]*rrmax
#             else:
#                 m = conv_ND([numpy.ones(self.n)],[self.idx],self.lat,rrmax)
#                 Sr = pyobs.core.mftools.intrsq(numpy.ones(v),self.lat,rrmax)
#                 Sr = Sr + 1*(Sr==0.0)
#                 m /= Sr
#             for a in range(self.size):
#                 g[a,:] = conv_ND([self.delta[a,:]],[self.idx],self.lat,rrmax)
#         return [m, g]
    

    def gamma(self,xmax,a,b=None):
        ones = numpy.reshape(numpy.ones(self.n),(1,self.n))
        isMC = self.lat is None
        
        if isMC:
            m = conv_ND(ones,self.idx,self.ncnfg(),xmax)
        else:
            rrmax = xmax
            v=self.vol()
            if v==len(self.idx):
                m = [v]*rrmax
            else:
                m = conv_ND(ones,self.idx,self.lat,rrmax)
                Sr = pyobs.core.mftools.intrsq(numpy.ones(v),self.lat,rrmax)
                Sr = Sr + 1*(Sr==0.0)
                m /= Sr
            
        g = conv_ND(self.delta,self.idx,self.ncnfg() if isMC else self.lat,xmax,a,b)
        return [m, g]
    
    
    # replica ensemble utility functions
    def wmax(self):
        return self.ncnfg()//2
    
    def rrmax(self):
        return int(numpy.sum((self.lat/2)**2)+1)

    def vol(self):
        return numpy.prod(self.lat)
