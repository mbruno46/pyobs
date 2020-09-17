#################################################################################
#
# derobs.py: implementation of the core function for derived observables
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
import pyobs

from .data import rdata, mfdata
from .cdata import cdata


def merge_idx(idx1,idx2):
    if (type(idx1) is range) and (type(idx2) is range):
        id0=min([idx1.start,idx2.start])
        id1=max([idx1.stop, idx2.stop])
        id2=min([idx1.step, idx2.step])
        return range(id0,id1,id2)
    else:
        u=set(idx1)
        return list(sorted(u.union(idx2)))

    
def derobs(inps,mean,grads,desc=None):
    t0=time()
    pyobs.check_type(inps,'inps',list)
    pyobs.check_type(mean,'mean',numpy.ndarray,int,float,numpy.float32,numpy.float64)
    pyobs.check_type(grads,'grads',list)
    if len(inps)!=len(grads):
        error_msg('Incompatible inps and grads')
    if desc is None:
        desc=', '.join(set([i.description for i in inps]))
    res = pyobs.obs(desc=desc)
    if isinstance(mean,(int,float,numpy.float32,numpy.float64)):
        res.mean = numpy.reshape(mean,(1,))
    else:
        res.mean = numpy.array(mean)
    res.shape = numpy.shape(res.mean)
    res.size = numpy.prod(res.shape)
    
    def core(datatype):
        allkeys = []
        for i in inps:
            for dn in i.__dict__[datatype]:
                if not dn in allkeys:
                    allkeys.append(dn)
        
        for key in allkeys:
            new_idx = []
            new_mask = []
            lat = None
            for i in range(len(inps)):
                if key in inps[i].__dict__[datatype]:
                    data = inps[i].__dict__[datatype][key]
                    oid = numpy.array(data.mask,dtype=numpy.int)
                    h = numpy.sum(grads[i][:,oid]!=0.0,axis=1)
                    if numpy.sum(h)>0:
                        new_mask += list(numpy.arange(res.size)[h>0])
                        if not new_idx:
                            new_idx = data.idx
                        else:
                            new_idx = merge_idx(new_idx, data.idx)
                        if datatype=='mfdata':
                            if lat is None:
                                lat = data.lat
                            else:
                                if numpy.any(lat != data.lat):
                                    error_msg(f'Unexpected lattice size for master fields with same tag')
            if len(new_mask)>0:
                if datatype=='rdata':
                    res.__dict__[datatype][key] = rdata(list(set(new_mask)), new_idx)
                else:
                    res.__dict__[datatype][key] = mfdata(list(set(new_mask)), new_idx, lat)
                for i in range(len(inps)):
                    if key in inps[i].__dict__[datatype]:
                        res.__dict__[datatype][key].axpy(grads[i],inps[i].__dict__[datatype][key])
        
    core('rdata')
    core('mfdata')
    
    res.edata = []
    for key in res.rdata:
        name = key.split(':')[0]
        if not name in res.edata:
            res.edata.append(name)
    
    res.mfname = []
    for key in res.mfdata:
        name = key.split(':')[0]
        if not name in res.mfname:
            res.mfname.append(name)
    
    res.cdata = {}
    allkeys = []
    for i in inps:
        for cd in i.cdata:
            if not cd in allkeys:
                allkeys.append(cd)
    for key in allkeys:
        for i in range(len(inps)):
            if key in inps[i].cdata:
                if not key in res.cdata:
                    d=inps[i].cdata[key].cov.shape[0]
                    res.cdata[key] = cdata(numpy.zeros((res.size,d)),inps[i].cdata[key].cov)
                res.cdata[key].axpy(grads[i], inps[i].cdata[key])

    pyobs.memory.add(res)
    if pyobs.is_verbose('derobs'):
        print(f'derobs executed in {time()-t0:g} secs')
    return res


def num_grad(x,f,eps=2e-4):
    if isinstance(x,pyobs.obs):
        x0 = x.mean
    else:
        x0 = numpy.array(x)
        
    s = x0.shape
    n = numpy.size(x0)
    dx = numpy.zeros((n,))
    
    f0 = f(x0)
    m = numpy.size(f0)
    df = numpy.zeros((m,n))
    
    for i in range(n):
        dx[i] = 1.0
        dx = numpy.reshape(dx,s)
        
        fp = f(x0+x0*eps*dx)
        fm = f(x0-x0*eps*dx)
        fpp = f(x0+x0*2.*eps*dx)
        fmm = f(x0-x0*2.*eps*dx)
        
        tmp = 2./3. * (fp - fm) - 1/12 * (fpp - fmm)
        df[:,i] = tmp.flatten() / (x0.flatten()[i]*eps)
        
        dx = numpy.reshape(dx,(n,))
        dx[i] = 0.0
        
    return df

def num_hess(x0,f,eps=2e-4):
    f0=f(x0)
    m=numpy.size(f0)
    n=numpy.size(x0)
    
    ddf = numpy.zeros((m,n,n))
    for i in range(m):
        for j in range(n):
            ddf[i,j,:] = num_grad(x0, lambda x: num_grad(x, f)[i,j])[0]
            
    return ddf

def errbias4(x,f):
    pyobs.check_type(x,'x',pyobs.obs)
    [x0,dx0] = x.error()
    bias4 = numpy.zeros((x.size,))
    hess = num_hess(x0, f)
    
    def core(data):
        oid = numpy.array(x.rdata[key].mask)
        idx = numpy.ix_(oid,oid)
        d2 = numpy.einsum('abc,bj,cj->aj',hess[:,idx[0],idx[1]],x.rdata[key].delta,x.rdata[key].delta)
        dd2 = numpy.sum(d2,axis=1)
        return dd2**2 /x.rdata[key].n**4
    
    for key in x.rdata:
        bias4 +=core(x.rdata)
        
    for key in x.mfdata:
        bias4 +=core(x.mfdata)       
    
    return numpy.sqrt(bias4)
