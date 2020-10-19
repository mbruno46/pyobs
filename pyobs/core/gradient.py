#################################################################################
#
# gradient.py: implementation of the core function for gradient of functions
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

import pyobs
import numpy

# grad is Na x Ni matrix
class gradient:
    def __init__(self,g,shape,gtype='full'):
        self.Na=numpy.size(g(numpy.ones(shape)))
        self.Ni=numpy.prod(shape)
        self.gtype = gtype
        
        if gtype is 'full':
            gsh = (self.Na,self.Ni)
        elif gtype is 'diag':
            gsh = self.Na
            if self.Na!=self.Ni:
                raise pyobs.PyobsError('diagonal gradient error')
        elif gtype is 'slice':
            gsh = self.Na
        else:
            raise pyobs.PyobsError('gradient error')
            
        self.grad = numpy.zeros(gsh,dtype=numpy.float64)
        
        if gtype is 'full':
            dx = numpy.zeros(self.Ni)
            for i in range(self.Ni):
                dx[i] = 1.0
                self.grad[:,i] = numpy.reshape(g(numpy.reshape(dx,shape)),self.Na)
                dx[i] = 0.0
        elif gtype is 'diag':
            self.grad = g(numpy.ones(shape)).flatten()
        elif gtype is 'slice':
            self.grad = numpy.reshape(g(numpy.arange(self.Ni).reshape(shape)),self.Na)
            
    def get_mask(self,mask):
        idx = numpy.array(mask,dtype=numpy.int32)
        if self.gtype is 'full':
            h = numpy.sum(self.grad[:,idx]!=0.0,axis=1)
            if numpy.sum(h)>0:
                return list(numpy.arange(self.Na)[h>0])
            else:
                return None
        elif self.gtype is 'diag':
            return mask
        elif self.gtype is 'slice':
            idx = numpy.nonzero(numpy.in1d(self.grad,mask))
            if idx[0].size>0:
                return list(numpy.arange(self.Na)[idx])
            else:
                return None
        
    def apply(self,icnfgs,dx,dy):
        if self.gtype is 'full':
            gvec = pyobs.slice_ndarray(self.grad, dx.mask, dy.mask)
            dx.delta[:,icnfgs] += gvec @ dy.delta
        elif self.gtype is 'diag':
            dx.delta[:,icnfgs] += self.grad[dy.mask,None] * dy.delta
        elif self.gtype is 'slice':
            idx = numpy.nonzero(numpy.in1d(self.grad,dy.mask))
            dx.delta[:,icnfgs] = dy.delta[idx,:]
            