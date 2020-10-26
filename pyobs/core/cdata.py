#################################################################################
#
# cdata.py: definition and properties of the inner class cdata
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

class cdata:
    def __init__(self, cov):
        if numpy.ndim(cov)==1:
            self.cov = numpy.diag(numpy.array(cov, dtype=numpy.float64))
        else:
            self.cov = numpy.array(cov, dtype=numpy.float64)
        
    def axpy(self,grad,cd):
        n = len(self.cov)
        g = numpy.zeros((n,n))
        m = len(cd.cov)
        grad.apply(g, range(n), None, numpy.eye(m), range(m))
        self.cov += g @ cd.cov @ g.T
        
    def sigmasq(self):
        return numpy.diag(self.cov)
    
    #def reduce(self):
    #    self.cov = self.grad @ self.cov @ self.grad.T
    #    self.grad = numpy.eye(self.cov.shape[0])
    
    def copy(self,cd):
        #self.grad = numpy.array(cd.grad)
        self.cov = numpy.array(cd.cov)

    def assign(self,mask,cd):
        self.cov[numpy.ix_[mask,mask]] = cd.cov