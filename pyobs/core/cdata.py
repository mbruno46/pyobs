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
    def __init__(self,grad,cov):
        self.grad = grad
        if numpy.ndim(cov)==1:
            self.cov = numpy.diag(cov)
        else:
            self.cov = numpy.array(cov)
    
    def axpy(self,grad,cd):
        self.grad += grad @ cd.grad
        
    def sigmasq(self):
        (na,nb) = self.grad.shape
        sigma = numpy.zeros((na,))
        for ia in range(na):
            sigma[ia] = self.grad[ia,:] @ self.cov[:,:] @ self.grad.T[:,ia]
        return sigma
    
    def reduce(self):
        self.cov = self.grad @ self.cov @ self.grad.T
        self.grad = numpy.eye(self.cov.shape[0])
        
