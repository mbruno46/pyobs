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
    def __init__(self, mask, cov):
        self.mask = mask
        self.size = len(mask)
        if numpy.ndim(cov) == 1:
            self.cov = numpy.diag(numpy.array(cov, dtype=numpy.float64))
        else:
            self.cov = numpy.array(cov, dtype=numpy.float64)
        pyobs.assertion(self.size == numpy.shape(self.cov)[0],"Mismatch between mask and cov")
        self.grad = None
        
    def _apply_grad(self):
        if self.grad is None:
            return self.cov
        return self.grag @ self.cov @ self.grad.T
    
    def axpy(self, grad, cd):
        n = len(self.cov)
        m = len(cd.cov)
        self.grad = numpy.zeros((n, m))
        grad.apply(self.grad, range(n), None, cd.grad, range(m))
#         self.cov += g @ cd.cov @ g.T

    def sigmasq(self):
        return numpy.diag(self._apply_grad())

    def assign(self, mask, cd):
        n = numpy.shape(cov)[0]
        self.grad[numpy.ix_[mask, range(n)]] = cd.grad
        self.cov = numpy.array(cd.cov)
#         self.cov[numpy.ix_[mask, mask]] = cd.cov
