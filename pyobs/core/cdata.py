#################################################################################
#
# cdata.py: definition and properties of the inner class cdata
# Copyright (C) 2020-2021 Mattia Bruno
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


class cdata:
    def __init__(self, cov, mask):
        if numpy.ndim(cov) == 1:
            self.cov = numpy.diag(numpy.array(cov, dtype=numpy.float64))
        else:
            self.cov = numpy.array(cov, dtype=numpy.float64)
        self.mask = mask
        self.size = len(mask)
        self.grad = numpy.zeros((self.size, numpy.shape(cov)[0]), dtype=numpy.float64)
        for a in mask:
            ia = mask.index(a)
            self.grad[ia,a] = 1.0

    def _apply_grad(self):
        if self.grad is None:
            return self.cov
        return self.grad @ self.cov @ self.grad.T
    
    def axpy(self, grad, cd):
        grad.apply(self.grad, self.mask, None, cd.grad, cd.mask)
#         self.cov += g @ cd.cov @ g.T

    def sigmasq(self):
        return numpy.diag(self._apply_grad())

    def assign(self, mask, cd):
        n = numpy.shape(cov)[0]
        self.grad[numpy.ix_[mask, range(n)]] = cd.grad
        self.cov = numpy.array(cd.cov)
#         self.cov[numpy.ix_[mask, mask]] = cd.cov
