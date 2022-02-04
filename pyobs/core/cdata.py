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
        n = numpy.shape(cov)[0]
        self.mask = mask
        self.size = len(self.mask)
        self.grad = numpy.zeros((self.size, n), dtype=numpy.float64)
        for a in self.mask:
            ia = self.mask.index(a)
            # special case cobs_scalar * obs_vector, then mask is vector but n=1
            if n == 1:
                self.grad[ia, 0] = 1.0
            else:
                self.grad[ia, a] = 1.0

    def copy(self):
        res = cdata(self.cov, self.mask)
        res.grad = numpy.array(self.grad)
        return res

    def axpy(self, grad, cd):
        self.grad[:, :] = 0.0
        grad.apply(self.grad, self.mask, None, cd.grad, cd.mask)

    def sigmasq(self, outer_shape):
        size = numpy.prod(outer_shape)
        tmp = numpy.diag(self.grad @ self.cov @ self.grad.T)
        out = numpy.zeros((size,))
        for a in self.mask:
            ia = self.mask.index(a)
            out[a] = tmp[ia]
        return numpy.reshape(out, outer_shape)

    def assign(self, submask, cd):
        pyobs.assertion(
            len(submask) == cd.size, "Dimensions do not match in assignment"
        )
        a = numpy.nonzero(numpy.in1d(self.mask, submask))[0]
        self.grad[a, :] = cd.grad

    def cholesky(self):
        L = numpy.linalg.cholesky(self.cov)
        n = numpy.shape(self.cov)[0]
        res = cdata(numpy.eye(n), self.mask)
        for ia in range(len(self.mask)):
            res.grad[ia, :] = self.grad[ia, :] @ L
        return res
