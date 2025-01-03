#################################################################################
#
# gradient.py: implementation of the core function for gradient of functions
# Copyright (C) 2020-2025 Mattia Bruno
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
from scipy.sparse import dia_matrix


# grad is Na x Ni matrix
class gradient:
    def __init__(self, g, x0=None, gtype="full"):
        if not callable(g):
            (self.Na, self.Ni) = numpy.shape(g)
            self.gtype = "full"
            self.grad = g
            return

        self.Na = numpy.size(g(x0))
        self.Ni = numpy.size(x0)
        self.gtype = gtype

        _grad = g(numpy.ones(x0.shape)).flatten()
        
        if gtype == "full":
            self.grad = pyobs.array((self.Na, self.Ni), _grad.dtype, zeros=True)
            dx = pyobs.double_array(self.Ni, zeros=True)
            for i in range(self.Ni):
                dx[i] = 1.0
                self.grad[:, i] = numpy.reshape(g(numpy.reshape(dx, x0.shape)), self.Na)
                dx[i] = 0.0
        elif gtype == "diag":
            self.grad = _grad.copy()
            pyobs.assertion(self.Na == self.Ni, "diagonal gradient error")
        else:  # pragma: no cover
            raise pyobs.PyobsError("gradient error")
            
        del _grad
        
    def get_mask(self, mask):
        idx = pyobs.int_array(mask)
        if self.gtype == "full":
            h = numpy.sum(self.grad[:, idx] != 0.0, axis=1)
            if numpy.sum(h) > 0:
                return list(numpy.arange(self.Na)[h > 0])
            else:
                return None
        elif self.gtype == "diag":
            return mask

    # u = grad @ v
    def apply(self, u, umask, uidx, v, vmask):
        if self.gtype == "full":
            gvec = pyobs.slice_ndarray(self.grad, umask, vmask)
            if uidx is None:
                u += gvec @ v
            else:
                u[:, uidx] += gvec @ v
        elif self.gtype == "diag":
            grad = dia_matrix(pyobs.slice_ndarray(numpy.diag(self.grad), umask, vmask))
            if uidx is None:
                u += grad.dot(v)
            else:
                u[:, uidx] += grad.dot(v)
