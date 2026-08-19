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


def _matrixfree(Ni, Na): 
    mfreegrad = pyobs.is_matrixfreegrad()
    if isinstance(mfreegrad, bool):
        return mfreegrad
    else: 
        return Ni*Na >= mfreegrad 

# grad is Na x Ni matrix
class gradient:
    def __init__(self, g, x0=None, gtype="full"):
        if not callable(g):
            (self.Na, self.Ni) = numpy.shape(g)
            self.gtype = "full"
            self.grad = g
            self.matrixfree = False 
            return

        self.Ni = numpy.size(x0)
        probe = g(numpy.ones(x0.shape))
        self.Na = numpy.size(probe)
        self.gtype = gtype

        if gtype == "full" and _matrixfree(self.Ni, self.Na): 
            self.matrixfree = True 
            self.g = g 
            self.dtype = probe.dtype
            self.x0shape = x0.shape 
            return 
        else: 
            _grad = probe.flatten() 
            self.matrixfree = False 

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

    def is_complex(self): 
        if self.matrixfree: 
            return numpy.iscomplexobj(numpy.zeros(1, dtype=self.dtype))
        else: 
            return numpy.iscomplexobj(self.grad)

    def get_mask(self, mask):
        if self.gtype == "full":
            if self.matrixfree: 
                h = numpy.zeros(self.Na)
                ej = numpy.zeros(self.Ni)
                for j in mask:
                    ej[j] = 1.0 
                    h += (self.g(ej.reshape(self.x0shape)).flatten() != 0.0)
                    ej[j] = 0.0
            else: 
                idx = pyobs.int_array(mask)
                h = numpy.sum(self.grad[:, idx] != 0.0, axis=1)
            if numpy.sum(h) > 0:
                return list(numpy.arange(self.Na)[h > 0])
            else:
                return None
        elif self.gtype == "diag":
            return mask

    # u = grad @ v
    def apply(self, u, umask, uidx, v, vmask):
        if self.matrixfree: 
            xfull = numpy.zeros(self.Ni, dtype=numpy.result_type(self.dtype, v.dtype))
            for cnfgidx in range(v.shape[1]): 
                xfull[vmask] = v[:, cnfgidx]
                y = self.g(xfull.reshape(self.x0shape)).flatten()
                u[:, cnfgidx if uidx is None else uidx[cnfgidx]] += y[umask]
        else: 
            if self.gtype == "full":
                gvec = pyobs.slice_ndarray(self.grad, umask, vmask)
                if uidx is None:
                    u += gvec @ v
                else:
                    u[:, uidx] += gvec @ v
            elif self.gtype == "diag":
                um = pyobs.int_array(umask)
                vm = pyobs.int_array(vmask)
                inter, uind, vind = numpy.intersect1d(um, vm, return_indices=True) # only forward fluctuations on the intersection since they contribute 0 otherwise (i.e. v has no fluctuations or u has no fluctuations)
                contrib = self.grad[inter][:, None] * v[vind, :]
                if uidx is None:
                    u[uind, :] += contrib
                else:
                    u[numpy.ix_(uind, uidx)] += contrib