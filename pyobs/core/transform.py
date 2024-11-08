#################################################################################
#
# slice.py: slicing and indexing of observables
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

import numpy
import pyobs

from .data import delta
from .cdata import cdata


# def indices_isin(a, b):
#     ia, ib = numpy.where(numpy.array(a)[:, None] == numpy.array(b)[None, :])
#     return list(ia), list(ib)

# for operations like transpose the order of a and b matters
# so the intersection should keep the correct ordering
# in1d does not preserve the ordering
# def indices_isin(a, b):
#     inner = lambda x, y: numpy.nonzero(numpy.in1d(x, y))[0]
#     return list(inner(a, b)), list(inner(b, a))


# returns indices of elements of a that are present in b preserving the order a
def indices_isin(a,b):
    idx = numpy.in1d(a,b)
    return list(numpy.arange(len(a))[idx]), list(numpy.array(a)[idx])

def transform(obs, f):
    new_mean = f(obs.mean)
    res = pyobs.observable(description=obs.description)
    res.set_mean(new_mean)

    subset_mask = f(numpy.reshape(numpy.arange(obs.size), obs.shape)).flatten()
    
    for key in obs.delta:
        d = obs.delta[key]
        idx_subset_mask, idx_mask = indices_isin(subset_mask, d.mask)
        if len(idx_subset_mask) > 0:
            res.delta[key] = d[idx_mask]
            res.delta[key].mask = idx_subset_mask

    for key in obs.cdata:
        cd = obs.cdata[key]
        idx_subset_mask, idx_mask = indices_isin(subset_mask, cd.mask)
        if len(idx_subset_mask) > 0:
            res.cdata[key] = cdata(cd.cov, list(idx_subset_mask))
            res.cdata[key].grad[:, :] = cd.grad[list(idx_mask), :]

    res.ename_from_delta()
    pyobs.memory.update(res)
    return res
