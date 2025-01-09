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

# def indices_isin(a,b):
#     idx_a = []
#     idx_b = []
#     for ia, _a in enumerate(a):
#         if _a in b:
#             idx_a += [ia]
#             idx_b += [b.index(_a)]
#     return idx_a, idx_b


def indices_isin(a, b):
    # idx_a = indices of elements of a that are present in b preserving the order a
    idx_a = numpy.arange(len(a))[numpy.in1d(a, b)]
    if idx_a.size == 0:
        return [], []
    # idx_b = indices of elements of b that are present in a preserving the order a
    idx_b = numpy.arange(len(b))[numpy.in1d(b, a)]
    mask = numpy.array(a)[idx_a, None] == numpy.array(b)[idx_b]
    idx_b_2 = numpy.stack([idx_b] * len(idx_a))[mask]
    return list(idx_a), list(idx_b_2)


def transform(obs, f):
    new_mean = f(obs.mean)
    res = pyobs.observable(description=obs.description)
    res.set_mean(new_mean)

    subset_mask = f(numpy.reshape(numpy.arange(obs.size), obs.shape)).flatten()

    for key in obs.delta:
        d = obs.delta[key]
        idx_subset_mask, idx_mask = indices_isin(subset_mask, d.mask)
        if pyobs.is_verbose("transform"):
            print("\ntransform debugging")
            print("subset_mask", subset_mask, f"{key}->mask", d.mask)
            print("idx_subset_mask", idx_subset_mask, "idx_mask", idx_mask)
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
