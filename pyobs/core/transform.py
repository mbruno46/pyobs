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
    

def transform(obs, f):
    new_mean = f(obs.mean)
    res = pyobs.observable(description = obs.description)
    res.set_mean(new_mean)
    
    subset_mask = list(f(numpy.reshape(numpy.arange(obs.size),obs.shape)).flatten())
    
    for key in obs.delta:
        d = obs.delta[key]
        d_mask = []
        _mask = []
        for ia in range(len(subset_mask)):
            a = subset_mask[ia]
            if a in d.mask:
                d_mask += [d.get_mask(a)]
                _mask += [ia]
        if d_mask:
            res.delta[key] = delta(_mask, d.idx, lat=d.lat)
            res.delta[key].delta = d.delta[numpy.array(d_mask),:]

    for key in obs.cdata:
        cd = obs.cdata[key]
        cd_mask = []
        _mask = []
        for ia in range(len(subset_mask)):
            a = subset_mask[ia]
            if a in cd.mask:
                cd_mask += [cd.mask.index(a)]
                _mask += [ia]
        if cd_mask:
            res.cdata[key] = cdata(cd.cov, _mask)
            res.cdata[key].grad = cd.grad[numpy.array(cd_mask),:]
        
    res.ename_from_delta()
    pyobs.memory.update(res)
    return res
    
    
    
    