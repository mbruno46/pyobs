#################################################################################
#
# default.py: default file format based on the bison file format
# Copyright (C) 2021 Mattia Bruno
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
import bison
import warnings


class observable_decoder:
    def __init__(self):
        self.type = "pyobs.core.ndobs.observable"

    def decode(self, obj):
        out = pyobs.observable(description=obj["description"])
        out.set_mean(obj["mean"])
        assert out.shape == obj["shape"]
        assert out.size == obj["size"]
        out.ename = obj["ename"]
        out.delta = obj["delta"]
        out.cdata = obj["cdata"]
        pyobs.memory.update(out)
        return out


class delta_decoder:
    def __init__(self):
        self.type = "pyobs.core.data.delta"

    def decode(self, obj):
        out = pyobs.core.data.delta(obj["mask"], obj["idx"], lat=obj["lat"])
        out.delta = out.delta.astype(obj["delta"].dtype)
        out.delta[:, :] = obj["delta"][:, :]
        return out


class cdata_decoder:
    def __init__(self):
        self.type = "pyobs.core.cdata.cdata"

    def decode(self, obj):
        # this guarantees backwards compatibility
        if "grad" in obj:
            out = pyobs.core.cdata.cdata(obj["cov"], obj["mask"])
            out.grad[:, :] = obj["grad"][:, :]
            return out
        else:  # pragma: no cover
            warnings.warn("Loading from older file format; cdata may be wrongly used!")
            out = pyobs.core.cdata.cdata(obj["cov"])
            return out


def save(fname, *args):
    bison.save(fname, *args)


def load(fname):
    return bison.load(fname, decoder=[observable_decoder, delta_decoder, cdata_decoder])
