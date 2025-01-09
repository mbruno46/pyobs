#################################################################################
#
# data.py: definition and properties of the class with the fluctuations
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

import numpy as np
import pyobs

__all__ = ["delta"]


def is_int(x):
    return isinstance(x, (int, np.int32, np.int64))


def expand_data(data, idx, shape):
    v = np.prod(shape)
    tmp = None
    if type(idx) is range:
        tmp = pyobs.double_array(data)
    else:
        tmp = pyobs.double_array(v, zeros=True)
        for j in range(len(idx)):
            tmp[idx[j] - idx[0]] = data[j]
    return tmp


def create_fft_data(data, idx, shape, fft_ax):
    tmp = expand_data(data, idx, shape)
    if len(fft_ax) > 1:
        tmp = np.reshape(tmp, shape)
    # in-place, even if it adds one element at the end
    return np.fft.rfftn(tmp, s=shape, axes=fft_ax)


# NOTE: if lat is integer then Monte carlo assumed, ie open BC at the boundary
# of the markov chain. if lat is a list then periodic BC assumed in all dirs,
# even if lat is 1D
# improved version does not suffer from roundoff errors from overall normalization
# of data, eg data is of order 1e-15
def conv_ND(data, idx, lat, xmax, a=0, b=None):
    if is_int(lat):
        shape = (2 * lat,)
        lat = [lat]
        ismf = False
    else:
        shape = tuple(lat)
        ismf = True

    D = len(lat)
    fft_ax = tuple(range(D))
    v = np.prod(lat)

    aux = []
    rescale = 1
    for index in [a, b]:
        if index is not None:
            f = np.max(data[index, :])
            rescale *= f
            aux += [create_fft_data(data[index, :] / f, idx, shape, fft_ax)]

    if len(aux) == 1:
        aux[0] *= aux[0].conj()
        aux += [np.fft.irfftn(aux[0], s=shape, axes=fft_ax)]
        rescale *= rescale
    else:
        aux[0] *= aux[1].conj()
        aux[1] = np.fft.irfftn(aux[0].real, s=shape, axes=fft_ax)

    # important to rescale here for mf
    aux[1] *= rescale
    if ismf:
        aux[1] = np.reshape(aux[1], (v,))
        return pyobs.core.mftools.intrsq(aux[1], lat, xmax)

    return aux[1][0:xmax]


# TODO: remove blocks that are zeros
def block_data(data, idx, lat, bs):
    if is_int(lat):
        lat = pyobs.int_array([lat])
        bs = pyobs.int_array([bs])

    shape = tuple(lat)
    norm = np.max(data)
    dat = expand_data(data / norm, idx, shape)
    return pyobs.core.mftools.blockdata(dat, lat, bs) * norm


class delta:
    def __init__(self, mask, idx, data=None, mean=None, lat=None):
        # idx is expected to be a list or range
        self.size = len(mask)
        self.mask = [m for m in mask]
        self.it = 0
        if lat is None:
            self.lat = None
        else:
            self.lat = pyobs.int_array(lat)

        if type(idx) is list:
            dc = np.unique(np.diff(idx))
            pyobs.assertion(np.any(dc > 0), "Unsorted idx")
            if len(dc) == 1:
                self.idx = range(idx[0], idx[-1] + dc[0], dc[0])
            else:
                self.idx = list(idx)
        elif type(idx) is range:
            self.idx = idx
        else:  # pragma: no cover
            raise pyobs.PyobsError("Unexpected idx")
        self.n = len(self.idx)

        self.delta = pyobs.double_array((self.size, self.n), zeros=True)

        if mean is not None:
            self.delta = np.reshape(data, (self.n, self.size)).T - np.stack(
                [mean] * self.n, axis=1
            )

    def copy(self, second=None):
        if second is None:
            res = delta(self.mask, self.idx, lat=self.lat)
            res.delta = pyobs.array(self.delta, self.delta.dtype)
        else:
            res = delta(self.mask, self.idx, lat=self.lat)
            res.delta = pyobs.double_array(second(self.delta))
        return res

    def __getitem__(self, args):
        sliced_delta = pyobs.slice_ndarray(self.delta, args, [])
        res = delta(range(sliced_delta.shape[0]), self.idx, lat=self.lat)
        res.delta = res.delta.astype(sliced_delta.dtype)
        res.delta[:, :] = sliced_delta
        return res

    def __setitem__(self, submask, rd):
        pyobs.assertion(
            len(submask) == len(rd.mask), "Dimensions do not match in assignment"
        )
        a = np.nonzero(np.in1d(self.mask, submask))[0]
        self.delta[a, :] = rd.delta

    def ncnfg(self):
        if type(self.idx) is range:
            return self.n
        else:
            return (
                int(self.idx[-1] - self.idx[0]) + 1
            )  # first and last config included!

    # def get_mask(self, a):
    #     if a in self.mask:
    #         return self.mask.index(a)
    #     else:
    #         return -1

    def start_idx(self):
        self.it = 0

    def get_idx(self, index):
        if type(self.idx) is range:
            return self.idx.index(index)
        else:
            while self.idx[self.it] != index:
                self.it += 1
            return self.it

    def axpy(self, grad, d):
        N = d.delta.shape[1]

        # prepare index list
        self.start_idx()
        d.start_idx()
        jlist = []
        for i in range(N):
            k = d.idx[i]
            jlist.append(self.get_idx(k))
        jlist = pyobs.int_array(jlist)

        # takes into accounts holes present in d.delta but absent in self.delta
        rescale_delta = self.n / d.n

        if (np.iscomplexobj(grad.grad)) or (np.iscomplexobj(d.delta)):
            self.delta = self.delta.astype(pyobs.complex)
        grad.apply(self.delta, self.mask, jlist, d.delta * rescale_delta, d.mask)

    def gamma(self, xmax, a, b=None):
        ones = np.reshape(np.ones(self.n), (1, self.n))
        isMC = self.lat is None

        if isMC:
            m = conv_ND(ones, self.idx, self.ncnfg(), xmax)
        else:
            rrmax = xmax
            v = self.vol()
            if v == len(self.idx):
                m = [v] * rrmax
            else:
                m = conv_ND(ones, self.idx, self.lat, rrmax)
                Sr = pyobs.core.mftools.intrsq(np.ones(v), self.lat, rrmax)
                Sr = Sr + 1 * (Sr == 0.0)
                m /= Sr

        g = conv_ND(
            self.delta, self.idx, self.ncnfg() if isMC else self.lat, xmax, a, b
        )
        return [m, g]

    def bias4(self, hess):
        oid = np.array(self.mask)
        idx = np.ix_(oid, oid)
        # no rescaling factor; prone to roundoff errors
        d2 = np.einsum("abc,bj,cj->aj", hess[:, idx[0], idx[1]], self.delta, self.delta)
        return np.sum(d2, axis=1)

    def blocked(self, bs):
        isMC = self.lat is None

        if isMC:
            if is_int(bs):
                v = self.ncnfg()  # (self.ncnfg()+bs-1) - ((self.ncnfg()+bs-1)%bs)
                v //= bs
                lat = None
            else:  # pragma: no cover
                raise pyobs.PyobsError("Unexpected block size")
        else:
            pyobs.assertion(
                np.sum(self.lat % np.array(bs)) == 0,
                "Block size does not divide lattice",
            )
            pyobs.assertion(len(bs) == len(self.lat), "Block size does match lattice")
            lat = self.lat / np.array(bs)
            v = int(np.prod(lat))
            bs = pyobs.int_array(bs)

        res = delta(self.mask, range(v), lat=lat)
        for a in range(self.size):
            res.delta[a, :] = block_data(
                self.delta[a, :], self.idx, self.ncnfg() if isMC else self.lat, bs
            )
        return res

    # replica ensemble utility functions
    def wmax(self):
        return self.ncnfg() // 2

    def rrmax(self):
        return int(np.sum((self.lat / 2) ** 2) + 1)

    def vol(self):
        return np.prod(self.lat)
