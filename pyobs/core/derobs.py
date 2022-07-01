#################################################################################
#
# derobs.py: implementation of the core function for derived observables
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


def merge_idx(idx1, idx2):
    if (type(idx1) is range) and (type(idx2) is range):
        id0 = min([idx1.start, idx2.start])
        id1 = max([idx1.stop, idx2.stop])
        id2 = min([idx1.step, idx2.step])
        return range(id0, id1, id2)
    else:
        u = set(idx1)
        return list(sorted(u.union(idx2)))


def get_keys(inps, name):
    allkeys = []
    for i in inps:
        for dn in i.__dict__[name]:
            if dn not in allkeys:
                allkeys.append(dn)
    return allkeys


@pyobs.log_timer("derobs")
def derobs(inps, mean, grads, description=None):
    pyobs.check_type(inps, "inps", list)
    pyobs.check_type(
        mean, "mean", numpy.ndarray, int, float, numpy.float32, numpy.float64
    )
    pyobs.check_type(grads, "grads", list)
    if len(inps) != len(grads):
        raise pyobs.PyobsError("Incompatible inps and grads")
    if description is None:
        description = ", ".join(set([i.description for i in inps]))
    res = pyobs.observable(description=description)
    res.set_mean(mean)

    for key in get_keys(inps, "delta"):
        new_idx = []
        new_mask = []
        lat = None
        for i in range(len(inps)):
            if key in inps[i].delta:
                d = inps[i].delta[key]
                h = grads[i].get_mask(d.mask)
                if h is not None:
                    new_mask += h
                    if not new_idx:
                        new_idx = d.idx
                    else:
                        new_idx = merge_idx(new_idx, d.idx)
                    if lat is None:
                        lat = d.lat
                    else:
                        pyobs.assertion(
                            numpy.any(lat == d.lat),
                            "Unexpected lattice size for master fields with same tag",
                        )
        if len(new_mask) > 0:
            res.delta[key] = delta(list(set(new_mask)), new_idx, lat=lat)
            for i in range(len(inps)):
                if key in inps[i].delta:
                    res.delta[key].axpy(grads[i], inps[i].delta[key])

    res.ename_from_delta()
    
    for key in get_keys(inps, "cdata"):
        new_mask = []
        cov = None
        for i in range(len(inps)):
            if key in inps[i].cdata:
                cd = inps[i].cdata[key]
                h = grads[i].get_mask(cd.mask)
                if h is not None:
                    new_mask += h
                    if cov is None:
                        cov = cd.cov
                    else:
                        pyobs.assertion(
                            numpy.all(cov == cd.cov),
                            "Unexpected cov. matrix for cdata with same tag",
                        )
        if len(new_mask) > 0:
            res.cdata[key] = cdata(cov, list(set(new_mask)))
            res.cdata[key].grad[:, :] = 0.0
            for i in range(len(inps)):
                if key in inps[i].cdata:
                    res.cdata[key].axpy(grads[i], inps[i].cdata[key])

    pyobs.memory.update(res)
    return res


def num_grad(x, f, eps=2e-4):
    if isinstance(x, pyobs.observable):
        x0 = x.mean
    else:
        x0 = numpy.array(x)

    s = x0.shape
    n = numpy.size(x0)
    dx = numpy.zeros((n,))

    f0 = f(x0)
    m = numpy.size(f0)
    df = numpy.zeros((m, n))

    for i in range(n):
        dx[i] = 1.0
        dx = numpy.reshape(dx, s)

        fp = f(x0 + x0 * eps * dx)
        fm = f(x0 - x0 * eps * dx)
        fpp = f(x0 + x0 * 2.0 * eps * dx)
        fmm = f(x0 - x0 * 2.0 * eps * dx)

        tmp = 2.0 / 3.0 * (fp - fm) - 1 / 12 * (fpp - fmm)
        df[:, i] = tmp.flatten() / (x0.flatten()[i] * eps)

        dx = numpy.reshape(dx, (n,))
        dx[i] = 0.0

    return df


def num_hess(x0, f, eps=2e-4):
    f0 = f(x0)
    m = numpy.size(f0)
    n = numpy.size(x0)

    ddf = numpy.zeros((m, n, n))
    for i in range(m):
        for j in range(n):
            ddf[i, j, :] = num_grad(x0, lambda x: num_grad(x, f)[i, j])[0]

    return ddf


def error_bias4(x, f):
    pyobs.check_type(x, "x", pyobs.observable)
    [x0, dx0] = x.error()
    bias4 = numpy.zeros((x.size,))
    hess = num_hess(x0, f)

    for key in x.delta:
        oid = numpy.array(x.delta[key].mask)
        idx = numpy.ix_(oid, oid)
        d2 = numpy.einsum(
            "abc,bj,cj->aj",
            hess[:, idx[0], idx[1]],
            x.delta[key].delta,
            x.delta[key].delta,
        )
        dd2 = numpy.sum(d2, axis=1)
        bias4 += dd2**2 / x.delta[key].n ** 4

    return numpy.sqrt(bias4)
