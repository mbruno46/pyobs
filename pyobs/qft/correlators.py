#################################################################################
#
# correlators.py: construct correlators from single operators
# Copyright (C) 2025 Mattia Bruno
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


def two_point_correlator(op1, op2, axis=0, sources=False, separations=None):
    pyobs.assertion(
        len(op1.ename) == 1, "Operators should be measured on single ensemble"
    )
    pyobs.assertion(
        sorted(op1.ename) == sorted(op2.ename),
        "Ensemble tags from operators do not match",
    )
    pyobs.assertion(op1.shape == op2.shape, "Shapes of operators do not match")

    k1 = pyobs.get_keys(op1, op1.ename[0])
    k2 = pyobs.get_keys(op2, op2.ename[0])

    corr = pyobs.observable()

    for key in set(k1).intersection(k2):
        pyobs.assertion(op1.delta[key].idx == op2.delta[key].idx, "")

        if sources:
            seps = range(op1.shape[axis]) if separations is None else separations
            d = []
            for op in [op1, op2]:
                d += [op1.delta[key].delta.T.reshape((op.delta[key].n,) + op.shape)]

            aux = np.zeros(
                (
                    d[0].shape[0],
                    len(seps),
                )
                + op1.shape
            )
            for i, dt in enumerate(seps):
                aux[:, i] = d[0] * np.roll(d[1], dt, axis=axis + 1)

            corr.create(
                key.split(":")[0],
                np.array(aux).flatten(),
                shape=(len(seps),) + op1.shape,
            )
        else:
            aux = []
            for op in [op1, op2]:
                d = op.delta[key].delta.T.reshape((op.delta[key].n,) + op.shape)
                aux += [np.fft.fftn(d, axes=[axis + 1])]

            tmp = (
                np.fft.ifftn(aux[0] * np.conj(aux[1]), axes=[axis + 1]).real
                / op1.shape[axis]
            )
            corr.create(
                key.split(":")[0],
                tmp.flatten(),
                rname=key.split(":")[1],
                shape=op1.shape,
            )

    return corr
