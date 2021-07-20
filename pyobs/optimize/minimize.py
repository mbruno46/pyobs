#################################################################################
#
# minimize.py: implementation of the Levenberg–Marquardt algorithm
# Copyright (C) 2020 Mattia Bruno
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


class outputformat:
    def __init__(self, x, f0, i, msg):
        self.x = x
        self.fun = f0
        self.nit = i
        self.message = msg


def lm(
    fun,
    x0,
    jac,
    hess,
    opts={
        "TolX": 1e-8,
        "Tol": 1e-6,
        "LAMBDA_START": 1e-4,
        "maxiter": 1024,
        "LAMBDA_MAX": 1e8,
    },
):
    x = numpy.array(x0)
    f0 = fun(x)
    g = jac(x)
    h = hess(x)
    # [g,h]=grad(x)

    lam = opts["LAMBDA_START"]
    step = 10.0
    for i in range(1, opts["maxiter"] + 1):
        hd = numpy.diag(numpy.diag(h))
        alpha = h + lam * hd
        [u, s, vh] = numpy.linalg.svd(alpha)
        alpha_inv = (vh.conj().T * (1.0 / s)) @ u.conj().T

        beta = -g
        dx = beta @ alpha_inv
        x += dx

        if numpy.any(numpy.fabs(dx) < opts["TolX"] * numpy.fabs(x)):
            msg = f'Levenberg-Marquardt: reached  {opts["TolX"]:.1e} per-cent tolerance on x0'
            break

        f1 = fun(x)
        # forces the algorithm to increase lambda and rejects current parameters
        if numpy.isnan(f1):  # pragma: no cover
            f1 = f0 + 1.0

        if abs(f1 / f0 - 1.0) > opts["Tol"]:
            if f1 > f0:
                if lam < opts["LAMBDA_MAX"]:
                    lam *= step
                    x -= dx
                else:  # pragma: no cover
                    print(
                        "Warning: Levenberg-Marquardt: lambda parameter too large: stuck in valley"
                    )
            else:
                lam = lam / step
                f0 = f1
                g = jac(x)
                h = hess(x)
        else:
            msg = f'Levenberg-Marquardt: converged {opts["Tol"]:.1e} per-cent tolerance on fun'
            break

    if i == opts["maxiter"]:  # pragma: no cover
        msg = f'Levenberg-Marquardt: did not converge in {opts["maxiter"]} iterations'
    return outputformat(x, f0, i, msg)
