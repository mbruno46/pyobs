#################################################################################
#
# special.py: definitions of special functions for observables
# Copyright (C) 2020-2023 Mattia Bruno
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

import scipy
import pyobs

__all__ = [
    "besselk",
]


def besselk(v, x):
    """
    Modified Bessel function of the second kind of real order `v`, element-wise.

    Parameters:
       v (float): order of the Bessel function
       x (obs): real observable where to evaluate the Bessel function

    Returns:
       obs : the modified bessel function computed for the input observable
    """
    new_mean = scipy.special.kv(v, x.mean)
    aux = scipy.special.kv(v - 1, x.mean) + scipy.special.kv(v + 1, x.mean)
    g = pyobs.gradient(lambda x: -0.5 * aux * x, x.mean, gtype="diag")
    return pyobs.derobs(
        [x], new_mean, [g], description=f"BesselK[{v}] of {x.description}"
    )
