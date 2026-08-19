#################################################################################
#
# utils.py: generic utility routines
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

import numpy as np
import functools
import time
import numbers

__all__ = [
    "is_verbose",
    "set_verbose",
    "set_matrixfreegrad",
    "is_matrixfreegrad",
    "log_timer",
    "message",
    "complex",
    "double",
    "int",
]

complex = np.complex128
double = np.float64
int = np.int32

verbose = ["save", "load", "mfit"]
matrixfree = False


def is_verbose(func):
    if func in verbose:
        return True
    return False


def set_verbose(func, yesno=True):
    if yesno:
        if func not in verbose:
            verbose.append(func)
    else:
        if func in verbose:
            verbose.remove(func)


def set_matrixfreegrad(mode):
    """
    Modify the way the `gradient` class is instantiated and the way gradients are applied
    i.e. the way the error propagation is handled:

    if mode = False always allocate gradient matrix (default)
    if mode = True  never allocate gradient matrix
    if mode = N     allocate gradient matrix if Na x Ni < N else no allocation

    the mode then affects the way the fluctuations are propagated in the following way:
    - if the `Na x Ni` gradient matrix exists then its applied to the `Ni x Ncnfg` fluctuation matrix
    - else the gradient is applied to each column of length `Ni` separately

    Parameters:
        mode (boolean, int): the new mode

    Notes:
        The mode is required at instatiation time and does not affect already constructed `gradient` objects.
        The matrix mode is typically faster for small observables, but has a larger memory footprint.
        The matrix free mode is typically faster for large observables and has a constant memory footprint.
    """
    global matrixfree
    if isinstance(mode, numbers.Integral):
        matrixfree = mode
    else:
        raise Exception(f"mode must be True, False or an integer (not {mode})")


def is_matrixfreegrad():
    return matrixfree


def log_timer(tag):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            t1 = time.time()
            if is_verbose(tag):
                print(f"{tag} executed in {t1 - t0:g} secs")
            return result

        return wrapper

    return decorator


def message(*args, **kwargs):
    head = "[pyobs] : "
    print(head + " ".join(map(str, args)), **kwargs)
