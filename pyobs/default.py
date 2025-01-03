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

import numpy
import functools
import time

__all__ = [
    "is_verbose",
    "set_verbose",
    "log_timer",
    "complex",
    "double",
    "int"
]

complex = numpy.complex128
double = numpy.float64
int = numpy.int32

verbose = ["save", "load", "mfit"]

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

def log_timer(tag):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            t1 = time.time()
            if is_verbose(tag):
                print(f"{tag} executed in {t1-t0:g} secs")
            return result

        return wrapper

    return decorator
