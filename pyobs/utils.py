#################################################################################
#
# utils.py: generic utility routines
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
import sys

__all__ = [
    "PyobsError",
    "check_type",
    "check_not_type",
    "is_verbose",
    "set_verbose",
    "valerr",
    "textable",
    "slice_ndarray",
]

verbose = ["save", "load"]


class PyobsError(Exception):
    pass


def is_verbose(func):
    if func in verbose:
        return True
    return False


def set_verbose(func):
    if func not in verbose:
        verbose.append(func)


def valerr(v, e):
    def core(v, e):
        if e > 0:
            dec = -int(numpy.floor(numpy.log10(e)))
        else:
            dec = 1
        if dec > 0:
            outstr = f"{v:.{dec+1}f}({e*10**(dec+1):02.0f})"
        elif dec == 0:
            outstr = f"{v:.{dec+1}f}({e:02.{dec+1}f})"
        else:
            outstr = f"{v:.0f}({e:.0f})"
        return outstr

    if numpy.ndim(v) == 0:
        return core(v, e)
    elif numpy.ndim(v) == 1:
        return " ".join([core(v[i], e[i]) for i in range(len(v))])
    elif numpy.ndim(v) == 2:
        return "\n".join([valerr(v[i, :], e[i, :]) for i in range(numpy.shape(v)[0])])
    else:  # pragma: no cover
        raise PyobsError("valerr supports up to 2D arrays")


def textable(mat, fmt=None):
    """
    Formats a matrix to a tex table.

    Parameters:
       mat (array): 2D array
       fmt (list, optional): a list of formats for each columns; if 0 is passed
           on two consecutive columns the program assumes that they correspond
           to value and error and prints them together in a single column; otherwise
           accepted values are 'd' for integers, '.2f' for floats with 2 digit
           precision, etc ...

    Returns:
       list of str: a list where each element is a line of the tex table

    Examples:
       >>> tsl = [0,1,2,3,4] # time-slice
       >>> [c, dc] = correlator.error()
       >>> mat = numpy.c_[tsl, c, dc] # pack data
       >>> pyobs.textable(mat, fmt=['d',0,0])
    """

    if numpy.ndim(mat) != 2:  # pragma: no cover
        raise PyobsError("textable supports only 2D arrays")

    (n, m) = numpy.shape(mat)
    if fmt is None:
        fmt = [".2f"] * m

    outstr = []
    for a in range(n):
        h = []
        i = 0
        while i < m:
            if fmt[i] == 0:
                h += [valerr(mat[a, i], mat[a, i + 1])]
                i += 2
            else:
                h += [f"%{fmt[i]}" % mat[a, i]]
                i += 1
        outstr += [fr'{" & ".join(h)} \\ ']

    return outstr


def check_type(obj, s, *t):
    c = 0
    for tt in t:
        if not type(obj) is tt:
            c += 1
    if c == len(t):
        raise TypeError(f"Unexpected type for {s} [{t}]")


def check_not_type(obj, s, t):
    if type(obj) is t:
        raise TypeError(f"Unexpected type for {s}")


def slice_ndarray(t, *args):
    """
    Slices a N-D array.

    Parameters:
       t (array): N-D array
       *args (list): a series of lists or arrays with the indices
                     used for the extraction. `[]` is interpreted
                     as taking all elements along that given axis.

    Returns:
       array: the sliced N-D array.
       Note that the number of dimensions does not change even
       when only a single coordinate is selected along a given axis.

    Examples:
       >>> mat = numpy.arange(12).reshape(2,2,3)
       >>> pyobs.slice_tensor(mat,[],[0],[0,3])
       array([[[0, 2]],
              [[6, 8]]])
    """
    s = numpy.shape(t)
    if len(args) != len(s):  # pragma: no cover
        raise TypeError("Dimensions of tensor do not match indices")

    aa = []
    for a in args:
        ia = args.index(a)
        if (a is None) or (not a):
            aa.append(range(s[ia]))
        elif isinstance(a, (numpy.ndarray, list)):
            aa.append(a)

    return t[numpy.ix_(*aa)]
