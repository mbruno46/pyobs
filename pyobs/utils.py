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
import inspect
import re
import time
import functools

__all__ = [
    "PyobsError",
    "assertion",
    "check_type",
    "is_verbose",
    "set_verbose",
    "log_timer",
    "valerr",
    "tex_table",
    "slice_ndarray",
    "import_string",
    "double_array",
    "int_array",
]

verbose = ["save", "load", "mfit"]


class PyobsError(Exception):
    pass


def assertion(condition, message):
    if not condition:
        stk = inspect.stack()[1]
        mod = inspect.getmodule(stk[0])
        raise PyobsError(f"{mod}.{stk[3]}\n{message: >16}")


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


def valerr(value, error, significant_digits=2):
    """
    Converts arrays in formatted strings in the form "value(error)"".

    Parameters:
       value (array): a float, 1-D or 2-D array
       error (array): a float, 1-D or 2-D array

    Returns:
       string: the formatted string

    Examples:
       >>> [v, e] = obs.error()
       >>> print(pyobs.valerr(v,e))
       1234(4)
    """

    d = significant_digits - 1

    def core(v, e):
        if e == 0:
            return f"{v:g}"
        exp = int(numpy.floor(numpy.log10(e)) - d)
        if exp < 0:
            out = f"%.{-exp}f" % v
            if (exp + d) >= 0:
                out += f"(%.{-exp}f)" % (e * 10 ** -(exp + d))
            else:
                out += f"({e * 10 ** -(exp):.0f})"
        else:
            out = f"{v:.0f}({e:.0f})"
        return out

    if numpy.ndim(value) == 0:
        return core(value, error)
    elif numpy.ndim(value) == 1:
        return " ".join([core(_v, _e) for _v, _e in zip(value, error)])
    elif numpy.ndim(value) == 2:
        return "\n".join(
            [valerr(value[i, :], error[i, :]) for i in range(numpy.shape(value)[0])]
        )
    else:  # pragma: no cover
        raise PyobsError("valerr supports up to 2D arrays")


def tex_table(mat, fmt=None):
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
       >>> pyobs.tex_table(mat, fmt=['d',0,0])
    """

    assertion(numpy.ndim(mat) == 2, "textable supports only 2D arrays")

    (n, m) = numpy.shape(mat)
    fmt = [".2f"] * m if fmt is None else fmt

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
        outstr += [rf'{" & ".join(h)} \\ ']

    return outstr


def check_type(obj, s, *t):
    c = 0
    for tt in t:
        if not type(obj) is tt:
            c += 1
    if c == len(t):
        raise TypeError(f"Unexpected type for {s} [{t}]")


def slice_to_range(sl, n):
    return list(range(n)[sl])


def slice_ndarray(t, *args):
    """
    Slices a N-D array.

    Parameters:
       t (array): N-D array
       *args (list): a series of lists, arrays, slices or integers with the indices
                     used for the extraction. `[]` is interpreted as taking
                     all elements along that given axis, like slice(None).

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
    assertion(len(args) == len(s), "Dimensions of tensor do not match indices")

    aa = []
    for ia in range(len(args)):
        a = args[ia]
        if a is None:
            aa.append(range(s[ia]))
        elif isinstance(a, slice):
            aa.append(slice_to_range(a, s[ia]))
        elif isinstance(a, numpy.ndarray):
            aa.append(a)
        elif isinstance(a, list):
            if not a:
                aa.append(range(s[ia]))
            else:
                aa.append(a)
        elif isinstance(a, (int, numpy.int32, numpy.int64)):
            aa.append([a])
        else:  # pragma: no cover
            raise PyobsError("slicing not understood")

    return t[numpy.ix_(*aa)]


def import_string(data):
    """
    Imports strings in the format value(error) into arrays

    Examples:
       >>> arr = pyobs.import_string(['1234(4)','0.02345(456)'])
       >>> print(arr)
       array([[1.234e+03, 4.000e+00],
              [2.345e-02, 4.560e-03]])
    """

    def core(string):
        m = re.search(r"(^\d+).?(\d*)\((\d+)\)", string)
        d0 = m.group(1)
        d1 = m.group(2)
        e = m.group(3)
        return [numpy.float64(f"{d0}.{d1}"), numpy.float64(e) * 10 ** -len(d1)]

    if isinstance(data, list):
        out = [core(s) for s in data]
        return numpy.array(out)
    return core(data)


def pyobs_array(arg, type, zeros):
    if zeros:
        return numpy.zeros(arg, dtype=type)
    return numpy.atleast_1d(arg).astype(type)


def double_array(arg, zeros=False):
    return pyobs_array(arg, numpy.float64, zeros)


def int_array(arg, zeros=False):
    return pyobs_array(arg, numpy.int32, zeros)
