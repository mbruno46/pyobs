#################################################################################
#
# __init__.py: basic IO interface
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

from . import json, default
import pyobs
import os
import datetime

__all__ = ["save", "load"]


def save(fname, *args):
    """
    Save data to disk.

    Parameters:
       name (str): string with the destination (path+filename). To select
             the file format the user must provide one file extension among
             `.pyobs` (default) and `.json.gz`.
       args : data to save (see below)

    Notes:
       Available output formats:

       * pyobs: the default binary format, with automatic checksums
         for file corruptions and fast read/write speed. It is based
         on the `bison <https://mbruno46.github.io/bison/>`_ file
         format. `args` can be an arbitrary sequence of python basic
         types, numpy arrays and observables. (check the bison
         documentation for more information).

       * json.gz: apart from the compression with gunzip, the file
         is a plain text file generated with json format, for easy
         human readability and compatibility with other programming
         languages (json format is widely supported). Currently this
         format supports only a single observable.

    Examples:
       >>> obsA = pyobs.observable('obsA')
       >>> obsA.create('A',data)
       >>> pyobs.save('~/analysis/obsA.json.gz', obsA)
    """
    if os.path.isfile(fname) is True:  # pragma: no cover
        raise pyobs.PyobsError(f"File {fname} already exists")

    if ".pyobs" in fname:
        fmt = default
    elif ".json.gz" in fname:
        fmt = json
        if len(args) > 1 or not isinstance(args[0], pyobs.observable):
            raise pyobs.PyobsError("json file format supports only single observable")
    else:  # pragma: no cover
        raise pyobs.PyobsError("Format not supported")

    fmt.save(fname, *args)


def load(fname):
    """
    Load the observable/data from disk.

    Parameters:
       name (str): string with the source file (path+filename)

    Returns:
       observable: the loaded observable

    Examples:
       >>> obsA = pyobs.load('~/analysis/obsA.json.gz')
    """

    if not os.path.isfile(fname):  # pragma: no cover
        raise pyobs.PyobsError(f"File {fname} does not exists")

    if ".pyobs" in fname:
        fmt = default
    elif ".json.gz" in fname:
        fmt = json
    else:  # pragma: no cover
        raise pyobs.PyobsError("Format not supported")

    return fmt.load(fname)
