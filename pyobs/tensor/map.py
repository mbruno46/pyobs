#################################################################################
#
# container.py: generic container
# Copyright (C) 2020-2026 Mattia Bruno
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

import pyobs
import numpy as np

__all__ = ["TensorMap", "tensormap"]


def make_key(x):
    if isinstance(x, np.ndarray):
        return (x.shape, x.dtype.str, x.tobytes())
    elif isinstance(x, list):
        return tuple(x)
    return x


class TensorMap:
    """
    A multi-dimensional container mapping arbitrary tagged indices to objects.

    TensorMap generalizes NumPy-like indexing to arbitrary (hashable or
    non-hashable) tag types such as strings, numbers, lists, or NumPy arrays.
    Internally, tags are converted to a hashable representation, while the
    original objects are preserved for user interaction.

    Examples
       >>> a1, a2 = np.array(...), np.array(...)
       >>> tm = pyobs.TensorMap('name', 'key')
       >>> tm.append(a1, name='array1', key=np.array([1,2]))
       >>> tm.append(a2, name='array1', key=np.array([3,4]))
       >>> tm['array1', np.array([3,4])]
       [ ... content of a1 ... ]
       >>> tm['array1', :]
       [[ ... content of a1 ... ],
        [ ... content of a1 ... ]]
       >>> print(tm)
       ...
    """

    def __init__(self, *tags):
        self.ndim = len(tags)
        self._tags = tags

        for t in tags:
            setattr(self, t, [])
        self._data = {}
        self._keys = [[] for _ in range(self.ndim)]

        self.__repr__ = self.__str__

    def append(self, obj, *args, **kwargs):
        """
        Insert an object with associated tags.

        Parameters
        ----------
        obj : any
            The object to store.
        *args : tuple
            A sequence of tags, one per axis. The number of tags must match
            the dimensionality of the TensorMap. The order is assumed to match
            the order of the tags passed in the constructor.
        *kwargs : tuple
            A sequence of tags, one per axis with key=value pairs.
            The number of tags must match the dimensionality of the TensorMap.
        """

        if len(args) == 0:
            pyobs.assertion(len(kwargs) == self.ndim, f"Expected {self.ndim} kwargs")
        elif len(args) == self.ndim:
            if len(kwargs) > 0:
                print(f"Warning : {kwargs} ignored")
            kwargs = {self._tags[i]: args[i] for i in range(self.ndim)}
        else:
            pyobs.PyobsError(
                f"Expected either arguments or keyword arguments, not both!"
            )

        reordered_tags = [kwargs[t] for t in self._tags]
        key = tuple(make_key(t) for t in reordered_tags)
        self._data[key] = obj

        for i, k in enumerate(key):
            if not k in self._keys[i]:
                self._keys[i].append(k)
                getattr(self, self._tags[i]).append(reordered_tags[i])
            # [i][k] = reordered_tags[i]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) != self.ndim:
            raise ValueError(f"Need {self.ndim} indices")

        norm_key = []
        for i, k in enumerate(key):
            _k = make_key(k)
            if _k == slice(None):
                norm_key.append(self._keys[i])
            else:
                norm_key.append({_k})

        result = []
        for tags, obj in self._data.items():
            if all(tags[i] in norm_key[i] for i in range(self.ndim)):
                result.append(obj)

        return result[0] if len(result) == 1 else result

    def __str__(self):
        out = [""]
        for key, obj in self._data.items():
            for i, k in enumerate(key):
                j = self._keys[i].index(k)
                out.append(f" - {self._tags[i]}: {getattr(self, self._tags[i])[j]}")
            out.append(str(obj))
        return "\n".join(out)


def tensormap(*args):
    return TensorMap(*args)
