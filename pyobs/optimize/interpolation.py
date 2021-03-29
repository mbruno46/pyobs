
#################################################################################
#
# interpolation.py: interpolate a set of data points as observables 
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

import numpy
import pyobs

__all__ = ['interpolate']

class interpolate:
    """
    Finds the polynomial with maximal degree passing exactly 
    through the points given by `x` and `y`.

    Parameters:
       x (array): x coordinates of the points
       y (observable): y coordinates of the points
    """
    def __init__(self,x,y):
        N = len(x)
        if (len(y.shape)>1):
            raise pyobs.PyobsError(f'Unexpected observable with shape ${x.shape}; only vectors are supported')
        if (y.size!=N):
            raise pyobs.PyobsError(f'Unexpected observable with shape ${x.shape} not matching size of x')
        Minv = numpy.linalg.inv(numpy.array([[x[i]**k for k in range(N)] for i in range(N)]).astype('f8'))
        mean = Minv @ y.mean
        g = pyobs.gradient( lambda x: Minv @ x, y.mean)
        self.coeff = pyobs.derobs([y],mean,[g])
        self.k = N
        
    def __call__(self,x):
        """
        Evaluates the polynomial at the locations `x`.

        Parameters:
           x (array or float): location where the interpolated 
               function should be evaluated

        Returns:
           observable: the evaluated function at `x`.
        """
        N = len(x)
        x = numpy.array(x)
        res = pyobs.repeat(self.coeff[0],N)
        for i in range(1,self.k):
            res += pyobs.repeat(self.coeff[i],N) * (x**i)
        return res
