#################################################################################
#
# random.py: routines for the generation of synthetic autocorrelated data
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
import hashlib

__all__ = ["generator"]


class generator:
    """
    A random number generator based on `numpy.random`. It preserves the internal
    state and guarantees complete reproducibility.
    
    Parameters:
       seed (string or int): if a string is passed a sha256 bytearray is generated
          and used to seed the random number generator.
    """

    def __init__(self, seed):
        if isinstance(seed, str):
            b = bytearray(hashlib.sha256(seed.encode()).digest())[0::8]
            _seed = numpy.uint32(numpy.frombuffer(b, dtype="<i4"))[0]
        elif isinstance(seed, (int, numpy.int32, numpy.int64)):
            _seed = numpy.uint32(seed)
        self.seed = _seed
        print(f"Random generator initialized with seed = {self.seed} [{seed}]")
        numpy.random.seed(self.seed)
        self.state = numpy.random.get_state()

    def sample_normal(self, N):
        numpy.random.set_state(self.state)
        r = numpy.random.normal(0.0, 1.0, N)
        self.state = numpy.random.get_state()
        return r

    def acrand(self, mu, sigma, tau, N):
        """
        Create synthetic autocorrelated Monte Carlo data

        Parameters:
           mu (int or float): the central value
           sigma (float): the target error
           tau (float): the integrated autocorrelation time
           N (int): the number of configurations/measurements

        Returns:
           list : the synthetic data

        Examples:
           >>> rng = pyobs.random.generator('test')
           >>> data = rng.acrand(0.1234,0.0001,4.0,1000)
           >>> obs = pyobs.observable(description='test-acrand')
           >>> obs.create('A',data)
           >>> print(obs)
           0.12341(26)

        """
        r = self.sample_normal(N)

        if tau > 0.0:
            f = numpy.exp(-1.0 / tau)
        else:
            f = 0.0
            tau = 0.5
        ff = numpy.sqrt(1.0 - f * f)
        rn = numpy.zeros((N,))
        rn[0] = ff * r[0]
        for i in range(1, N):
            rn[i] = ff * r[i] + f * rn[i - 1]
        return list(mu + sigma * numpy.sqrt(N / (2 * tau)) * rn)

    def acrandn(self, mu, cov, tau, N):
        """
        Create synthetic correlated Monte Carlo 1-D data

        Parameters:
           mu (list of array): the central values of corresponding observable;
              a 1-D array is expected
           cov (array): the covariance matrix of the observable (in absence of
              autocorrelations); if a 1-D array is passed, a diagonal covariance
              matrix is assumed
           tau (float): the integrated autocorrelation time
           N (int): the number of configurations/measurements

        Returns:
           numpy.ndarray : 2-D array with the synthetic data, such that each row corresponds to a configuration
        """
        if len(mu) != numpy.shape(cov)[0]:  # pragma: no cover
            raise ValueError
        nf = len(mu)
        if tau > 0:
            f = numpy.exp(-1.0 / tau)
        else:
            f = 0.0
        ff = numpy.sqrt(1.0 - f * f)

        r = numpy.reshape(self.sample_normal(N * nf), (N, nf))
        rn = numpy.zeros((N, nf))
        rn[0, :] = ff * r[0, :]

        for i in range(N):
            rn[i, :] = ff * r[i, :] + f * rn[i - 1, :]

        if numpy.ndim(cov) == 1:
            Q = numpy.diag(numpy.sqrt(cov))
        else:
            [w, v] = numpy.linalg.eig(cov)
            Q = numpy.diag(numpy.sqrt(w)) @ v.T
        return rn @ Q + numpy.array(mu)
