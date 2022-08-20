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
import pyobs

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

    def sample_flat(self, elements, N):
        numpy.random.set_state(self.state)
        r = numpy.random.choice(elements, N)
        self.state = numpy.random.get_state()
        return r
        
    def sample_boolean(self, N):
        return self.sample_flat([True, False], N)
        
    def acrand(self, tau, N, n=1):
        r = numpy.reshape(self.sample_normal(N * n), (N, n))

        if tau > 0.5:
            f = numpy.exp(-1.0 / tau)
        else:
            f = 0.0
            tau = 0.5
        ff = numpy.sqrt(1.0 - f * f)
        rn = numpy.zeros(numpy.shape(r))
        rn[0, :] = ff * r[0, :]
        for i in range(1, N):
            rn[i, :] = ff * r[i, :] + f * rn[i - 1, :]
        return rn

    def markov_chain(self, mu, cov, taus, N, couplings=None):
        """
        Create synthetic autocorrelated Monte Carlo data

        Parameters:
           mu (float or 1D array): the central value
           cov (float or array): the target covariance matrix; if a 1-D array is
              passed, a diagonal covariance matrix is assumed
           taus (float or array): the autocorrelation time(s). Values smaller
           than 0.5 are ignored and set to automatically to 0.5.
           N (int): the number of configurations/measurements
           couplings (optional, float or array): the couplings of the modes
           to the observable.

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
        mu = pyobs.array(mu)
        pyobs.assertion(numpy.ndim(mu) == 1, "only 1D arrays are supported")
        na = len(mu)

        cov = pyobs.array(cov)
        pyobs.assertion(
            numpy.shape(cov)[0] == na, "covariance matrix does not match central values"
        )

        taus = pyobs.array(taus)
        taus = 0.5 * (taus <= 0.5) + taus * (taus > 0.5)
        nt = len(taus)
        if couplings is None:
            couplings = pyobs.array(numpy.ones((na, nt)))
        else:
            couplings = pyobs.array(couplings)
            pyobs.assertion(
                numpy.shape(couplings) == (na, nt),
                f"unexpected couplings for {na} values and {nt} modes",
            )

        rn = numpy.zeros((N, na))
        _c = numpy.stack([couplings] * N)
        for i in range(len(taus)):
            rn += _c[:, :, i] * self.acrand(taus[i], N, na)

        pref = numpy.sqrt(N / (2 * (couplings**2 @ taus)))

        if numpy.ndim(cov) == 1:
            Q = numpy.diag(numpy.sqrt(cov))
        else:
            [w, v] = numpy.linalg.eig(cov)
            Q = numpy.diag(numpy.sqrt(w)) @ v.T

        if na == 1:
            return (mu + (pref * rn) @ Q).reshape((N,))
        return mu + (pref * rn) @ Q


#     def acrandn(self, mu, cov, tau, N):
#         """
#         Create synthetic correlated Monte Carlo 1-D data

#         Parameters:
#            mu (list of array): the central values of corresponding observable;
#               a 1-D array is expected
#            cov (array): the covariance matrix of the observable (in absence of
#               autocorrelations); if a 1-D array is passed, a diagonal covariance
#               matrix is assumed
#            tau (float): the integrated autocorrelation time
#            N (int): the number of configurations/measurements

#         Returns:
#            numpy.ndarray : 2-D array with the synthetic data, such that each row corresponds to a configuration
#         """
#         if len(mu) != numpy.shape(cov)[0]:  # pragma: no cover
#             raise ValueError
#         nf = len(mu)
#         if tau > 0:
#             f = numpy.exp(-1.0 / tau)
#         else:
#             f = 0.0
#         ff = numpy.sqrt(1.0 - f * f)

#         r = numpy.reshape(self.sample_normal(N * nf), (N, nf))
#         rn = numpy.zeros((N, nf))
#         rn[0, :] = ff * r[0, :]

#         for i in range(N):
#             rn[i, :] = ff * r[i, :] + f * rn[i - 1, :]

#         if numpy.ndim(cov) == 1:
#             Q = numpy.diag(numpy.sqrt(cov))
#         else:
#             [w, v] = numpy.linalg.eig(cov)
#             Q = numpy.diag(numpy.sqrt(w)) @ v.T
#         return rn @ Q + numpy.array(mu)
