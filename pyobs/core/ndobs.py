################################################################################
#
# ndobs.py: definition and properties of the core class of the library
# Copyright (C) 2020-2021 Mattia Bruno
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
import copy
import gzip, json
import os, pwd, re
import datetime

import pyobs

from .data import delta
from .cdata import cdata
from .error import gamma_error, covariance, plot_piechart
from .transform import transform

__all__ = ["observable"]


class observable:
    """
    Class defining an observable

    Parameters:
       orig (observable, optional): creates a copy of orig
       description (str, optional): description of the observable

    Examples:
       >>> from pyobs import observable
       >>> a = observable(description='test')
    """

    def __init__(self, orig=None, description="unknown"):
        if orig is None:
            pyobs.check_type(description, "text", str)
            self.description = description
            self.www = [
                pwd.getpwuid(os.getuid())[0],
                os.uname()[1],
                datetime.datetime.now().strftime("%c"),
            ]
            self.shape = []
            self.size = 0
            self.mean = []
            self.ename = []
            self.delta = {}
            self.cdata = {}
        else:
            if isinstance(orig, pyobs.observable):
                self.description = orig.description
                self.www = orig.www
                self.shape = orig.shape
                self.size = numpy.prod(self.shape)
                self.mean = numpy.array(orig.mean)  # or orig.mean.copy()

                self.ename = [e for e in orig.ename]
                self.delta = {}
                for key in orig.delta:
                    self.delta[key] = orig.delta[key].copy()

                self.cdata = {}
                for key in orig.cdata:
                    self.cdata[key] = orig.cdata[key].copy()
                pyobs.memory.add(self)
            else:  # pragma: no cover
                raise pyobs.PyobsError("Unexpected orig argument")
        pyobs.memory.add(self)

    @pyobs.log_timer("create")
    def create(self, ename, data, icnfg=None, rname=None, shape=(1,), lat=None):
        """
        Create an observable

        Parameters:
           ename (str): label of the ensemble
           data (array, list of arrays): the data generated from a single
              or multiple replica
           icnfg (array of ints or list of arrays of ints, optional): indices
              of the configurations corresponding to data; if not passed the
              measurements are assumed to be contiguous and to start from index 0
           rname (str or list of str, optional): identifier of the replica; if
              not passed integers from 0 are automatically assigned
           shape (tuple, optional): shape of the observable, data must be passed accordingly
           lat (list of ints, optional): the size of each dimension of the master-field;
              if passed data is assumed to be obtained from observables measured at different
              sites and `icnfg` is re-interpreted as the index labeling the sites; if `icnfg`
              is not passed data is assumed to be contiguous on all sites.

        Note:
           For data and icnfg array can mean either a list or a 1-D numpy.array.
           If the observable has already been created, calling create again will add
           a new replica to the same ensemble.

        Examples:
           >>> data = [0.43, 0.42, ... ] # a scalar observable
           >>> a = pyobs.observable(description='test')
           >>> a.create('EnsembleA',data)

           >>> data0 = [0.43,0.42, ... ] # replica 0
           >>> data1 = [0.40,0.41, ... ] # replica 1
           >>> a = pyobs.observable(description='test')
           >>> a.create('EnsembleA',[data0,data1],rname=['r0','r1'])

           >>> data = [0.43, 0.42, 0.44, ... ]
           >>> icnfg= [  10,   11,   13, ... ]
           >>> a = pyobs.observable(description='test')
           >>> a.create('EnsembleA',data,icnfg=icnfg)

           >>> data = [1.0, 2.0, 3.0, 4.0, ... ]
           >>> a = pyobs.observable(description='matrix')
           >>> a.create('EnsembleA',data,shape=(2,2))

        Examples:
           >>> data = [0.43, 0.42, 0.44, ... ]
           >>> lat = [64,32,32,32]
           >>> a = pyobs.observable(description='test-mf')
           >>> a.create('EnsembleA',data,lat=lat)

           >>> data = [0.43, 0.42, 0.44, ... ]
           >>> idx = [0, 2, 4, 6, ...] # measurements on all even points of time-slice
           >>> lat = [32, 32, 32]
           >>> a = pyobs.observable(description='test-mf')
           >>> a.create('EnsembleA',data,lat=lat,icnfg=idx)
        """
        pyobs.check_type(ename, "ename", str)
        pyobs.assertion(":" not in ename, f"Column symbol not allowed in ename {ename}")
        pyobs.check_type(shape, "shape", tuple)
        self.shape = shape
        self.size = numpy.prod(shape)
        mask = range(self.size)
        if ename not in self.ename:
            self.ename.append(ename)

        pyobs.check_type(data, "data", list, numpy.ndarray)
        if isinstance(
            data[0],
            (int, numpy.int32, numpy.int64, float, numpy.float64, numpy.float32),
        ):
            data = [data]

        R = len(data)
        nc = [len(data[ir]) // self.size for ir in range(R)]
        if rname is None:
            rname = list(range(R))
        elif isinstance(rname, (str, int, numpy.int32, numpy.int64)):
            rname = [rname]
        if icnfg is None:
            icnfg = [range(_nc) for _nc in nc]
        elif isinstance(icnfg[0], (int, numpy.int32, numpy.int64)):
            icnfg = [icnfg]

        pyobs.check_type(rname, "rname", list)
        pyobs.assertion(len(rname) == R, "Incompatible rname and data")
        pyobs.check_type(icnfg, "icnfg", list)
        for ir in range(R):
            pyobs.assertion(
                len(icnfg[ir]) * self.size == len(data[ir]),
                f"Incompatible icnfg[{ir}] and data[{ir}], for shape={shape}",
            )

        mean_data = numpy.zeros((self.size,))
        for ir in range(R):
            mean_data += numpy.sum(numpy.reshape(data[ir], (nc[ir], self.size)), 0)
        mean_data *= 1.0 / sum(nc)

        if numpy.size(self.mean) != 0:
            N0 = sum([self.delta[key].n for key in self.delta])
            mean_old = numpy.reshape(self.mean, (self.size,))
            self.mean = (N0 * mean_old + sum(nc) * mean_data) / (N0 + sum(nc))
            shift = sum(nc) * (mean_old - mean_data) / (N0 + sum(nc))
            for key in self.delta:
                self.delta[key].delta += shift[:, None]
        else:
            self.mean = mean_data

        for ir in range(R):
            key = f"{ename}:{rname[ir]}"
            self.delta[key] = delta(mask, icnfg[ir], data[ir], self.mean, lat)

        self.mean = numpy.reshape(self.mean, self.shape)
        pyobs.memory.update(self)

    def create_from_cov(self, cname, value, covariance):
        """
        Create observables based on covariance matrices

        Parameters:
           cname (str): label that uniquely identifies the data set
           value (array): central value of the observable; only 1-D arrays are supported
           covariance (array): a 2-D covariance matrix; if `covariance` is a 1-D array of
              same length as `value`, a diagonal covariance matrix is assumed.

        Examples:
           >>> mpi = pyobs.observable(description='pion masses, charged and neutral')
           >>> mpi.create_from_cov('mpi-pdg18',[139.57061,134.9770],[0.00023**2,0.0005**2])
           >>> print(mpi)
           139.57061(23)    134.97700(50)
        """
        if isinstance(value, (int, float, numpy.float64, numpy.float32)):
            self.mean = numpy.reshape(value, (1,))
            cov = numpy.reshape(covariance, (1,))
        else:
            self.mean = numpy.array(value)
            cov = numpy.array(covariance)
        self.shape = numpy.shape(self.mean)
        pyobs.assertion(
            numpy.ndim(self.shape) == 1,
            "Unexpected value, only 1-D arrays are supported",
        )
        self.size = numpy.prod(self.shape)
        if cov.shape != (self.size,) and cov.shape != (self.size, self.size):
            raise pyobs.PyobsError(f"Unexpected shape for covariance {cov.shape}")
        pyobs.check_type(cname, "cname", str)
        self.cdata[cname] = cdata(cov, list(range(self.size)))
        pyobs.memory.update(self)

    def add_syst_err(self, name, err):
        """
        Add a systematic error to the observable

        Parameters:
           name (str): label that uniquely identifies the syst. error
           err (array): array with the same dimensions of the observable
              with the systematic error

        Examples:
           >>> data = [0.198638, 0.403983, 1.215960, 1.607684, 0.199049, ... ]
           >>> vec = pyobs.observable(description='vector')
           >>> vec.create('A',data,shape=(4,))
           >>> print(vec)
           0.201(13)    0.399(26)    1.199(24)    1.603(47)
           >>> vec.add_syst_err('syst.err',[0.05,0.05,0,0])
           >>> print(vec)
           0.201(52)    0.399(56)    1.199(24)    1.603(47)

        """
        pyobs.check_type(name, "name", str)
        pyobs.assertion(name not in self.cdata, f"Label {name} already used")
        pyobs.assertion(
            numpy.shape(err) == self.shape,
            f"Unexpected error, dimensions do not match {self.shape}",
        )
        cov = numpy.reshape(numpy.array(err) ** 2, (self.size,))
        self.cdata[name] = cdata(cov, list(range(self.size)))
        pyobs.memory.update(self)

    def __del__(self):
        pyobs.memory.rm(self)

    def ename_from_delta(self):
        self.ename = []
        for key in self.delta:
            name = key.split(":")[0]
            if name not in self.ename:
                self.ename.append(name)

    def rename(self, src, dst):
        """
        Rename ensembles and replica.

        Parameters:
           src (string or tuple): if a string is passed all replica belonging to the
              ensembles defined by `src` are renamed into `dst`; if a tuple with two
              strings is passed, then the first element is taken as the ensemble tag,
              the second as the replica tag and only this replicum is renamed.
           dst (string or tuple): the new name for the ensembles and replica; the behavior
              is the same of `src`.

        Examples:
           >>> obsA = pyobs.observable()
           >>> obsA.create('EnsA', data, icnfg, rname=['r001','r002'])
           >>> obsA.rename('EnsA','EnsembleA') # rename both replica with the new ensemble tag
           >>> obsA.rename(('EnsembleA','r001'),('EnsA','stream0')) # rename only a single replica
        """

        def rename_delta(e0, r0, e1, r1):
            _tag = f"{e0}:{r0}"
            tag = f"{e1}:{r1}"
            pyobs.assertion(e0 in self.ename, f"Ensemble tag {e0} not found")
            pyobs.assertion(_tag in self.delta, f"Repliaca tag {r0} not found")
            self.delta[tag] = self.delta.pop(_tag)

        if isinstance(src, str):
            pyobs.check_type(dst, "dst", str)
            for key in self.delta:
                e, r = key.split(":")
                if e == src:
                    rename_delta(src, r, dst, r)
        elif isinstance(src, tuple):
            pyobs.check_type(dst, "dst", tuple)
            rename_delta(src[0], src[1], dst[0], dst[1])

        self.ename_from_delta()

    ##################################
    # pretty string representations

    def peek(self):
        """
        Display a brief summary of the content of the observable, including
        its memory footprint and requirements (for error computation), its
        description and ensemble/replica content

        Example:
           >>> obs.peek()
           Observable with shape = (1, 4)
            - description: vector-test
            - created by mbruno at macthxbruno.fritz.box on Wed Aug 11 10:51:26 2021
            - size: 82 KB
            - mean: [[0.20007161 0.40085252 1.19902686 1.60184989]]
            - Ensemble A
               - Replica 0 with mask [0, 1, 2, 3] and ncnfg 500
                    temporary additional memory required 0.015 MB

        """
        print(f"Observable with shape = {self.shape}")
        print(f" - description: {self.description}")
        print(f" - created by {self.www[0]} at {self.www[1]} on {self.www[2]}")
        print(f" - size: {pyobs.memory.get(self)}")
        print(f" - mean: {self.mean}")

        for name in self.ename:
            print(f" - Ensemble {name}")
            m = 0
            for key in self.delta:
                rn = key.split(":")
                if rn[0] == name:
                    outstr = f'    - {"Replica" if self.delta[key].lat is None else "Master-field"} {rn[1]}'
                    outstr = f'{outstr} with {f"ncnfg {self.delta[key].n}" if self.delta[key].lat is None else f"lattice {self.delta[key].lat}"}'
                    print(outstr)
                    mm = (
                        self.delta[key].ncnfg() * 8.0 * 2.0
                        if self.delta[key].lat is None
                        else (self.delta[key].vol() + 1) * 8.0
                    )
                    m = (mm > m) * mm + (mm <= m) * m
            print(f"         temporary additional memory required {m/1024.**2:.2g} MB")

        for cd in self.cdata:
            print(f" - Data {cd} with cov. matrix {self.cdata[cd].cov.shape}")
        print("")

    def __str__(self):
        [v, e] = self.error()
        D = len(self.shape)
        out = ""
        if D == 1:
            out += "\t".join([pyobs.valerr(v[i], e[i]) for i in range(self.shape[0])])
            out += "\n"
        elif D == 2:
            for i in range(self.shape[0]):
                out += "\t".join(
                    [pyobs.valerr(v[i, j], e[i, j]) for j in range(self.shape[1])]
                )
                out += "\n"
        return out

    def __repr__(self):  # pragma: no cover
        return self.__str__()

    ##################################
    # overloaded indicing and slicing

    def set_mean(self, mean):
        if isinstance(mean, (int, float, numpy.float32, numpy.float64)):
            self.mean = numpy.reshape(mean, (1,))
        else:
            self.mean = numpy.array(mean)
        self.shape = numpy.shape(self.mean)
        self.size = numpy.size(self.mean)

    @pyobs.log_timer("slice")
    def slice(self, *args):
        """
        Slices an N-D observable.

        Parameters:
           *args: accepted arguments are lists, arrays, slices or integers
                  with the indices used for the extraction. `[]` is interpreted
                  as taking all elements along that given axis, like slice(None).
                  The number of input arguments must match the dimension of the
                  observable.

        Returns:
           observable: the sliced N-D observable.
           Note that the number of dimensions does not change even
           when only a single coordinate is selected along a given axis.

        Examples:
           >>> obs = pyobs.observable()
           >>> obs.create('EnsA', data, shape=(4,3,6))
           >>> obs.slice([0],[],[0,2,4]) # returns an observable with shape = (1,3,3)
        """
        na = len(args)
        pyobs.assertion(na == len(self.shape), "Unexpected argument")

        def f(x):
            return pyobs.slice_ndarray(x, *args)

        return transform(self, f)

    def __getitem__(self, args):
        if isinstance(args, (int, numpy.int32, numpy.int64, slice, numpy.ndarray)):
            args = [args]
        na = len(args)
        pyobs.assertion(na == len(self.shape), "Unexpected argument")

        def f(x):
            return pyobs.slice_ndarray(x, *args)

        return transform(self, f)

    def __setitem__(self, args, yobs):
        if isinstance(args, (slice, numpy.ndarray)):
            args = tuple(args)
        elif isinstance(args, (int, numpy.int32, numpy.int64)):
            args = tuple([args])
        else:
            args = tuple(
                [
                    [a] if isinstance(a, (int, numpy.int32, numpy.int64)) else a
                    for a in args
                ]
            )
        if self.mean[tuple(args)].size == 1:
            pyobs.assertion(yobs.size == 1, "set item : dimensions do not match")
        else:
            pyobs.assertion(
                self.mean[tuple(args)].shape == yobs.shape,
                "set item : dimensions do not match",
            )
        self.mean[tuple(args)] = yobs.mean

        idx = numpy.arange(self.size).reshape(self.shape)[tuple(args)]
        submask = idx.flatten()

        for key in yobs.delta:
            pyobs.assertion(
                key in self.delta, "Ensembles do not match; can not set item"
            )
            self.delta[key].assign(submask, yobs.delta[key])

        for key in yobs.cdata:
            pyobs.assertion(
                key in self.cdata, "Covariance data do not match; can not set item"
            )
            self.cdata[key].assign(submask, yobs.cdata[key])

    ##################################
    # overloaded basic math operations

    def __addsub__(self, y, sign):
        g0 = pyobs.gradient(lambda x: x, self.mean, gtype="diag")
        if isinstance(y, observable):
            g1 = pyobs.gradient(lambda x: sign * x, y.mean, gtype="diag")
            return pyobs.derobs([self, y], self.mean + sign * y.mean, [g0, g1])
        else:
            return pyobs.derobs([self], self.mean + sign * y, [g0])

    def __add__(self, y):
        return self.__addsub__(y, +1)

    def __sub__(self, y):
        return self.__addsub__(y, -1)

    def __neg__(self):
        g0 = pyobs.gradient(lambda x: -x, self.mean, gtype="diag")
        return pyobs.derobs([self], -self.mean, [g0])

    def __mul__(self, y):
        if isinstance(y, pyobs.observable):
            if self.shape == y.shape:
                g0 = pyobs.gradient(lambda x: x * y.mean, self.mean, gtype="diag")
                g1 = pyobs.gradient(lambda x: self.mean * x, y.mean, gtype="diag")
            elif self.shape == (1,):
                g0 = pyobs.gradient(lambda x: x * y.mean, self.mean, gtype="full")
                g1 = pyobs.gradient(lambda x: self.mean * x, y.mean, gtype="diag")
            elif y.shape == (1,):
                g0 = pyobs.gradient(lambda x: x * y.mean, self.mean, gtype="diag")
                g1 = pyobs.gradient(lambda x: self.mean * x, y.mean, gtype="full")
            else:
                raise pyobs.PyobsError("Shape mismatch, cannot multiply")
            return pyobs.derobs([self, y], self.mean * y.mean, [g0, g1])
        else:
            # if gradient below was 'full' it would allow scalar_obs * array([4,5,6])
            # which would create a vector obs. right now that generates an error
            # but is faster for large gradients
            g0 = pyobs.gradient(lambda x: x * y, self.mean, gtype="diag")
            return pyobs.derobs([self], self.mean * y, [g0])

    def __matmul__(self, y):
        if isinstance(y, observable):
            g0 = pyobs.gradient(lambda x: x @ y.mean, self.mean)
            g1 = pyobs.gradient(lambda x: self.mean @ x, y.mean)
            return pyobs.derobs([self, y], self.mean @ y.mean, [g0, g1])
        else:
            g0 = pyobs.gradient(lambda x: x @ y, self.mean)
            return pyobs.derobs([self], self.mean @ y, [g0])

    def reciprocal(self):
        new_mean = numpy.reciprocal(self.mean)
        g0 = pyobs.gradient(lambda x: -x * (new_mean ** 2), self.mean, gtype="diag")
        return pyobs.derobs([self], new_mean, [g0])

    def __truediv__(self, y):
        if isinstance(y, observable):
            return self * y.reciprocal()
        else:
            return self * (1 / y)

    # __array_priority__=1000
    __array_ufunc__ = None

    def __radd__(self, y):
        return self + y

    def __rsub__(self, y):
        return -self + y

    def __rmul__(self, y):
        return self * y

    def __rtruediv__(self, y):
        return self.reciprocal() * y

    def __rmatmul__(self, y):
        g0 = pyobs.gradient(lambda x: y @ x, self.mean)
        return pyobs.derobs([self], y @ self.mean, [g0])

    def __pow__(self, a):
        new_mean = self.mean ** a
        g0 = pyobs.gradient(
            lambda x: a * x * self.mean ** (a - 1), self.mean, gtype="diag"
        )
        return pyobs.derobs([self], new_mean, [g0])

    # in-place operations
    def __iadd__(self, y):
        tmp = self + y
        del self
        self = pyobs.observable(tmp)
        del tmp
        return self

    def __isub__(self, y):
        tmp = self - y
        del self
        self = pyobs.observable(tmp)
        del tmp
        return self

    def __imul__(self, y):
        tmp = self * y
        del self
        self = pyobs.observable(tmp)
        del tmp
        return self

    def __itruediv__(self, y):
        tmp = self / y
        del self
        self = pyobs.observable(tmp)
        del tmp
        return self

    ##################################
    # Error functions

    def error_core(self, errinfo, plot, pfile):
        sigma_tot = numpy.zeros(self.shape)
        dsigma_tot = numpy.zeros(self.shape)
        sigma = {}
        for e in self.ename:
            if e in errinfo:
                res = gamma_error(self, e, plot, pfile, errinfo[e])
            else:
                res = gamma_error(self, e, plot, pfile)
            sigma[e] = numpy.reshape(res[0], self.shape)
            sigma_tot += sigma[e]
            dsigma_tot += numpy.reshape(res[1], self.shape)

        for cd in self.cdata:
            sigma[cd] = self.cdata[cd].sigmasq(self.shape)
            sigma_tot += sigma[cd]
        return [sigma, sigma_tot, dsigma_tot]

    @pyobs.log_timer("error")
    def error(self, errinfo={}, plot=False, pfile=None):
        """
        Estimate the error of the observable, by summing in quadrature
        the systematic errors with the statistical errors computed from
        all ensembles and master fields.

        Parameters:
           errinfo (dict, optional): dictionary containg one instance of
              the errinfo class for each ensemble/master-field. The errinfo
              class provides additional details for the automatic or manual
              windowing procedure in the Gamma method. If not passed default
              parameters are assumed.
           plot (bool, optional): if specified a plot is produced, for
              every element of the observable, and for every ensemble/master-field
              where the corresponding element has fluctuations. In addition
              one piechart plot is produced for every element, showing the
              contributions to the error from the various sources, only if there
              are multiple sources, ie several ensembles. It is recommended to use
              the plotting function only for observables with small dimensions.
           pfile (str, optional): if specified all plots produced with the flag
              `plot` are saved to disk, using `pfile` as base name with an additional
              suffix.


        Returns:
           list of two arrays: the central value and error of the observable.

        Note:
           By default, the errors are computed with the Gamma method, with the `Stau`
           parameter equal to 1.5. Additionally the jackknife method can be used
           by passing the appropriate `errinfo` dictionary with argument `bs` set
           to a non-zero integer value. For master fields the error is computed
           using the master-field approach and the automatic windowing procedure
           requires the additional argument `k` (see main documentation), which
           by default is zero, but can be specified via the errinfo dictionary.
           Through the `errinfo` dictionary the user can treat every ensemble
           differently, as explained in the examples below.


        Examples:
           >>> obsA = pyobs.observable('obsA')
           >>> obsA.create('A',dataA) # create the observable A from ensemble A
           >>> [v,e] = obsA.error() # collect central value and error in v,e
           >>> einfo = {'A': errinfo(Stau=3.0)} # specify non-standard Stau for ensemble A
           >>> [_,e1] = obsA.error(errinfo=einfo)
           >>> print(e,e1) # check difference in error estimation

           >>> obsB = pyobs.observable('obsB')
           >>> obsB.create('B',dataB) # create the observable B from ensemble B
           >>> obsC = obsA * obsB # derived observable with fluctuations from ensembles A,B
           >>> einfo = {'A': errinfo(Stau=3.0), 'B': errinfo(W=30)}
           >>> [v,e] = obsC.error(errinfo=einfo,plot=True)
        """
        [sigma, sigma_tot, _] = self.error_core(errinfo, plot, pfile)

        if plot:  # pragma: no cover
            h = [len(self.ename), len(self.cdata)]
            if sum(h) > 1:
                plot_piechart(self.description, sigma, sigma_tot)

        return [self.mean, numpy.sqrt(sigma_tot)]

    def error_breakdown(self, errinfo={}):
        """
        Returns a dictionary with the squared error of each component.
        """
        [sigma, _, _] = self.error_core(errinfo, False, False)
        return sigma

    def error_of_error(self, errinfo={}):
        """
        Returns the error of the error based on the analytic
        prediction obtained by U. Wolff.

        Parameters:
           errinfo (dict, optional): see the documentation of the `error`
              method.

        Returns:
           array: the error of the error
        """
        [_, _, dsigma_tot] = self.error_core(errinfo, False, None)
        return numpy.sqrt(dsigma_tot)

    def tauint(self):
        """
        Estimates the integrated autocorrelation time and its error for every
        ensemble, with the automatic windowing procedure.

        Notes:
           To be added in future versions: support for arbitrary values of Stau
        """
        # to be improved - add errinfo
        tau = {}
        for e in self.ename:
            res = gamma_error(self, e)
            tau[e] = [
                numpy.reshape(res[2][:, 0], self.shape),
                numpy.reshape(res[2][:, 1], self.shape),
            ]

        return tau

    def variance(self):
        """
        Estimates the integrated autocorrelation function and its error for every
        ensemble.

        Returns:
            dict: one key for each ensemble.

        Notes:
            for every ensemble a list of 3 arrays is returned, corresponding to the
            x-axis (MC time), the integrated autocorrelation function and its error.
        """
        cgam = {}
        for e in self.ename:
            res = gamma_error(self, e)
            cgam[e] = res[3:6]
        return cgam

    def covariance_matrix(self, errinfo):
        """
        Estimates the covariance matrix using a fixed window `W` for all observables.

        Parameters:
           errinfo (dict): see the documentation of the `error` method. For each ensemblee
              the parameter `W` is used to define the summation window.

        Returns:
           list of two arrays: the covariance matrix and its error.

        Examples:
            >>> obsA = pyobs.observable()
            >>> obsA.create('EnsA', data, shape=((8,)))
            >>> [cm, dcm] = obsA.covariance_matrix(errinfo = {'EnsA', pyobs.errinfo(W=10)})
            >>> print(pyobs.valerr(cm,dcm))
        """
        covmat = numpy.zeros((self.size, self.size))
        dcovmat = numpy.zeros((self.size, self.size))
        for e in self.ename:
            if e in errinfo:
                res = covariance(self, e, errinfo[e].W)
                covmat += res[0]
                dcovmat += res[1]
        return [covmat, dcovmat]

    def blocked(self, blocks):
        """
        Returns an observable with blocked/binned fluctuations.

        Parameters:
           blocks (dict): dictionary where the key specifies the ensemble and
                the value the blocking strategy. If the ensemble was generated
                from a Monte Carlo chain, then an integer number is expected. If
                the ensemble is a master-field then a list of integers is expected.

        Note:
           After the blocking procedure the knowledge of configuration indices is
           lost and replaced by an index running over the blocks/bins from 0 to the
           maximal value. To keep the analysis consistent, the same blocking procedure
           must be applied to all observables involved.

        Examples:
            >>> obsA = pyobs.observable()
            >>> len(data)
            1000
            >>> obsA.create('EnsA', data)
            >>> obsAbin = obsA.blocked({'EnsA': 20})
            >>> obsAbin.peek() # the 1000 configs are compressed to 50 bins of length 20
            Observable with shape = (1,)
             - description: unknown
             - size: ... KB
             - mean: [...]
             - Ensemble EnsA
                - Replica 0 with ncnfg 50
                     temporary additional memory required ... MB
        """
        res = pyobs.observable(description=self.description)
        res.set_mean(self.mean)
        res.ename = [e for e in self.ename]

        for key in self.delta:
            e = key.split(":")[0]
            if e in blocks:
                res.delta[f"{e}:{key.split(':')[1]}-blocking{blocks[e]}"] = self.delta[
                    key
                ].blocked(blocks[e])
            else:
                res.delta[key] = self.delta[key].copy()

        for key in self.cdata:
            res.cdata[key] = self.cdata[key].copy()

        pyobs.memory.update(res)
        return res
