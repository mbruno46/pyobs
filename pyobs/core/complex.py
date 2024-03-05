################################################################################
#
# complex.py: temporary interface for complex matrices
# Copyright (C) 2020-2024 Mattia Bruno
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
import time

class complex_observable:
    def __init__(self, real=None, imag=None):
        pyobs.assertion((not real is None) or (not imag is None), "Unexpected real/imaginary parts")
        if real is None:
            self.real = pyobs.observable(description='auxiliary observable')
            self.real.create_from_cov(f"{time.time()}", imag.mean*0.0, imag.mean.flatten()*0.0)
        else:
            self.real = real
            
        if imag is None:
            self.imag = pyobs.observable(description='auxiliary observable')
            self.imag.create_from_cov(f"{time.time()}", real.mean*0.0, real.mean.flatten()*0.0)
        else:
            self.imag = imag
            
        pyobs.assertion(imag.shape == real.shape, "Real and imaginary parts must have same dimensions")
        self.mean = self.real.mean + 1j * self.imag.mean
    
    def unary_derobs(self, func, grad):
        mean = func(self.mean)
        if not type(mean) in (tuple,list):
            mean = [mean]

        out = []
        for i, m in enumerate(mean):

            g_re = [pyobs.gradient(lambda x: grad[i](mean,x).real, self.real.mean), 
                    pyobs.gradient(lambda x: grad[i](mean,x).imag, self.imag.mean)]

            g_im = [pyobs.gradient(lambda x: grad[i](mean,x).imag, self.real.mean), 
                    pyobs.gradient(lambda x: grad[i](mean,x).real, self.imag.mean)]            
            
            re = pyobs.derobs([self.real, self.imag], m.real, g_re)
            im = pyobs.derobs([self.real, self.imag], m.imag, g_im)
                
            out.append(pyobs.complex_observable(re, im))

        if len(out)==1:
            return out[0]
        return out
    
    def __matmul__(self, y):
        pyobs.assertion(isinstance(y, pyobs.complex_observable), "Only multiplication among complex matrices is supported")
        re, im = self.real @ y.real - self.imag @ y.imag, self.real @ y.imag + self.imag @ y.real
        return pyobs.complex_observable(re, im)

    def __mul__(self,y):
        pyobs.assertion(isinstance(y, pyobs.complex_observable), "Only multiplication among complex matrices is supported")
        re, im = self.real * y.real - self.imag * y.imag, self.real * y.imag + self.imag * y.real
        return pyobs.complex_observable(re, im)

    def __str__(self):
        return self.real.__str__() + '\n' + self.imag.__str__()
    
    def T(self):
        return pyobs.complex_observable(pyobs.transpose(self.real), pyobs.transpose(self.imag))
    
    def inv(self):
        return self.unary_derobs(lambda x: np.linalg.inv(x), [lambda mean, x: -mean[0] @ x @ mean[0]])
    
    def eig(self):
        # d l_n = (v_n, dA v_n)
        gw = lambda mean, x: np.diag(mean[1].T @ x @ mean[1])

        # d v_n = sum_{m \neq n} (v_m, dA v_n) / (l_n - l_m) v_m
        def gv(mean, y):
            w = mean[0].real
            tmp = mean[1].T.conj() @ y @ mean[1]
            h = []
            for m in range(self.mean.shape[0]):
                h.append((w != w[m]) * 1.0 / (w - w[m] + 1e-16))
            h = np.array(h)
            return mean[1] @ (tmp * h)

        return self.unary_derobs(lambda x: np.linalg.eig(x), [gw, gv])    
