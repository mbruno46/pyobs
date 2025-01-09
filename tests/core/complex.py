import pyobs
import numpy as np

pyobs.set_verbose('create')

val=np.array([0.5, 1.5])
sig=val*0.01

N=5000
tau=10.0

rng = pyobs.random.generator('test=1')
data_r = rng.markov_chain(val,sig**2,tau,N).flatten()
data_i = rng.markov_chain(val,sig**2,tau,N).flatten()

obsB = pyobs.observable()
obsB.create('a', data_r + 1j*data_i, shape=(2,))

obsA_r = pyobs.observable()
obsA_r.create('a', data_r, shape=(2,))

obsA_i = pyobs.observable()
obsA_i.create('a', data_i, shape=(2,))

print(obsA_r, obsA_i)

def check(A,B):
    b, db = B.error()
    a, da = A.error()
    assert abs(np.sum(a - b)) < 1e-10
    assert abs(np.sum(da - db))<1e-10
    
for B,A in zip([obsB.real(), obsB.imag()], [obsA_r, obsA_i]):
    check(A,B)
    
obsC = 1j*obsB

for B,A in zip([-obsC.real(), obsC.imag()], [obsA_i, obsA_r]):
    check(A,B)

obsC = 1j + obsB

for B,A in zip([obsC.real(), obsC.imag()], [obsA_r, obsA_i+1.0]):
    check(A,B)

vec = np.array([1.2, 3j])
obsC = vec * obsB

for B,A in zip([obsC.real(), obsC.imag()], 
               [vec.real * obsA_r - vec.imag * obsA_i, 
                vec.real * obsA_i + vec.imag * obsA_r]):
    check(A,B)
    
mat = np.array([[1.2, 3j],[3j, 1.2]])
obsC = mat @ obsB

for B,A in zip([obsC.real(), obsC.imag()], 
               [mat.real @ obsA_r - mat.imag @ obsA_i, 
                mat.real @ obsA_i + mat.imag @ obsA_r]):
    check(A,B)

    
#####################

class complex_observable:
    """
    Interface for complex observables. It is a simple wrapper for
    the real and imaginary parts of a vector or matrix, which are
    individually normal pyobs observables. They must have the same
    shape.

    Parameters:
       real (observable): the real part
       imag (observable): the imaginary part

    Examples:
       >>> cobs = pyobs.complex_observable(re, im)
       >>> print(cobs)
    """

    def __init__(self, real=None, imag=None):
        pyobs.assertion(
            (not real is None) or (not imag is None), "Unexpected real/imaginary parts"
        )
        if real is None:
            self.real = pyobs.observable(description="auxiliary observable")
            self.real.create_from_cov(
                f"{time.time()}", imag.mean * 0.0, imag.mean.flatten() * 0.0
            )
        else:
            self.real = real

        if imag is None:
            self.imag = pyobs.observable(description="auxiliary observable")
            self.imag.create_from_cov(
                f"{time.time()}", real.mean * 0.0, real.mean.flatten() * 0.0
            )
        else:
            self.imag = imag

        pyobs.assertion(
            imag.shape == real.shape,
            "Real and imaginary parts must have same dimensions",
        )
        self.mean = self.real.mean + 1j * self.imag.mean

    def unary_derobs(self, func, grad):
        mean = func(self.mean)
        if not type(mean) in (tuple, list):
            mean = [mean]

        out = []
        for i, m in enumerate(mean):
            g_re = [
                pyobs.gradient(lambda x: grad[i](mean, x).real, self.real.mean),
                pyobs.gradient(lambda x: -grad[i](mean, x).imag, self.imag.mean),
            ]

            g_im = [
                pyobs.gradient(lambda x: grad[i](mean, x).imag, self.real.mean),
                pyobs.gradient(lambda x: grad[i](mean, x).real, self.imag.mean),
            ]

            re = pyobs.derobs([self.real, self.imag], m.real, g_re)
            im = pyobs.derobs([self.real, self.imag], m.imag, g_im)

            out.append(pyobs.complex_observable(re, im))

        if len(out) == 1:
            return out[0]
        return out

    def __matmul__(self, y):
        pyobs.assertion(
            isinstance(y, pyobs.complex_observable),
            "Only multiplication among complex matrices is supported",
        )
        re, im = (
            self.real @ y.real - self.imag @ y.imag,
            self.real @ y.imag + self.imag @ y.real,
        )
        return pyobs.complex_observable(re, im)

    def __mul__(self, y):
        pyobs.assertion(
            isinstance(y, pyobs.complex_observable),
            "Only multiplication among complex matrices is supported",
        )
        re, im = (
            self.real * y.real - self.imag * y.imag,
            self.real * y.imag + self.imag * y.real,
        )
        return pyobs.complex_observable(re, im)

    def __str__(self):
        return self.real.__str__() + "\n" + self.imag.__str__()

    def T(self):
        return pyobs.complex_observable(
            pyobs.transpose(self.real), pyobs.transpose(self.imag)
        )

    def inv(self):
        """
        Calculates the inverse if the complex observable is a square matrix.

        Returns:
           pyobs.complex_obserable
        """
        return self.unary_derobs(
            lambda x: np.linalg.inv(x), [lambda mean, x: -mean[0] @ x @ mean[0]]
        )

    def eig(self):
        """
        Calculates the complex eigenvalues and eigenvectors, if the observable is a square matrix.

        Returns:
           pyobs.complex_observable: the eigenvalues.
           pyobs.complex_observable: the eigenvectors.
        """
        # d l_n = (v_n, dA v_n)
        gw = lambda mean, x: np.diag(mean[1].T.conj() @ x @ mean[1])

        # d v_n = sum_{m \neq n} (v_m, dA v_n) / (l_n - l_m) v_m
        def gv(mean, y):
            tmp = mean[1].T.conj() @ y @ mean[1]
            h = []
            w = mean[0]
            for _w in w:
                h.append((w != _w) * 1.0 / (w - _w + 1e-16))
            h = np.array(h)
            return mean[1] @ (tmp * h)

        return self.unary_derobs(lambda x: list(np.linalg.eig(x)), [gw, gv])


val=np.array([0.5, 1.5, -1.5, 6.0])
sig=val*0.01

N=5000
tau=10.0

rng = pyobs.random.generator('test=1')
data_r = rng.markov_chain(val,sig**2,tau,N).flatten()
data_i = rng.markov_chain(val,sig**2,tau,N).flatten()

obsB = pyobs.observable()
obsB.create('a', data_r + 1j*data_i, shape=(2,2))
obsB = obsB + pyobs.transpose(obsB)

obsA_r = pyobs.observable()
obsA_r.create('a', data_r, shape=(2,2))
obsA_r = obsA_r + pyobs.transpose(obsA_r)

obsA_i = pyobs.observable()
obsA_i.create('a', data_i, shape=(2,2))
obsA_i = obsA_i + pyobs.transpose(obsA_i)

obsC = complex_observable(obsA_r, obsA_i)

for i in [0,1]:
    for B,A in zip([obsC.eig()[i].real, obsC.eig()[i].imag], 
                   [pyobs.linalg.eig(obsB)[i].real(), pyobs.linalg.eig(obsB)[i].imag()]):
        check(A,B)

for B,A in zip([obsC.inv().real, obsC.inv().imag], 
               [pyobs.linalg.inv(obsB).real(), pyobs.linalg.inv(obsB).imag()]):
    check(A,B)