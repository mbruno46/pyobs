import pyobs
import numpy

pyobs.set_verbose('error')

val=0.5
sig=val*0.01

N=1000
tau=0.0

numpy.random.seed(46)

data = pyobs.random.acrand(val,sig,tau,N)
obsA = pyobs.observable()
obsA.create('EnsA',data)

[a,da] = obsA.error()
dda = obsA.error_of_error()
assert abs(a-val) < da
assert abs(da - sig) < dda
print(f'Estimated error {da[0]:g} vs expeceted {sig:g}')

N=10000
tau=0.0

data = pyobs.random.acrand(val,sig,tau,N)
obsA = pyobs.observable()
obsA.create('EnsA',data)

[a,da] = obsA.error()
dda = obsA.error_of_error()
assert abs(a-val) <da
assert abs(da - sig) < dda
print(f'Estimated error {da[0]:g} vs expeceted {sig:g}')

[a, da] = obsA.error(errinfo={'EnsA': pyobs.errinfo(W=0)})
assert abs(a-val) < da

N=4000
tau=4.0

data = pyobs.random.acrand(val,sig,tau,N)
obsA = pyobs.observable()
obsA.create('EnsA',data)

[a,da] = obsA.error()
dda = obsA.error_of_error()
assert abs(a-val) < da
assert abs(da - sig) < dda
print(f'Estimated error {da[0]:g} vs expeceted {sig:g}')

_tau = obsA.tauint()
assert abs(_tau['EnsA'][0]-tau) < 1.5*_tau['EnsA'][1]
