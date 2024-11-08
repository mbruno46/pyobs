import pyobs
import numpy

pyobs.set_verbose('error')

val=0.5
sig=val*0.01

N=1000
tau=0.0

rng = pyobs.random.generator(67)

for tau, N in zip([0.0,4.0],[10000,4000]):
    data = rng.markov_chain(val,sig**2,tau,N)
    obsA = pyobs.observable()
    obsA.create('EnsA',data)

    [a,da] = obsA.error()
    dda = obsA.error_of_error()
    assert abs(a-val) < da
    assert abs(da - sig) < dda
    print(f'Estimated error {da[0]:g} vs expeceted {sig:g}')

_tau = obsA.tauint()
assert abs(_tau['EnsA'][0]-tau) < 1.5*_tau['EnsA'][1]

obsA.variance()