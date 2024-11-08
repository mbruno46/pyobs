import pyobs
import numpy

pyobs.set_verbose('create')

val=0.5
sig=val*0.01

N=5000
tau=10.0

rng = pyobs.random.generator('test=1')
data = rng.markov_chain(val,sig**2,tau,N).flatten()

obsA = pyobs.observable()
obsA.create('a', data)

a, da = obsA.error()

assert abs(a - numpy.mean(data)) < 1e-10
assert abs(da - 0.0055072)<1e-8

a, da = obsA.error(errinfo = {'a': pyobs.errinfo(gamma_bias=False)})

assert abs(a - numpy.mean(data)) < 1e-10
assert abs(da - 0.00547229)<1e-8
