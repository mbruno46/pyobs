import pyobs
import numpy

val=[-230, 23]
cov=[(val[0]*0.05)**2, (val[1]*0.02)**2]

N=1000
tau=2.0

rng = pyobs.random.generator('blocked')

obsA = pyobs.observable()
data = rng.markov_chain(val,cov,tau,N)
obsA.create('EnsA',data.flatten(),rname='r0',shape=(2,))
data = rng.markov_chain(val,cov,tau,N)
obsA.create('EnsA',data.flatten(),rname='r1',shape=(2,))

[a,da] = obsA.error()
dda = obsA.error_of_error()

obsA.peek()
print('Unbinned = ', obsA)

_tau = obsA.tauint()
print('Tauint = ',pyobs.valerr(_tau['EnsA'][0],_tau['EnsA'][1]))

obsAbin = obsA.blocked({'EnsA': 10})
[aa, daa] = obsAbin.error()

obsAbin.peek()
print('Binned   = ',obsAbin)

_tau = obsAbin.tauint()
print('Tauint = ',pyobs.valerr(_tau['EnsA'][0],_tau['EnsA'][1]))

assert numpy.all(abs((da - daa)/dda) < 1 )