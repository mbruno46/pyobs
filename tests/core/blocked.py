import pyobs
import numpy

val=[-230, 23]
cov=[(val[0]*5)**2, (val[1]*2)**2]

N=1000
tau=2.0

data = pyobs.random.acrandn(val,cov,tau,N)

obsA = pyobs.observable()
obsA.create('EnsA',data.flatten(),shape=(2,))

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