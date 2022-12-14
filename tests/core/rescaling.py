import pyobs
import numpy

val= numpy.array([1, 1e-3, 1e-6])*1e-12
cov= (val*0.1)**2

N=1000
tau=0.0

rng = pyobs.random.generator('test=1')
data = rng.markov_chain(val,cov,tau,N)
obsA = pyobs.observable()
obsA.create('a',data.flatten(),shape=(3,))

print(obsA.error())
print((obsA*1).error())

obsB = pyobs.observable()
obsB.create('b',data.flatten()*1e12,shape=(3,))

print()
print((obsB*1e-12).error())