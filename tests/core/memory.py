import pyobs
import numpy

val=0.5

N=1000
err=val*0.01
sig=err*numpy.sqrt(N)
tau=0.0

rng = pyobs.random.generator('memory')
data = rng.markov_chain(val,err**2,tau,N)
obsA = pyobs.observable()
obsA.create('EnsA',data)

obsA.peek()

pyobs.memory.info()

N=100000
data = rng.markov_chain(val,err**2,tau,N)
obsB = pyobs.observable()
obsB.create('EnsB',data)

pyobs.memory.info()
obsB.peek()

del obsB

pyobs.memory.info()
pyobs.memory.available()
