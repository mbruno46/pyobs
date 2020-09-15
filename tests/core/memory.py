import pyobs
import numpy

val=0.5

N=1000
err=val*0.01
sig=err*numpy.sqrt(N)
tau=0.0

data = pyobs.random.acrand(val,err,tau,N)
obsA = pyobs.obs()
obsA.create('EnsA',data)

obsA.peek()

pyobs.memory()

N=100000
data = pyobs.random.acrand(val,err,tau,N)
obsB = pyobs.obs()
obsB.create('EnsB',data)

pyobs.memory()
pyobs.sysmem()
obsB.peek()

del obsB

pyobs.memory()
