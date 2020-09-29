import pyobs
import numpy

val=[-230, 23]
cov=[(val[0]*5)**2, (val[1]*2)**2]

N=1000
tau=2.0

data = pyobs.random.acrandn(val,cov,tau,N)

obsA = pyobs.observable()
obsA.create('EnsA',data.flatten(),shape=(2,))
print(obsA)