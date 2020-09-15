import pyobs
import numpy

val=[-2.3, 0.5]
cov=[(val[0]*0.05)**2, (val[1]*0.02)**2]

N=1000
tau=2.0

data = pyobs.random.acrandn(val,cov,tau,N)
