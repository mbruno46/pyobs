import pyobs
import numpy

pyobs.set_verbose('slice', True)

T=16
L=16
mass=0.25
p=0.0

xax=range(T//2)
corr_ex = [pyobs.qft.free_scalar.Cphiphi(x,mass,p,T,L) for x in xax]
cov_ex = pyobs.qft.free_scalar.cov_Cphiphi(mass,p,T,L)[0:T//2,0:T//2]

N=4000
tau=1.0
data = pyobs.random.acrandn(corr_ex,cov_ex,tau,N)

corr = pyobs.observable()
corr.create(f'm{mass:g}-{L}x{T}', data.flatten(), shape=(len(xax),))
print(corr)
[c,dc] = corr.error()

tmp = corr.slice([0])
[v,e] = corr.slice([0]).error()
assert abs(v- c[0]) < 1e-12
assert abs(e-dc[0]) < 1e-12
del tmp

obs = pyobs.reshape(corr, (T//4, 2))
print(obs)

[v0, e0] = obs[0,:].error()
[v1, e1] = obs.slice([0],[]).error()
assert numpy.all(abs(v0-v1) < 1e-12)
assert numpy.all(abs(e0-e1) < 1e-12)

[v1, e1] = obs.slice([0],None).error()
assert numpy.all(abs(v0-v1) < 1e-12)
assert numpy.all(abs(e0-e1) < 1e-12)

[v1, e1] = obs.slice([0],numpy.array([0,1])).error()
assert numpy.all(abs(v0-v1) < 1e-12)
assert numpy.all(abs(e0-e1) < 1e-12)

pyobs.set_verbose('slice', False)