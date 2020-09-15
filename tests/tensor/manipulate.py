import pyobs
import numpy

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

corr = pyobs.obs()
corr.create(f'm{mass:g}-{L}x{T}', data.flatten(), dims=(len(xax),))
print(corr)
[c,dc] = corr.error()

[c0, dc0] = pyobs.reshape(corr, (len(xax)//2,2)).error()
assert numpy.any(abs(dc0 - numpy.reshape(dc, (len(xax)//2,2)))) < 1e-12

corr1 = corr[0:4]
corr2 = corr[4:]

corr3 = pyobs.concatenate(corr1,corr2)
print(corr3)
[c0, dc0] = corr3.error()
assert numpy.any(abs(c0 - numpy.transpose(c))) < 1e-12
assert numpy.any(abs(dc0 - numpy.transpose(dc))) < 1e-12

[c0, dc0] = pyobs.transpose(corr).error()
assert numpy.any(abs(dc0 - numpy.transpose(dc))) < 1e-12

[c0, dc0] = pyobs.sort(corr).error()
idx = numpy.argsort(c)
assert numpy.any(abs(dc0 - dc[idx])) < 1e-12

[c0, dc0] = pyobs.diag(corr).error()
assert numpy.any(abs(dc0 - numpy.diag(dc))) < 1e-12
