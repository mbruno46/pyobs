import pyobs
import numpy

T=16
L=16
mass=0.30
p=0.0

xax=range(T//2)
corr_ex = [pyobs.qft.free_scalar.Cphiphi(x,mass,p,T,L) for x in xax]
cov_ex = 1e-2*pyobs.qft.free_scalar.cov_Cphiphi(mass,p,T,L)[0:T//2,0:T//2]

N=4000
tau=1.0
rng = pyobs.random.generator('tensor')
data = rng.markov_chain(corr_ex,cov_ex,tau,N)

corr = pyobs.observable()
corr.create(f'm{mass:g}-{L}x{T}', data.flatten(), shape=(len(xax),))
print(corr)
[c,dc] = corr.error()

corr2d = pyobs.reshape(corr, (len(xax)//2,2))
[c0, dc0] = corr2d.error()
assert numpy.all(abs(dc0 - numpy.reshape(dc, (len(xax)//2,2))) < 1e-12)

corr1 = corr[0:4]
corr2 = corr[4:]

tmp = pyobs.observable()
tmp = pyobs.concatenate(tmp, corr1)
del tmp

tmp = pyobs.observable()
tmp = pyobs.concatenate(corr1, tmp)
del tmp


corr3 = pyobs.concatenate(corr1,corr2)
print(corr3)
[c0, dc0] = corr3.error()
assert numpy.all(abs(c0 - numpy.transpose(c)) < 1e-12)
assert numpy.all(abs(dc0 - numpy.transpose(dc)) < 1e-12)

[c0, dc0] = pyobs.transpose(corr).error()
assert numpy.all(abs(dc0 - numpy.transpose(dc)) < 1e-12)

[c0, dc0] = pyobs.sort(corr).error()
idx = numpy.argsort(c)
assert numpy.all(abs(dc0 - dc[idx]) < 1e-12)

[c0, dc0] = pyobs.diag(pyobs.reshape(corr[0:4],(2,2))).error()
assert numpy.all(abs(dc0 - numpy.array([dc[0],dc[3]])) < 1e-12)

print('orig')
vec = corr[0:4]
print(vec)

print('repeat')
print(pyobs.repeat(vec,3))

print('tile')
print(pyobs.tile(vec,(3,1)))

print('stack')
print(pyobs.stack([vec,vec,vec]))

print('roll\n')
[c0, dc0] = pyobs.roll(corr, -1).error()
assert numpy.all(numpy.abs(c[1:] - c0[:-1]) < 1e-12)
assert numpy.all(numpy.abs(dc[1:] - dc0[:-1]) < 1e-12)

print('diag')
print(pyobs.diag(vec))