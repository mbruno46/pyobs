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

print(obsA[0] - obsA[1])

print(-obsA)

print('matmul', obsA @ numpy.array([2.,3.]))

print(1.0 + obsA)
print(3.4 - obsA)
print(3.14 / obsA)

pyobs.memory.info()
obsB = obsA[0]
obsB += obsA[1]
pyobs.memory.info()
obsB.peek()

obsB -= obsA[0]

obsB *= 8.0

obsB /= obsA[0]**2
print(obsB)

[c,dc] = obsA.error()
mat = numpy.c_[[0,1],c,dc]
print(pyobs.textable(mat,fmt=['d',0,0]))
print(pyobs.textable(mat))

obsC = pyobs.concatenate(obsA[0],obsA[1]**2)
obsA[1] = obsA[1]**2
[v1, e1] = obsC.error()
[v2, e2] = obsA.error()
assert numpy.all(abs(v1-v2) < 1e-12)
assert numpy.all(abs(e1-e2) < 1e-12)

print('diag cov = ',e2**2)
[cm,dcm] = obsA.covariance_matrix(errinfo={'EnsA': pyobs.errinfo(W=20)})
print('cov matrix = \n',pyobs.valerr(cm,dcm))
