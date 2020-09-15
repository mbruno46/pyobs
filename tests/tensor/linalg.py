import pyobs
import numpy

mat = [2,0.5,0.5,3.5]
cov = (numpy.array(mat)*0.05)**2

data = pyobs.random.acrandn(mat,cov,1.0,4000)

omat = pyobs.obs()
omat.create('test',data.flatten(),dims=(2,2))
print(omat)

#check inverse
def func(x):
    return numpy.linalg.inv(x)
g = pyobs.num_grad(omat, func)
[v0, e0] = pyobs.derobs([omat], func(omat.mean), [g]).error()
[v1, e1] = pyobs.linalg.inv(omat).error()

assert numpy.all(numpy.fabs(v0-v1) < 1e-12)
assert numpy.all(numpy.fabs(e0-e1) < 1e-10)

# check eigenvalues, both symmetric and non-symmetric cases
def func(x):
    return numpy.linalg.eig(x)[0]

omatsym = (omat + pyobs.transpose(omat))*0.5
g = pyobs.num_grad(omatsym, func)
[v0, e0] = pyobs.derobs([omatsym], func(omatsym.mean), [g]).error()
[v1, e1] = (pyobs.linalg.eig(omatsym)[0]).error()

assert numpy.all(numpy.fabs(v0-v1) < 1e-12)
assert numpy.all(numpy.fabs(e0-e1) < 1e-10)

g = pyobs.num_grad(omat, func)
[v0, e0] = pyobs.derobs([omat], func(omat.mean), [g]).error()
[v1, e1] = (pyobs.linalg.eigLR(omat)[0]).error()

assert numpy.all(numpy.fabs(v0-v1) < 1e-12)
assert numpy.all(numpy.fabs(e0-e1) < 1e-10)

# check right eigenvectors, both symmetric and non-symmetric cases
def func(x):
    return numpy.linalg.eig(x)[1]

omatsym = (omat + pyobs.transpose(omat))*0.5
g = pyobs.num_grad(omatsym, func)
[v0, e0] = pyobs.derobs([omatsym], func(omatsym.mean), [g]).error()
[v1, e1] = (pyobs.linalg.eig(omatsym)[1]).error()

assert numpy.all(numpy.fabs(v0-v1) < 1e-12)
assert numpy.all(numpy.fabs(e0-e1) < 1e-10)

g = pyobs.num_grad(omat, func)
[v0, e0] = pyobs.derobs([omat], func(omat.mean), [g]).error()
[v1, e1] = (pyobs.linalg.eigLR(omat)[1]).error()

assert numpy.all(numpy.fabs(v0-v1) < 1e-12)
assert numpy.all(numpy.fabs(e0-e1) < 1e-10)

# check left eigenvectors
def func(x):
    return numpy.linalg.eig(x.T)[1]

g = pyobs.num_grad(omat, func)
[v0, e0] = pyobs.derobs([omat], func(omat.mean), [g]).error()
[v1, e1] = (pyobs.linalg.eigLR(omat)[2]).error()

assert numpy.all(numpy.fabs(v0-v1) < 1e-12)
assert numpy.all(numpy.fabs(e0-e1) < 1e-10)

omatsqrt = pyobs.linalg.matrix_power(omatsym, -0.5)
[v0, e0] = (omatsqrt @ omatsym @ omatsqrt).error()
assert numpy.all(v0 - numpy.eye(len(v0)) < 1e-12)
