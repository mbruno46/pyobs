import pyobs
import numpy
import os

L=[32,32,32]
p=os.path.realpath(__file__)[:-9]
data = numpy.loadtxt(f'{p}/mfield-32x32x32-obs.txt.gz')

mfobs = pyobs.observable()
mfobs.create(f'test-mfield',data,lat=L)
mfobs.peek()

print(mfobs)

print('a random operation',mfobs * 45 + mfobs**2)

[v, e] = mfobs.error(errinfo={'test-mfield': pyobs.errinfo(W=12.0)})
print(v,e)
[v, e] = mfobs.error(errinfo={'test-mfield': pyobs.errinfo(W=12.01)})
print(v,e)
[v, e] = mfobs.error(errinfo={'test-mfield': pyobs.errinfo(W=[12.01])})
print(v,e)
[v, e] = mfobs.error(errinfo={'test-mfield': pyobs.errinfo(W=[12.0])})
print(v,e)

def func(x):
    return x*45 + x**2

assert pyobs.error_bias4(mfobs, func) < func(mfobs).error_of_error()

print(mfobs.tauint())

mfobs2 = pyobs.concatenate(mfobs, pyobs.log(mfobs))
[cm,dcm] = mfobs2.covariance_matrix(errinfo={'test-mfield': pyobs.errinfo(W=10)})
print('diag cov mat = ', mfobs2.error()[1]**2)
print('cov mat \n',pyobs.valerr(cm,dcm))

[v0, e0] = mfobs.error()
[v1, e1] = mfobs.blocked({"test-mfield": [2,2,2]}).error()
assert numpy.all(abs(v0-v1) < 1e-12)
assert abs(e0/e1-1) < 0.2

# with holes
v = numpy.prod(L)
sites = numpy.arange(v)

rng = pyobs.random.generator('mfield')
holes = rng.sample_boolean(v)

mfobs2 = pyobs.observable()
mfobs2.create('test-mfield', data[holes], icnfg=list(sites[holes]), lat=L)
[v0,e0] = mfobs.error()
[v1,e1] = mfobs2.error()
print(mfobs, mfobs2)
assert numpy.all(numpy.abs(v0-v1) < e1)
assert numpy.all(e1>e0)