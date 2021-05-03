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