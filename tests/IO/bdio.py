import pyobs
import os
import numpy

rng = pyobs.random.generator('io')
nrep = 100
data = [rng.acrandn([2.31,3.14],[0.2**2,0.1**2],4.0,1000).flatten() for _ in range(nrep)]

test = pyobs.observable(description='save/load bdio')
test.create('ensA',data,rname=[f"r{i}" for i in range(nrep)],shape=(2,))

text = 'user info 1'
arr = numpy.array([5.,6.], dtype=numpy.float64)
pyobs.save('test-io.bdio', text, arr, test[0], test[1])

res = pyobs.load('test-io.bdio')

assert res[0] == text
assert numpy.all(res[1]==arr)
for i in [0,1]:
    [v0, e0] = test[i].error()
    [v1, e1] = res[2+i].error()
    assert abs(v0-v1) < 1e-12
    assert abs(e0-e1) < 1e-12

os.popen('rm ./test-io.bdio')