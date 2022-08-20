import pyobs
import numpy
import os

fname = f"{os.path.dirname(os.path.abspath(__file__))}/observable.xml.gz"

res = pyobs.load(fname)
[v, e] = pyobs.remove_tensor(res).error()

assert abs(v-1.12207286) < 1e-8
assert abs(e-0.01189337) < 1e-8

rng = pyobs.random.generator('io')
nrep = 100
data = [rng.markov_chain([2.31,3.14],[0.2**2,0.1**2],4.0,1000).flatten() for _ in range(nrep)]

test = pyobs.observable(description='save/load bdio')
test.create('ensA',data,rname=[f"r{i}" for i in range(nrep)],shape=(2,))

pyobs.save('test.xml.gz',test)
test2 = pyobs.remove_tensor(pyobs.load('test.xml.gz'))

[v1, e1] = test.error()
[v2, e2] = test2.error()

assert numpy.all(abs(v1-v2) < 1e-12)
assert numpy.all(abs(e1-e2) < 1e-12)

os.popen('rm ./test.xml.gz')