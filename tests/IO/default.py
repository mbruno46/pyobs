import pyobs
import os
import numpy
import time

rng = pyobs.random.generator('io')
data = rng.markov_chain([2.31,3.14],[0.2**2,0.1**2],4.0,3000)

test = pyobs.observable(description='save/load test')
test.create('test',data.flatten(),icnfg=range(0,3000*4,4),rname='rep1',shape=(2,))

test.add_syst_err('syst. err',[0.01,0.01])

p=os.path.realpath(__file__)[:-10]
data = numpy.loadtxt(f'{p}../core/mfield-32x32x32-obs.txt.gz')
mfobs = pyobs.observable()
mfobs.create(f'test-mfield',data,lat=[32,32,32])

test1 = pyobs.concatenate(test[0], test[0]*mfobs)

pyobs.save('./test-io.pyobs',{'test-observable': test1, 'metadata': 'the test observable', 'indices': [4,8]})
test2 = pyobs.load('./test-io.pyobs')

[v, e] = test1.error()

test3 = test2['test-observable']
[v2, e2] = test3.error()
assert numpy.all(e==e2)
assert numpy.all(test1.mean == test3.mean)
for key in test.delta:
    assert numpy.all(test1.delta[key].delta == test3.delta[key].delta)

assert (test2['metadata'] == 'the test observable')
assert (test2['indices'] == [4,8])

pyobs.memory.info()

os.popen('rm ./test-io.pyobs')
time.sleep(1)

# complex case

data = rng.markov_chain([2.31,3.14],[0.2**2,0.1**2],4.0,3000)
test = pyobs.observable(description='save/load test')
test.create('test',(1-2j)*data.flatten(),icnfg=range(0,3000*4,4),rname='rep1',shape=(2,))
[v, e] = test.error()

pyobs.save('./test-io.pyobs', test)

test2 = pyobs.load('./test-io.pyobs')
[v2, e2] = test2.error()

assert numpy.all(v==v2)
assert numpy.all(e==e2)

os.popen('rm ./test-io.pyobs')
time.sleep(1)

# tensor map

tm = pyobs.tensormap('c', 's')
for x in range(4):
    for y in range(4):
        tm.append(x*x+y*y, c=numpy.array([x,y]), s=str(4*x))

pyobs.save('./test-io.pyobs', tm)

tm2 = pyobs.load('./test-io.pyobs')
for _c in tm.c:
    for _s in tm.s:
        assert tm2[_c, _s] == tm[_c, _s]

del tm, tm2

os.popen('rm ./test-io.pyobs')
