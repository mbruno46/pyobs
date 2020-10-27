import pyobs
import os
import numpy

data = pyobs.random.acrandn([2.31,3.14],[0.2**2,0.1**2],4.0,3000)

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

os.popen('rm ./test-io.pyobs')
