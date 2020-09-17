import pyobs
import os
import numpy

data = pyobs.random.acrandn([2.31,3.14],[0.2**2,0.1**2],4.0,3000)

test = pyobs.obs(desc='save/load test')
test.create('test',data.flatten(),icnfg=range(0,3000*4,4),rname='rep1',shape=(2,))

[v, e] = test.error()

pyobs.save('./test-io.json.gz',test)

test2 = pyobs.load('./test-io.json.gz')

[v2, e2] = test2.error()
assert numpy.all(e==e2)
assert numpy.all(test.mean == test2.mean)

for key in test.rdata:
    assert numpy.all(test.rdata[key].delta == test2.rdata[key].delta)

for key in test.mfdata:
    assert numpy.all(test.mfdata[key].delta == test2.mfdata[key].delta)
    
os.popen('rm ./test-io.json.gz')