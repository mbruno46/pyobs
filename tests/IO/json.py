import pyobs
import os
import numpy

data = pyobs.random.acrandn([2.31,3.14],[0.2**2,0.1**2],4.0,3000)

test = pyobs.observable(desc='save/load test')
test.create('test',data.flatten(),icnfg=range(0,3000*4,4),rname='rep1',shape=(2,))

test.add_syst_err('syst. err',[0.01,0.01])

p=os.path.realpath(__file__)[:-7]
data = numpy.loadtxt(f'{p}../core/mfield-32x32x32-obs.txt.gz')
mfobs = pyobs.observable()
mfobs.create(f'test-mfield',data,lat=[32,32,32])

test1 = pyobs.concatenate(test[0], test[0]*mfobs)

pyobs.save('./test-io.json.gz',test1)
test2 = pyobs.load('./test-io.json.gz')

[v, e] = test1.error()
[v2, e2] = test2.error()
assert numpy.all(e==e2)
assert numpy.all(test1.mean == test2.mean)

for key in test.rdata:
    assert numpy.all(test1.rdata[key].delta == test2.rdata[key].delta)

for key in test.mfdata:
    assert numpy.all(test1.mfdata[key].delta == test2.mfdata[key].delta)
    
try:
    pyobs.save('./test-io.json.gz',test1)
    assert False
except pyobs.PyobsError:
    print('error caught')

try:
    pyobs.save('./test-io.dat',test1)
    assert False
except pyobs.PyobsError:
    print('error caught')
    
try:
    pyobs.load('./test-io-2.json.gz')
    assert False
except pyobs.PyobsError:
    print('error caught')

try:
    pyobs.load('./test-io-2.dat')
    assert False
except pyobs.PyobsError:
    print('error caught')

os.popen('rm ./test-io.json.gz')
