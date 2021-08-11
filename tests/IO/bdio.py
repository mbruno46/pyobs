import pyobs
import os

rng = pyobs.random.generator('io')
data = rng.acrandn([2.31,3.14],[0.2**2,0.1**2],4.0,3000)

test = pyobs.observable(description='save/load bdio')
test.create('ensA',data.flatten(),icnfg=range(0,3000*4,4),rname='rep1',shape=(2,))

text = 'user info 1'
pyobs.save('test-io.bdio', text, test[0], test[1])

res = pyobs.load('test-io.bdio')

assert res['file_content'][0] == text
for o in res['observables']:
    i = res['observables'].index(o)
    [v0, e0] = test[i].error()
    [v1, e1] = o.error()
    assert abs(v0-v1) < 1e-12
    assert abs(e0-e1) < 1e-12

os.popen('rm ./test-io.bdio')