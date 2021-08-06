import pyobs
import numpy

val=[-230, 23]
cov=[(val[0]*5)**2, (val[1]*2)**2]

N=1000
tau=2.0

rng = pyobs.random.generator('math')
data = rng.acrandn(val,cov,tau,N)

obsA = pyobs.observable()
obsA.create('EnsA',data.flatten(),shape=(2,))
print(obsA)

data = rng.acrandn(val,cov,tau,N)
obsB = pyobs.observable()
obsB.create('EnsB',data.flatten(),shape=(2,))

fcore = [
    lambda x,y: x+y,
    lambda x,y: x-y,
    lambda x,y: x*y,
    lambda x,y: x/y,
]

for f in fcore:
    [v0, e0] = (f(obsA,obsB)).error()
    g = pyobs.num_grad(obsA, lambda x: f(x, obsB.mean))
    g0 = pyobs.gradient(g)
    g = pyobs.num_grad(obsB, lambda x: f(obsA.mean, x))
    g1 = pyobs.gradient(g)
    [v1, e1] = pyobs.derobs([obsA, obsB], f(obsA.mean, obsB.mean), [g0,g1]).error()
    assert numpy.all(abs(v1-v0) < 1e-12)
    assert numpy.all(abs(e1/e0-1) < 1e-12)


vec = numpy.array([1,1])

funary = []
for f in fcore:
    for _f in [lambda x: f(x,vec), lambda x: f(vec, x)]:
        [v0, e0] = (_f(obsA)).error()
        g = pyobs.num_grad(obsA, _f)
        g0 = pyobs.gradient(g)
        [v1, e1] = pyobs.derobs([obsA], _f(obsA.mean), [g0]).error()
        assert numpy.all(abs(v1-v0) < 1e-12)
        assert numpy.all(abs(e1/e0-1) < 1e-12)



    