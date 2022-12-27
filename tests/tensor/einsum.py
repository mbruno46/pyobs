import pyobs
import numpy

mat = [2,0.5,0.5,3.5]
cov = (numpy.array(mat)*0.05)**2

rng = pyobs.random.generator('einsum')
data = rng.markov_chain(mat,cov,1.0,4000)

omat = pyobs.observable()
omat.create('test',data.flatten(),shape=(2,2))

tr1 = pyobs.einsum('aa',omat)
tr2 = pyobs.trace(omat)
[v1, e1] = tr1.error()
[v2, e2] = tr2.error()
assert numpy.any(abs(v1-v2) < 1e-10)
assert numpy.any(abs(e1-e2) < 1e-10)

p1 = pyobs.einsum('ab,bc->ac',omat,omat)
p2 = omat @ omat
[v1, e1] = p1.error()
[v2, e2] = p2.error()
assert numpy.any(abs(v1-v2) < 1e-10)
assert numpy.any(abs(e1-e2) < 1e-10)

p1 = pyobs.einsum('ab,bc->ac',omat,omat.mean)
p2 = omat @ omat.mean
[v1, e1] = p1.error()
[v2, e2] = p2.error()
assert numpy.any(abs(v1-v2) < 1e-10)
assert numpy.any(abs(e1-e2) < 1e-10)

for f in ['sum']:
    [v0, e0] = pyobs.__dict__[f](omat,axis=0).error()
    func = lambda x: numpy.__dict__[f](x,axis=0)
    mean = func(omat.mean)
    g = pyobs.num_grad(omat, func)
    g0 = pyobs.gradient(g)
    [v1, e1] = pyobs.derobs([omat], mean, [g0]).error()
    assert numpy.all(numpy.fabs(e1-e0) < 1e-10)
