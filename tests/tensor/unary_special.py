import pyobs
import numpy
import scipy

T=32
L=16
mass=0.25
p=0.0

xax=range(T//2)
corr_ex = [pyobs.qft.free_scalar.Cphiphi(x,mass,p,T,L)+2.0 for x in xax]
cov_ex = pyobs.qft.free_scalar.cov_Cphiphi(mass,p,T,L)[0:T//2,0:T//2]

N=1000
tau=2.0
rng = pyobs.random.generator('unary')
data = rng.markov_chain(corr_ex,cov_ex/N,tau,N)

corr = pyobs.observable()
corr.create(f'm{mass:g}-{L}x{T}', data.flatten(), shape=(len(xax),))
print(corr)
[c,dc] = corr.error()

mat = pyobs.reshape(corr/corr.mean[0]/1.01, (T//8,T//8))
print(mat)
flist = pyobs.tensor.unary.__all__
flist.remove('arccosh')

for f in flist:
    [v0, e0] = pyobs.__dict__[f](mat).error()
    mean = numpy.__dict__[f](mat.mean)
    g = pyobs.num_grad(mat, numpy.__dict__[f])
    g0 = pyobs.gradient(g)
    [v1, e1] = pyobs.derobs([mat], mean, [g0]).error()
    assert numpy.all(numpy.fabs(e1-e0) < 2e-9)
    
mat = pyobs.reshape(corr, (T//8,T//8))
flist = ['besselk']
slist = ['kv']
args = [1.0]

for i in range(len(flist)):
    [v0, e0] = pyobs.__dict__[flist[i]](args[i], mat).error()
    f = lambda x: scipy.special.__dict__[slist[i]](args[i], x)
    mean = f(mat.mean)
    g = pyobs.num_grad(mat, f)
    g0 = pyobs.gradient(g)
    [v1, e1] = pyobs.derobs([mat], mean, [g0]).error()
    assert numpy.all(numpy.fabs(e1-e0) < 1e-10)
    
