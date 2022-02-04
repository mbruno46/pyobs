import pyobs
import numpy

x = numpy.arange(4.,7.,1.)
aex = [0.02, 0.06, 0.]
y = aex[0] + aex[1]*x + aex[2]*x*x
dy = (y*0.1)**2

N=800
tau=3.0

rng = pyobs.random.generator('root')
data = rng.markov_chain(y,dy**2,tau,N)
yobs = pyobs.observable()
yobs.create('test',data.flatten(),shape=(len(x),))

int = pyobs.optimize.interpolate(x, yobs)
print(int.coeff)

f = lambda x,a: a[0] + a[1]*x + a[2]*x*x - 0.3
dfx = lambda x,a: a[1] + 2*a[2]*x
dfa = lambda x,a: [1, x, x*x]


x0 = pyobs.optimize.root_scalar(int.coeff, lambda x,a: f(x,a), dfx, dfa, bracket=[min(x),max(x)])
print(x0)