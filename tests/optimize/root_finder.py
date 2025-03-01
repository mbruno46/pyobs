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

interp = pyobs.optimize.interpolate(x, yobs)
print(interp.coeff , numpy.array(aex))
print(interp([x[2]]), y[2])
print('solve')
print(interp.solve(y[1], bracket=[x[0],x[2]]), x[1])

f = lambda x,a: a[0] + a[1]*x + a[2]*x*x - 0.3
dfx = lambda x,a: a[1] + 2*a[2]*x
dfa = lambda x,a: [1, x, x*x]

print('\n root')
x0 = pyobs.optimize.root_scalar(interp.coeff, lambda x,a: f(x,a), dfx, dfa, bracket=[min(x),max(x)])
print(x0)