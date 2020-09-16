import pyobs
import numpy

pyobs.set_verbose('mfit')
pyobs.set_verbose('diff')

[f, df, _] = pyobs.symbolic.diff('a*exp(-m*x)','x','a,m')

a = 1.0
m = 0.2
T = 16

y  = numpy.array([a*numpy.exp(-m*t) for t in range(T)])
dy = (y*0.1)**2

N = 1000
tau = 1.0

data = pyobs.random.acrandn(y,dy,tau,N)
corr = pyobs.obs()
corr.create('test',data.flatten(),shape=(T,))

[c, dc] = corr.error()
W = 1./dc**2

fit = pyobs.mfit(numpy.arange(T),W,f,df)

pars = fit(corr)
print('fitted pars = ',pars)

[v, e] = pars[0].error()
assert abs(a-v) < 2.*e

[v, e] = pars[1].error()
assert abs(m-v) < 2.*e

# extrapolate value at zero
[yobs] = fit.eval(0.0,pars)

[v, e] = yobs.error()
assert abs(a-v) < 2.*e
