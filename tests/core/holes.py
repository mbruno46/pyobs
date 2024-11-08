import pyobs
import numpy

val= 1.234
cov= 0.002**2

N=100
tau=2.0
einfo = {'A': pyobs.errinfo(Stau=5)}

rng = pyobs.random.generator(46)

dat = numpy.zeros((N,2))
dat[:,0] = rng.markov_chain(val,cov,tau,N).flatten()

obsA = pyobs.observable()
obsA.create('A',dat.flatten(),icnfg=range(5,1000,10),shape=(2,))
obsA.peek()
print(obsA[0])
obsA[0].error(errinfo=einfo, plot=True)

N=1000
dat = numpy.zeros((N,2))
dat[:,1] = rng.markov_chain(-val,cov,tau,N).flatten()

obsB = pyobs.observable()
obsB.create('A',dat.flatten(),icnfg=range(0,1000,1),shape=(2,))
print(obsB.delta['A:0'].delta)
print(obsB[1])

obsC = obsA + obsB
print(obsC)
obsC.error(errinfo=einfo, plot=True)

e = obsC.error(errinfo=einfo)[1]
assert abs(e[0]/obsA[0].error(errinfo=einfo)[1] - 1.0) < 0.1
assert e[1] == obsB[1].error(errinfo=einfo)[1]