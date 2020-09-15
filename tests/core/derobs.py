import pyobs
import numpy

val=0.002
sig=val*0.5

N=1000
tau=0.0

data = pyobs.random.acrand(val,sig,tau,N)
obsA = pyobs.obs()
obsA.create('EnsA',data)

logobsA = pyobs.log(obsA)
print('obsA = ',obsA)
print('log(obsA) =', logobsA)

[a, da] = logobsA.error()
dda = logobsA.error_of_error()

def func(x):
    return numpy.log(x)

b4 = pyobs.errbias4(obsA, func)
print(f'Error log(obsA) {da}; 4th moment {b4}; ratio {da/b4}')
print(f'Error of error log(obsA) {dda}; 4th moment {b4}; ratio {dda/b4}')

assert b4 < dda
