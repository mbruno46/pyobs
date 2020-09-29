import pyobs
import numpy

# t0 = 0.02 fm
val=0.02
sig=val*0.01

N=1000
tau=4.0

data = pyobs.random.acrand(val,sig,tau,N)
t0 = pyobs.observable()
t0.create('EnsA',data,rname='r001')

# below we assume pion mass known to 1%
mpi = pyobs.observable()
mpi.create_from_cov('pion mass',134.9,0.1**2)
[m,dm] = mpi.error()
assert abs(dm[0]-0.1)<1e-12

print('t0 = ',t0)
print('mpi = ',mpi)

phi = t0 * (mpi/197.)**2
print('phi = ',phi)

[t,dt] = t0.error()
[m,dm] = (mpi/197.).error()
[p,dp] = phi.error()

assert dp - numpy.sqrt((dt*m**2)**2 + (2.0*t*m*dm)**2) < 1e-10

phi_copy = pyobs.observable(phi)
print(f'phi_copy = {phi}')

masses = pyobs.observable(desc='pion, kaon')
masses.create_from_cov('pion/kaon',[134.9766,497.648],[0.0006**2,0.022**2])
print('masses = ',masses)

phi.peek()
