import pyobs
import numpy

pyobs.set_verbose('create')

val=0.5
sig=val*0.01

N=1000
tau=0.0

data = pyobs.random.acrand(val,sig,tau,N)
obsA = pyobs.observable()
try:
    obsA.create(12,data)
except TypeError:
    print('catched error')
try:
    obsA.create('Ens:::A',data)
except pyobs.PyobsError:
    print('catched error')
    
obsA.create('EnsA',data,rname='r001')
obsA.peek()
print(obsA)

obsB = pyobs.observable(obsA)

data2 = pyobs.random.acrand(val,sig,tau,N)
obsC = pyobs.observable()
obsC.create('EnsA',[data[0::2],data2],icnfg=[range(0,N,2),range(100,100+N)],rname=['r001','r002'])
obsC.peek()

obsD = pyobs.observable()
obsD.create('EnsA',[data,data2])

print('obsC = ',obsC)

try:
    from IPython.display import display
    display(obsC)
except:
    pass

obsB.create('EnsA',data2,icnfg=range(N))
obsB.peek()
print('Before adding sys. err = ',obsB)
[_,db] = obsB.error()
se=0.003
obsB.add_syst_err('syst.err #1',[se])
print(f'After adding sys. err ({db[0]:g})({se:g}) = ({numpy.sqrt(db[0]**2+se**2):g}) : {obsB}')
