import pyobs
import numpy

T = 32
L = 16
m = 0.2
p = 0.0

corr_ex = [pyobs.qft.free_scalar.Cphiphi(t,m,p,T,L) for t in range(T//2)]
cov_ex = pyobs.qft.free_scalar.cov_Cphiphi(m,p,T,L)[0:T//2,0:T//2]

tau = 1.0
N = 2000
data = pyobs.random.acrandn(corr_ex, cov_ex, tau, N)

corr1 = pyobs.obs()
corr1.create('EnsA',data.flatten(),shape=(T//2,))

func = pyobs.qft.free_scalar.Cphiphi_string('t','m',p,T,L)
[f,df,_] = pyobs.symbolic.diff(func,'t','m')

[c, dc] = corr1.error()

xax = numpy.arange(T//2)
W = 1./dc**2
fit1 = pyobs.mfit(xax,W,f,df,v='t')

print(fit1(corr1))

tau = 1.0
N = 1000
data = pyobs.random.acrandn(corr_ex, cov_ex, tau, N)

corr2 = pyobs.obs()
corr2.create('EnsA',data.flatten(),shape=(T//2,))

[c, dc] = corr2.error()
W = 1./dc**2
fit2 = pyobs.mfit(xax,W,f,df,v='t')

fit3 = fit1 + fit2
print(fit3([corr1,corr2]))