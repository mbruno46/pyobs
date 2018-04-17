
import numpy

def derfit(xdata, ydata, W, pars, func, grad, hess):
	np = len(pars)
	n = len(ydata)
	E = numpy.array([func(pars,*xdata[i])-ydata[i] for i in range(n)])
	EW = E.dot(W)

	dfunc = numpy.zeros((n,np))
	for i in range(n):
		dfunc[i,:] = numpy.array(grad(pars,*xdata[i]))
	
	WG = W.dot(dfunc)
	Hmat = dfunc.transpose().dot(WG)

	ddfunc = numpy.zeros((np,np))
	for i in range(n):
		ddfunc = numpy.array(hess(pars,*xdata[i]))
		Hmat = Hmat + EW[i]*ddfunc
	Hinv = numpy.linalg.inv(Hmat)

	dpars = numpy.zeros((1,np)+(1,n))
	tmp = WG.dot(Hinv)
	for k in range(np):
		for i in range(n):
			dpars[0,k,0,i] = tmp[i,k]
	return dpars	
