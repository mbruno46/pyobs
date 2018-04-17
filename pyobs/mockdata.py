#from copy import deepcopy

import numpy

# noise is an array with normally distributed data
# with mean zero and unit variance

def generate_mock_data(func, N, taus, strength, mixing=None):
	if not isinstance(func,(int,float)):
		nf = len(func)
	else:
		nf = 1
	nt = len(taus)
	noise = numpy.reshape( numpy.random.normal(0.0,1.0,N*nt), (N,nt) )
	apars = [ (2.*t-1.)/(2.*t+1.) for t in taus ]
	res = numpy.zeros( (N,nf) )
	nu_old = [0]*nt
	if (mixing==None):
		mixing = numpy.reshape(numpy.random.random(nt*nf), (nt,nf))
	ff = [numpy.sqrt(1-a*a) for a in apars]
	for i in range(N):
		nui = noise[i,:]  #numpy.array([noise[i] for _ in range(nt)])
		if (i>0):
			nui = nui*ff + apars * nu_old
		h = nui.dot(mixing) #nui * mixing
		res[i] = func + strength * h
		nu_old = numpy.array(nui) #deepcopy(nui)
	return res


def generate_gaussian_data(func, cov, N):
	if not isinstance(func,(int,float)):
		nf = len(func)
	else:
		nf = 1
	noise = numpy.reshape( numpy.random.normal(0.0,1.0,N*nf), (N,nf) )
	[evals, evecs] = numpy.linalg.eig(cov)
	Q = evecs.dot(numpy.diag(numpy.sqrt(evals)))
	res = numpy.zeros((N,nf))
	for i in range(N):
		res[i,:] = noise[i,:].dot(Q) + func
	return res
