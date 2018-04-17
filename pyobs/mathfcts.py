import numpy
import pickle
import os.path

from sympy import IndexedBase, Idx, symbols
from sympy import Matrix, diff, lambdify

__all__ = ['math_sum','math_det','math_tr',
		'math_inv','math_add','math_mul',
		'math_dot','math_scalar','math_binary']


def SymMat(dims,symb):
	i = Idx('i', dims[0])
	j = Idx('j', dims[1])
	A = IndexedBase(symb)
	H = []
	for i in range(dims[0]):
		H.append( [A[i,j] for j in range(dims[1])] )
	return Matrix(H)

def Subs(mean,x):
    subs = []
    for a in range(len(x)):
        dims = numpy.shape(mean[a])
        for i in range(dims[0]):
            for j in range(dims[1]):
                subs.append( (x[a][i,j], mean[a][i,j]) )
    return subs

def math_generic(expr, x, xlabel, mean):
    new_dims = expr.shape
    old_dims = x[0].shape
    
    func = numpy.zeros(new_dims)
    grad = []
    for a in range(len(x)):
        old_dims = x[a].shape
        grad.append( numpy.zeros(new_dims+old_dims) )

	s = Subs(mean, x)
    for i in range(new_dims[0]):
        for j in range(new_dims[1]):
            func[i,j] = expr[i,j].subs(s)
            
            for a in range(len(x)):
                old_dims = x[a].shape
            
                for k in range(old_dims[0]):
                    for l in range(old_dims[1]):
						dfunc = lambdify(symbols(xlabel), diff(expr[i,j], x[a][k,l]), "numpy")
						grad[a][i,j,k,l] = dfunc(*mean)
    
    return [func, grad]

#######################################################################

def math_sum(mean,axis):
	dims = numpy.shape(mean)
	M = SymMat(dims,'M')
	s = Subs([mean],[M])
	new_dims = (1,)*(axis==0) + (dims[0],)*(axis!=0)
	new_dims = new_dims + (1,)*(axis==1) + (dims[1],)*(axis!=1)
	expr = Matrix(numpy.zeros(new_dims))
	for i in range(dims[0]):
		for j in range(dims[1]):
			if (axis==0):
				expr[0,j] = expr[0,j] + M[i,j]
			elif (axis==1):
				expr[i,0] = expr[i,0] + M[i,j]
	return math_generic(expr, [M], 'M', [mean])

def math_det(mean):
    dims = numpy.shape(mean)
    M = SymMat(dims,'M')
    s = Subs([mean],[M])
    expr = Matrix( [[M.det()]] )
    return math_generic(expr, [M], 'M', [mean])

def math_tr(mean):
    dims = numpy.shape(mean)
    M = SymMat(dims,'M')
    s = Subs([mean],[M])
    expr = Matrix( [[M.trace()]] )
    return math_generic(expr, [M], 'M', [mean])

def math_inv(mean):
	dims = numpy.shape(mean)
	func = numpy.linalg.inv(mean)

	grad = numpy.zeros(dims+dims)
	dfunc = numpy.zeros(dims)

	for k in range(dims[0]):
		for l in range(dims[1]):
			dfunc[k,l] = 1.0
			tmp = - func.dot(dfunc).dot(func)

			for i in range(dims[0]):
				for j in range(dims[1]):
					grad[i,j,k,l] = tmp[i,j]

	return [func, [grad]]

#######################################################################

def math_add(mean1,mean2):
	d1 = numpy.shape(mean1)
	d2 = numpy.shape(mean2)
	if (d1!=d2):
		if (d1!=(1,1) and d2!=(1,1)):
			raise InputError('operands incompatible for sum ' + d1 + ' , ' + d2)

	M1 = SymMat(d1,'M1')
	M2 = SymMat(d2,'M2')

	expr = M1+M2
	return math_generic(expr, [M1, M2], 'M1 M2', [mean1, mean2])

def math_mul(mean1,mean2):
	d1 = numpy.shape(mean1)
	d2 = numpy.shape(mean2)
	if (d1!=d2):
		if (d1!=(1,1) and d2!=(1,1)):
			raise InputError('operands incompatible for sum ' + d1 + ' , ' + d2)

	M1 = SymMat(d1,'M1')
	M2 = SymMat(d2,'M2')

	expr = M1*M2
	return math_generic(expr, [M1, M2], 'M1 M2', [mean1, mean2])

def math_dot(mean1,mean2):
	d1 = numpy.shape(mean1)
	d2 = numpy.shape(mean2)

	M1 = SymMat(d1,'M1')
	M2 = SymMat(d2,'M2')

	s = Subs([mean1,mean2],[M1, M2])
	h = [[0]*d2[1]]*d1[0]
	for i in range(d1[0]):
		for j in range(d2[1]):
			for k in range(d1[1]):
				h[i][j] = h[i][j] + M1[i,k]*M2[k,j]
	expr = Matrix(h)
	#expr = Matrix(M2.dot(M1)).reshape(d1[0],d2[1]).transpose()

	return math_generic(expr, [M1, M2], 'M1 M2', [mean1, mean2])


#########################################################

def math_scalar(x,i,a=None):
	dx=numpy.shape(x)
	if (i==0):
		return [numpy.log(x), numpy.reciprocal(x)]
	elif (i==1):
		return [numpy.exp(x), numpy.exp(x)]
	elif (i==2):
		return [numpy.power(x,a), a*numpy.power(x,a-1)]
	elif (i==10):
		return [numpy.sin(x), numpy.cos(x)]
	elif (i==11):
		return [numpy.cos(x), -numpy.sin(x)]
	elif (i==12):
		one = numpy.ones(dx)
		return [numpy.arcsin(x), numpy.reciprocal(numpy.sqrt(one-numpy.power(x,2)))]
	elif (i==13):
		one = numpy.ones(dx)
		return [numpy.arccos(x), -numpy.reciprocal(numpy.sqrt(one-numpy.power(x,2)))]
	elif (i==20):
		return [numpy.sinh(x), numpy.cosh(x)]
	elif (i==21):
		return [numpy.cosh(x), numpy.sinh(x)]
	elif (i==22):
		one = numpy.ones(dx)
		return [numpy.arcsinh(x), numpy.reciprocal(numpy.sqrt(one+numpy.power(x,2)))]
	elif (i==23):
		one = numpy.ones(dx)
		return [numpy.arccosh(x), numpy.reciprocal(numpy.sqrt(numpy.power(x,2)-one))]
	elif (i==30):
		return [numpy.reciprocal(x), -numpy.reciprocal(numpy.power(x,2))]
	elif (i==31):
		return [x+a, numpy.ones(dx)]
	elif (i==32):
		return [x-a, numpy.ones(dx)]
	elif (i==33):
		return [a-x, -numpy.ones(dx)]
	elif (i==34):
		return [numpy.multiply(x,a), a]
	elif (i==35):
		return [-x, -numpy.ones(dx)]

# assumes x and y same dimensions
def math_binary(x,y,i):
	d=numpy.shape(x)
	if (i==1):
		return [x+y, [numpy.ones(d), numpy.ones(d)]]
	elif (i==2):
		return [x-y, [numpy.ones(d), -numpy.ones(d)]]
	elif (i==3):
		return [y-x, [-numpy.ones(d), numpy.ones(d)]]
	elif (i==4):
		return [numpy.multiply(x,y), [y,x]]

