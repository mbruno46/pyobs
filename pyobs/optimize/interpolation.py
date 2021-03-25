import numpy
import pyobs

__all__ = ['interpolate']

def interpolate(x,y):
    """
    
    """
    N = len(x)
    if (len(y.shape)>1):
        raise pyobs.PyobsError(f'Unexpected observable with shape ${x.shape}; only vectors are supported')
    if (y.size!=N):
        raise pyobs.PyobsError(f'Unexpected observable with shape ${x.shape} not matching size of x')
    Minv = numpy.linalg.inv(numpy.array([[x[i]**k for k in range(N)] for i in range(N)]).astype('f8'))
    mean = Minv @ y.mean
    g = pyobs.gradient( lambda x: Minv @ x, y.mean)
    return pyobs.derobs([y],mean,[g])