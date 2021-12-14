import numpy
import pyobs

__all__ = ["grid"]


class grid:
    def __init__(self, dims, bcs):
        self.dimensions = numpy.array(dims, dtype=numpy.int32)
        self.boundaries = bcs
        self.ndims = len(dims)
        _s = []
        for mu in range(self.ndims):
            if bcs[mu]=='periodic':
                _s += [dims[mu]]
            elif bcs[mu]=='open':
                _s += [dims[mu]*2]
        self.fft_shape = tuple(_s)
        self.vol = numpy.prod(dims)
        
    # here data is assumed to be a scalar field over the dimensions
    def expand_data(self, data, idx):
        if type(idx) is range:
            return numpy.reshape(numpy.array(data, dtype=numpy.float64), tuple(self.dimensions))
        tmp = numpy.zeros((self.vol,), dtype=numpy.float64)
        for j in range(len(idx)):
            tmp[idx[j] - idx[0]] = data[j]
        return numpy.reshape(tmp, tuple(self.dimensions))
    
    
    def rrmax(self):
        return int(numpy.sum((self.dimensions / 2) ** 2) + 1)
    
    def create_fft_data(self, data, idx, fft_ax):
        tmp = self.expand_data(data, idx)
        # in-place, even if it adds one element at the end
        return numpy.fft.rfftn(tmp, s=self.fft_shape, axes=fft_ax)
    
    
    def convolution(self, data, idx, xmax, a=0, b=None):
        fft_ax = tuple(range(self.ndims))

        aux = []
        for index in [a, b]:
            if index is None:
                continue
            aux += [self.create_fft_data(data[index, :], idx, fft_ax)]

        if len(aux) == 1:
            aux[0] *= aux[0].conj()
            aux += [numpy.fft.irfftn(aux[0], s=self.fft_shape, axes=fft_ax)]
        else:
            aux[0] *= aux[1].conj()
            aux[1] = numpy.fft.irfftn(aux[0].real, s=self.fft_shape, axes=fft_ax)

#         aux[1] = numpy.reshape(aux[1], (self.vol,))
        
        return pyobs.core.mftools.intrsq(aux[1].flatten(), numpy.array(self.dimensions, dtype=numpy.int32), xmax)

    
    def solid_angle(self, xmax):
        return pyobs.core.mftools.intrsq(numpy.ones(self.vol), self.dimensions, xmax)
    
    
    def block(self, block_size):
        bs = numpy.array(block_size)
        pyobs.assertion(len(bs) == len(self.dimensions), "Block size does match inner data")
        pyobs.assertion(numpy.sum(self.dimensions % bs) == 0,"Block size does not divide inner data")
        return grid(self.dimensions // bs, self.boundaries)
        
    # TODO: remove blocks that are zeros
    def block_data(self, data, idx, block_size):
        if isinstance(block_size, (int, numpy.int)):
            block_size = numpy.array([block_size], dtype=numpy.int32)

        dat = expand_data(data, idx, tuple(self.dimensions))
        return pyobs.core.mftools.blockdata(dat, self.dimensions, block_size)

    


