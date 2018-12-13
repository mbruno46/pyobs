
import numpy

def jackknife_error(mean,delta,ncnfg,dims,bsize):
    N = numpy.sum(ncnfg)
    var = numpy.zeros(dims)

    # discards automatically last configs if bsize is not multiple of ncnfg
    nbins = int(numpy.floor(ncnfg/bsize))
    norm = 1.0/(float)(ncnfg-bsize)
    for ib in range(nbins):
        binj = numpy.zeros(dims)
        for ic in range(ib*bsize, (ib+1)*bsize):
            binj[:,:] += delta[:,:,ic]
        #var += numpy.power(mean-binj*norm,2)
        var += numpy.power(binj*norm,2)

    fact = (float)(nbins-1.0)/(float)(nbins)
    return var*fact
