import numpy
import matplotlib.pyplot as plt

def valerr(v,e):
    if e>0:
        dec = -int(numpy.floor( numpy.log10(e) ))
    else:
        dec = 1
    if (dec>0):
        digits = dec + 1
        outstr = "{0:.{1}f}".format(v, digits)
        outstr += "("
        outstr += "{:02.0f}".format(e*(10**digits), digits)
    elif (dec==0):
        outstr = "{0:.1f}".format(v, dec)
        outstr += "("
        outstr += "{:02.1f}".format(e, dec)
    else:
        outstr = "{0:.0f}".format(v, dec)
        outstr += "("
        digits = 0
        outstr += "{:.0f}".format(e, digits)
    outstr += ")"
    return outstr


def irregular_measurements(idx):
    if (len(idx)!=(idx[-1]-idx[0]+1)):
        return True
    return False

def fill_holes(idx_in, data_in, size=1):
    N = idx_in[-1] - idx_in[0] + 1
    idx = numpy.arange(idx_in[0],idx_in[-1]+1,1)
    data = numpy.array(numpy.zeros(N*size), dtype=numpy.double)
   
    norm = float(N)/float(len(data_in)/size)
    j=0
    for i in range(N):
        if (idx_in[j]==idx[i]):
            for k in range(size):
                data[i*size+k] = data_in[j*size+k] * norm
            j=j+1
    
    return [idx, data]

def pad_with_zeros(data, before, after):
    S = numpy.shape(data)
    n0 = S[-1]
    data2 = numpy.zeros( S[0:-1]+(before+n0+after,) )
    norm = float(before+n0+after)/float(n0)
    for i in range(before, before+n0):
        data2[:,:,i] = numpy.array(data[:,:,i-before])
    return data2 * norm

# returns the sorted union of the two lists
def union(a,b):
    u = sorted(set(a) | set(b))
    return list(u)

def double_union(a,b,an,bn):
    u = list(sorted(set(a) | set(b)))
    un = []
    for iu in u:
        try:
            un.append( an[a.index(iu)] )
        except:
            un.append( bn[b.index(iu)] )
    return [u, un]

def contains(a,b):
    for i in b:
        if (a==b):
            return b.index(a)
    return -1


def piechart(x,l,t):
    plt.subplots()
    plt.title('Components (%d,%d)' % t)    
    x2 = numpy.array(x)/numpy.sum(x)
    plt.pie(x2,labels=l, autopct='%.0f%%',radius=1.0)
    plt.axis('equal')
    plt.show()
