import pyobs
import bison

class observable_decoder:
    def __init__(self):
        self.type = 'pyobs.core.ndobs.observable'
    
    def decode(self, obj):
        out = pyobs.observable(description=obj['description'])
        out.set_mean(obj['mean'])
        assert (out.shape == obj['shape'])
        assert (out.size == obj['size'])
        out.ename = obj['ename']
        out.delta = obj['delta']
        out.cdata = obj['cdata']
        pyobs.memory.update(out)
        return out

class delta_decoder:
    def __init__(self):
        self.type = 'pyobs.core.data.delta'
    
    def decode(self, obj):
        out = pyobs.core.data.delta(obj['mask'],obj['idx'],lat=obj['lat'])
        out.delta[:,:] = obj['delta'][:,:]
        return out

class cdata_decoder:
    def __init__(self):
        self.type = 'pyobs.core.cdata.cdata'
    
    def decode(self, obj):
        out = pyobs.core.cdata.cdata(obj['cov'])
        return out

def save(fname, *args):
    bison.save(fname, *args)
    
def load(fname):
    return bison.load(fname, decoder=[observable_decoder, delta_decoder, cdata_decoder])
