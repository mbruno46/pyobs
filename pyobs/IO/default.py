import pyobs
import bison

class observable_decoder(bison.Decoder):
    def __init__(self):
        super().__init__('pyobs.core.ndobs.observable')
    
    def decode(self, obj):
        out = pyobs.observable(description=obj['description'])
        out.set_mean(obj['mean'])
        assert (out.shape == obj['shape'])
        assert (out.size == obj['size'])
        out.ename = obj['ename']
        out.delta = obj['delta']
        out.cdata = obj['cdata']
        return out

class delta_decoder(bison.Decoder):
    def __init__(self):
        super().__init__('pyobs.core.data.delta')
    
    def decode(self, obj):
        out = pyobs.core.data.delta(obj['mask'],obj['idx'],lat=obj['lat'])
        out.delta[:,:] = obj['delta'][:,:]
        return out

class cdata_decoder(bison.Decoder):
    def __init__(self):
        super().__init__('pyobs.core.cdata.cdata')
    
    def decode(self, obj):
        out = pyobs.core.cdata.cdata(obj['cov'])
        return out

def save(fname, *args):
    bison.save(fname, *args)
    
def load(fname):
    return bison.load(fname, decoder=[observable_decoder, delta_decoder, cdata_decoder])
