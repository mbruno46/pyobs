import json
import sys
import re
import zlib

little = sys.byteorder == 'little'

import time
import pyobs
import numpy

def __encoder__(obj):
    if isinstance(obj,numpy.integer):
        return int(obj)
    elif isinstance(obj,numpy.ndarray):
        if little:
            return f'{zlib.crc32(obj.tobytes("C")):8X}'
        else:
            tmp = numpy.ascontiguousarray(array, dtype='<f8')
            return f'{zlib.crc32(tmp.tobytes("C")):8X}'
    elif isinstance(obj,range):
        return f'range({obj.start},{obj.stop},{obj.step})'
    return obj.__dict__

def __safe_write(array, f):
    if little:
        f.write(array.tobytes("C"))
    else:
        tmp = numpy.ascontiguousarray(array, dtype='<f8')
        f.write(tmp.tobytes("C"))

def __safe_read(size, crc32, f):
    bb = f.read(size*8)
    _crc32 = f'{zlib.crc32(bb):8X}'
    if _crc32!=crc32:
        raise pyobs.PyobsError(f'Checksum failed')
    return numpy.frombuffer(bb, dtype='<f8')
    
def save(fname, obs):
    if not isinstance(obs,pyobs.observable):
        raise pyobs.PyobsError('Unexpected argument type, expected observable')
    
    tofile = json.dumps(obs, indent=2, default=__encoder__ )
    
    dt = -time.time()
    with open(fname, 'wb') as f:
        n = len(tofile)
        f.write(n.to_bytes(4, 'little'))
        f.write(str.encode(tofile))
        
        __safe_write(obs.mean, f)
        n += obs.mean.nbytes

        for key in obs.delta:
            __safe_write(obs.delta[key].delta, f)
            n += obs.delta[key].delta.nbytes
        
        for key in obs.cdata:
            __safe_write(obs.cdata[key].cov, f)
            n += obs.cdata[key].cov.nbytes
            
    dt += time.time()
    if pyobs.is_verbose('save'):
        print(f'Saved {n/1024.**2:g} MB at {n/1024.**2/dt:g} MB/s')

def load(fname):
    regex=re.compile('[(,)]')
    
    with open(fname, 'rb') as f:
        n = int.from_bytes(f.read(4), 'little')
        header = json.loads(f.read(n).decode('utf-8'))
        
        res = pyobs.observable(description=header['description'])
        res.www = list(header['www'])
        
        res.shape = tuple(header['shape'])
        res.size=numpy.prod(res.shape)
        res.mean = __safe_read(res.size, header['mean'], f).reshape(res.shape)
        
        res.ename = list(header['ename'])
        
        for key in header['delta']:
            d = header['delta'][key]
            if type(d['idx']) is str:
                h = regex.split(d['idx'])
                if h[0]=='range':
                    res.delta[key] = pyobs.core.data.delta(d['mask'],range(int(h[1]),int(h[2]),int(h[3])), lat=d['lat'])
                else:
                    raise pyobs.PyobsError('Load error; format not understood')
            else:
                res.delta[key] = pyobs.core.data.delta(d['mask'], d['idx'], lat=d['lat'])
            
            sh = (res.delta[key].size,res.delta[key].n)
            res.delta[key].delta = __safe_read(numpy.prod(sh), d['delta'], f).reshape(sh)
        
        for key in header['cdata']:
            n = res.size
            cov = __safe_read(n*n, header['cdata'][key]['cov'], f).reshape(n,n)
            res.cdata[key] = pyobs.core.cdata.cdata(numpy.eye(n), cov)
    return res
