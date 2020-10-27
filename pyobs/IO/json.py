import json
import gzip
import re

import time
import pyobs
import numpy

def __encoder__(obj):
    if isinstance(obj,numpy.integer):
        return int(obj)
    elif isinstance(obj,numpy.ndarray):
        return obj.tolist() #json.dumps(obj.tolist())
    elif isinstance(obj,range):
        return f'range({obj.start},{obj.stop},{obj.step})'
    return obj.__dict__

def save(fname, obs):
    dt = -time.time()
    with gzip.open(fname, 'wt') as f:
        json.dump(obs, f, indent=2, default=__encoder__ )
    dt += time.time()
    b = pyobs.memory.book[id(obs)]
    print(f'Written {b/1024.**2:g} MB at {b/dt/1024.**2:g} MB/s')
    
def load(fname):
    tmp = json.loads(gzip.open(fname, 'r').read())
    res = pyobs.observable(description=tmp['description'])
    res.www = list(tmp['www'])

    res.mean = numpy.array(tmp['mean'])
    res.shape = tuple(tmp['shape'])
    res.size=numpy.prod(res.shape)
    res.ename = list(tmp['ename'])
    
    for key in tmp['delta']:
        if (type(tmp['delta'][key]['idx']) is str):
            regex=re.compile('[(,)]')
            h = regex.split(tmp['delta'][key]['idx'])
            if h[0]!='range': # pragma: no cover
                raise pyobs.PyobsError('Unexpected idx')
            res.delta[key] = pyobs.core.data.delta(tmp['delta'][key]['mask'],range(int(h[1]),int(h[2]),int(h[3])),lat=tmp['delta'][key]['lat'])
        else:
            res.delta[key] = pyobs.core.data.delta(tmp['delta'][key]['mask'],tmp['delta'][key]['idx'],lat=tmp['delta'][key]['lat'])
        res.delta[key].delta = numpy.array(tmp['delta'][key]['delta'])
                
    for key in tmp['cdata']:
        res.cdata[key] = pyobs.core.cdata.cdata(tmp['cdata'][key]['cov'])
    pyobs.memory.update(res)
    return res
