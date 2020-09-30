import json
import gzip
import re

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
    with gzip.open(fname, 'wt') as f:
        tofile = json.dumps(obs, indent=2, default=__encoder__ )
        f.write( tofile )

def load(fname):
    tmp = json.loads(gzip.open(fname, 'r').read())
    res = pyobs.observable(description=tmp['description'])
    res.www = list(tmp['www'])

    res.mean = numpy.array(tmp['mean'])
    res.shape = tuple(tmp['shape'])
    res.size=numpy.prod(res.shape)
    res.edata = list(tmp['edata'])
    
    for key in tmp['rdata']:
        if (type(tmp['rdata'][key]['idx']) is str):
            regex=re.compile('[(,)]')
            h = regex.split(tmp['rdata'][key]['idx'])
            if h[0]!='range': # pragma: no cover
                raise pyobs.PyobsError('Unexpected idx')
            res.rdata[key] = pyobs.core.data.rdata(tmp['rdata'][key]['mask'],range(int(h[1]),int(h[2]),int(h[3])))
        else:
            res.rdata[key] = pyobs.core.data.rdata(tmp['rdata'][key]['mask'],tmp['rdata'][key]['idx'])
        res.rdata[key].delta = numpy.array(tmp['rdata'][key]['delta'])
        
    res.mfname = list(tmp['mfname'])
    for key in tmp['mfdata']:
        if (type(tmp['mfdata'][key]['idx']) is str):
            regex=re.compile('[(,)]')
            h = regex.split(tmp['mfdata'][key]['idx'])
            if h[0]!='range': # pragma: no cover
                raise pyobs.PyobsError('Unexpected idx')
            res.mfdata[key] = pyobs.core.data.mfdata(tmp['mfdata'][key]['mask'],
                                                     range(int(h[1]),int(h[2]),int(h[3])),tmp['mfdata'][key]['lat'])
        else:
            res.mfdata[key] = pyobs.core.data.mfdata(tmp['mfdata'][key]['mask'],
                                                     tmp['mfdata'][key]['idx'],tmp['mfdata'][key]['lat'])
        res.mfdata[key].delta = numpy.array(tmp['mfdata'][key]['delta'])
        
    for key in tmp['cdata']:
        res.cdata[key] = pyobs.core.cdata.cdata(tmp['cdata'][key]['grad'],tmp['cdata'][key]['cov'])
    pyobs.memory.update(res)
    return res
