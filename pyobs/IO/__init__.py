from . import json, default
import pyobs
import os
import datetime

__all__ = ['save','load']

def save(fname, obs):
    """
    Save the observable to disk. 

    Parameters:
       name (str): string with the destination (path+filename)
       obs (observable): observable to save

    Notes:
       Available output formats:

       * pyobs: the default binary format, with automatic checksums
         for file corruptions and fast read/write speed.

       * json.gz: apart from the compression with gunzip, the file 
         is a plain text file generated with json format, for easy 
         human readability and compatibility with other programming 
         languages (json format is widely supported).

    Examples:
       >>> obsA = pyobs.observable('obsA')
       >>> obsA.create('A',data)
       >>> pyobs.save('~/analysis/obsA.json.gz', obsA)
    """    
    if os.path.isfile(fname) is True:
        raise pyobs.PyobsError(f'File {fname} already exists')
    
    if 'pyobs' in fname:
        fmt = default
    elif 'json' in fname:
        fmt = json
    else:
        raise pyobs.PyobsError(f'Format not supported')

    obs.www[2] = datetime.datetime.now().strftime('%c')
    fmt.save(fname,obs)

    
def load(fname):
    """
    Load the observable from disk.

    Parameters:
       name (str): string with the source file (path+filename)

    Returns:
       observable: the loaded observable

    Examples:
       >>> obsA = pyobs.load('~/analysis/obsA.json.gz')
    """
    
    if not os.path.isfile(fname):
        raise pyobs.PyobsError(f'File {fname} does not exists')
    
    if 'pyobs' in fname:
        fmt = default 
    elif 'json' in fname:
        fmt = json
    else: # pragma: no cover
        raise pyobs.PyobsError(f'Format not supported')
        
    return fmt.load(fname)
