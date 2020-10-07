#################################################################################
#
# utils.py: generic utility routines
# Copyright (C) 2020 Mattia Bruno
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
#################################################################################

import numpy
import sys

__all__ = ['PyobsError','check_type','check_not_type',
        'is_verbose','set_verbose','valerr','textable']

verbose=[]

class PyobsError(Exception):
    pass

def is_verbose(func):
    if func in verbose:
        return True
    return False

def set_verbose(func):
    if func not in verbose:
        verbose.append(func)

def valerr(v,e):
    def core(v,e):
        if e>0:
            dec = -int(numpy.floor( numpy.log10(e) ))
        else:
            dec = 1
        if (dec>0):
            outstr = f'{v:.{dec+1}f}({e*10**(dec+1):02.0f})'
        elif (dec==0):
            outstr = f'{v:.{dec+1}f}({e:02.{dec+1}f})'
        else:
            outstr = f'{v:.0f}({e:.0f})'
        return outstr
    
    if numpy.ndim(v)==0:
        return core(v,e)
    elif numpy.ndim(v)==1:
        return ' '.join([core(v[i],e[i]) for i in range(len(v))])
    elif numpy.ndim(v)==2:
        return '\n'.join([valerr(v[i,:],e[i,:]) for i in range(numpy.shape(v)[0])])
    else:
        raise PyobsError('valerr supports up to 2D arrays')

def textable(mat,cols=None,fmt=None):
    """
    Formats a matrix to a tex table.
    
    Parameters:
       mat (array): 2D array
       cols (list, optional): an index from 0 to M, with M smaller
           than the number of columns `mat`. It specifies which columns
           go together in a value-error combination.
       fmt (list, optional): a list of formats for each columns; it is 
           ignored on the columns corresponding to a value-error combination.
           Accepted values are 'd' for integers, '.2f' for floats with 2 digit
           precision, etc ...
           
    Returns:
       str: the tex table
       
    Examples:
       >>> tsl = [0,1,2,3,4] # time-slice
       >>> [c, dc] = correlator.error()
       >>> mat = numpy.c_[tsl, c, dc] # pack data
       >>> pyobs.textable(mat, cols=[0,1,1], fmt=['d',0,0])
    """
    
    if numpy.ndim(mat)!=2:
        raise PyobsError('textable supports only 2D arrays')
    
    (n,m) = numpy.shape(mat)
    if cols is None:
        cols = range(m)
    if fmt is None:
        fmt = ['.2f']*m
        
    outstr = ''
    for a in range(n):
        (idx,rep) = numpy.unique(cols, return_counts=True)
        h= []
        for i in range(len(idx)):
            if rep[i]==1:
                h += [f'%{fmt[i]}' % mat[a,idx[i]]]
            elif rep[i]==2:
                h += [valerr(mat[a,idx[i]],mat[a,idx[i]+1])]
        outstr = f'{outstr}{" & ".join(h)}\n'
    
    return outstr
        
def check_type(obj,s,*t):
    c=0
    for tt in t:
        if not type(obj) is tt:
            c+=1
    if c==len(t):
        raise TypeError(f'Unexpected type for {s} [{t}]')

def check_not_type(obj,s,t):
    if type(obj) is t:
        raise TypeError(f'Unexpected type for {s}')
    
#def sort_data(idx,data):
#    out = numpy.zeros(numpy.shape(data))
#    new_idx = list(numpy.sort(idx))
#    for i in range(len(new_idx)):
#        j=idx.index(new_idx[i])
#        out[i,:] = data[j,:]
#    return [new_idx, out]
