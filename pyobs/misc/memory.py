#################################################################################
#
# memory.py: routines for the monitoring of the memory used by the library
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

import sys, os, numpy

book = {}
MB=1024.**2
GB=1024.**2

def get_size(obj):
    size=sys.getsizeof(obj)
    if isinstance(obj,dict):
        size += sum([get_size(v) for v in obj.values()])
        size += sum([get_size(k) for k in obj.keys()])
    elif isinstance(obj,numpy.ndarray):
        size += obj.nbytes
    elif hasattr(obj,'__dict__'):
        size += get_size(obj.__dict__)
    elif hasattr(obj,'__iter__') and not isinstance(obj,(str,bytes,bytearray)):
        size += sum([get_size(i) for i in obj])
    return size

def add(obj):
    book[id(obj)] = get_size(obj)
    
def update(obj):
    if id(obj) in book:
        book[id(obj)] = get_size(obj)
    
def rm(obj):
    del book[id(obj)]

def get(obj):
    size=book[id(obj)]
    if size>MB:
        return f'{size/MB:.0f} MB'
    else:
        return f'{size/1024.:.0f} KB'
    
def info():
    tot_size=0
    print('pyobs allocated memory:')
    n=1
    for k in book.keys():
        size = book[k]
        tot_size += size
        print(f' {n}) observable with size {size/MB:g} MB')
        n+=1
        
    if tot_size>MB:
        print(f' - TOTAL {tot_size/MB:.0f} MB\n')
    else:
        print(f' - TOTAL {tot_size/1024.:.0f} KB\n')

def available():
    platform = sys.platform
    if platform == "linux" or platform == "linux2":
        print('bla')
    elif platform == "darwin":
        out = os.popen('vm_stat').readlines()
        size = int(out[0].split('page size of')[1].split('bytes')[0].strip())
        pages = float(out[1].split(':')[1].strip())
        bb = pages * size
    elif platform == "win32":
        raise 
    
    if bb>GB:
        print(f' - Available memory {bb/GB:.0f} GB\n')
    else:
        print(f' - Available memory {bb/MB:.0f} MB\n')
    
    return bb