#################################################################################
#
# free_scalar.py: correlators in the free scalar theory
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

def aomega(m,p):
    return numpy.arccosh(1. + 0.5*(m**2 + p**2))

def Gt(t,m,p,L):
    phat = 2*numpy.sin(numpy.array(p)*numpy.pi/numpy.array(L))
    om = numpy.arccosh(1. + 0.5*(m**2 + numpy.sum(phat**2)))
    return numpy.exp(-numpy.abs(t)*om)*0.5/numpy.sinh(om)

def Cphiphi(t,m,p,T,L,N=1):
    c = numpy.sum([Gt(t+n*T,m,p,L)+Gt(t-n*T,m,p,L) for n in range(1,N+1)])
    return Gt(t,m,p,L)+c

def Cphiphi_string(t,m,p,T,L):
    omega=f'acosh(0.5*{m}**2 + 2*sin({p*numpy.pi/L})**2+1)'
    den=f'2*sinh({omega})'
    num=f'exp(-{omega}*abs({t})) + exp(-{omega}*abs({T}-{t})) + exp(-{omega}*abs({T}+{t}))'
    return f'({num})/({den})'

def cov_Cphiphi(m,p,T,L,N=1):
    c = numpy.zeros((T,T))
    for i in range(T):
        for j in range(i,T):
            c[i,j] += numpy.sum([Cphiphi(i+z0,m,p,T,L,N)*Cphiphi(z0-j,m,p,T,L,N) for z0 in range(T)])/(float)(T)
            if p==0.:
                c[i,j] += numpy.sum([Cphiphi(i-j+z0,m,p,T,L,N)*Cphiphi(z0,m,p,T,L,N) for z0 in range(T)])/(float)(T)
            if i!=j:
                c[j,i] = c[i,j]
    return c
