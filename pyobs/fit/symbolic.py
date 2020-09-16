#################################################################################
#
# symbolic.py: utility functions for symbolic differentiation
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

import sympy
from sympy.parsing.sympy_parser import parse_expr
sympy.init_printing()

from pyobs.core.utils import check_type, is_verbose

try:
    from IPython.display import display
except:
    def display(x):
        sympy.pprint(x)

def diff(f,x,dx):
    """
    Utility function to compute the gradient and hessian of a function using symbolic calculus
    
    Parameters:
       f (string): the reference function
       x (string): variables that are not differentiated; different variables
           must be separated by a comma
       dx (string): variables that are differentiated; different variables
           must be separated by a comma
    
    Returns:
       lambda : (scalar) function `f`
       lambda : (vector) function of the gradient of `f`
       lambda : (matrix) function of the hessian of `f`
    
    Notes: 
       The symbolic manipulation is based on the library `sympy` and the user 
       must follow the `sympy` syntax when passing the argument `f`. The analytic
       form of the gradient and hessian can be printed by activating the 'diff'
       verbose flag.
       
    Examples:
       >>> res = diff('a+b*x','x','a,b') # differentiate wrt to a and b
       a + b*x
       [1, x]
       [[0, 0], [0, 0]]
       >>> for i in range(3):
       >>>     print(res[i](0.4,1.,2.))
       1.8
       [1, 0.4]
       [[0, 0], [0, 0]]
    """
    check_type(f,'f',str)
    check_type(x,'x',str)
    check_type(dx,'dx',str)
    
    sym = {}
    for y in dx.rsplit(','):
        sym[y] = sympy.Symbol(y)
        
    expr = parse_expr(f, local_dict=sym)    
    allvars=f'{x},{dx}'
    func = sympy.lambdify(allvars, expr, 'numpy')        
    dexpr = []
    ddexpr=[]
    for y in sym:
        dexpr.append(sympy.diff(expr, sym[y]))
        tmp = []
        for z in sym:
            tmp.append(sympy.diff(dexpr[-1], sym[z]))
        ddexpr.append(tmp)
    
    if is_verbose('diff') or is_verbose('symbolic.diff'):
        display(expr)
        display(dexpr)
        display(ddexpr)
    
    dfunc = sympy.lambdify(allvars, dexpr, 'numpy')
    ddfunc = sympy.lambdify(allvars, ddexpr, 'numpy')
    return [func, dfunc, ddfunc]
