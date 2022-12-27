dest='pyobs/tensor/unary.py'

ops = {
    "log": ['Natural logarithm', 'logarithm', 'numpy.log', 'numpy.reciprocal'],
    "exp": ['exponential', 'exponential', 'numpy.exp', 'numpy.exp'],
    "sin": ['sine', 'sine', 'numpy.sin', 'numpy.cos'],
    "arcsin": ['Inverse sine', 'arcsine', 'numpy.arcsin', 'lambda x: (1-x*x)**(-0.5)'],
    "cos": ['cosine', 'cosine', 'numpy.cos', 'lambda x: -numpy.sin(x)'],
    "arccos": ['Inverse sine', 'arcsine', 'numpy.arcsin', 'lambda x: -(1-x*x)**(-0.5)'],
    "tan": ['tangent', 'tangent', 'numpy.tan', 'lambda x: 1 / numpy.cos(x) ** 2'],
    "arctan": ['inverser tangent', 'arctangent', 'numpy.arctan', 'lambda x: 1 / (1 + x*x)'],
    "cosh": ['Hyperbolic cosine', 'hyperbolic cosine', 'numpy.cosh', 'numpy.sinh'],
    "arccosh": ['inverse Hyperbolic cosine', 'inverse hyperbolic cosine', 'numpy.arccosh', 'lambda x: (x*x - 1)**(-0.5)'],
    "sinh": ['Hyperbolic sine', 'hyperbolic sine', 'numpy.sinh', 'numpy.cosh'],
    "arcsinh": ['inverse Hyperbolic sine', 'inverse hyperbolic sine', 'numpy.arcsinh', 'lambda x: (x*x + 1)**(-0.5)'],
}


def make_py(key, l):
    return f'''
def {key}(x):
    """
    Return the {l[0]} element-wise.

    Parameters:
       x (obs): input observable

    Returns:
       obs : the {l[1]} of the input observable, element-wise.

    Examples:
       >>> y = pyobs.{key}(x)
    """
    return __unary(x, {l[2]}, {l[3]})

'''

with open(dest,'w') as f:
    f.write("""#################################################################################
#
# unary.py: definitions of unary operations
# Copyright (C) 2020-2023 Mattia Bruno
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
import pyobs
import numpy

def __unary(x, f, df):
    new_mean = f(x.mean)
    aux = df(x.mean)
    g = pyobs.gradient(lambda xx: xx * aux, x.mean, gtype="diag")
    return pyobs.derobs([x], new_mean, [g])

""")
            
    f.write('\n\n')
            
    all = '__all__ = ['
    for key in ops:
        all += f'"{key}",'
    all += '\n]\n\n'
    f.write(all)
            
    for key in ops:
        f.write(make_py(key, ops[key]))