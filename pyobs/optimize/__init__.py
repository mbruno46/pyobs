#################################################################################
#
# __init__.py: methods and functionalities of the library accessible to users
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

from .rootscalar import root_scalar
from .chisquare import mfit
from . import symbolic
from .interpolation import interpolate

__all__ = ['symbolic']
__all__.extend(chisquare.__all__)
__all__.extend(rootscalar.__all__)
__all__.extend(interpolation.__all__)
