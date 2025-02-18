#################################################################################
#
# __init__.py: methods and functionalities of the library accessible to users
# Copyright (C) 2025 Mattia Bruno
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


class empty_plt:
    def figure(*args, **kwargs):
        pass

    def plot(*args, **kwargs):
        pass

    def pie(*args, **kwargs):
        pass

    def hist(*args, **kwargs):
        pass

    def show(*args, **kwargs):
        pass

    def fill_between(*args, **kwargs):
        pass

    def xlabel(*args, **kwargs):
        pass

    def ylabel(*args, **kwargs):
        pass

    def title(*args, **kwargs):
        pass

    def xlim(*args, **kwargs):
        pass

    def ylim(*args, **kwargs):
        pass

    def legend(*args, **kwargs):
        pass


try:
    import matplotlib.pyplot as plt

    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False

plt = plt if MATPLOTLIB else empty_plt
