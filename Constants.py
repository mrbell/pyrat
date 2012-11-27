"""
Constants.py

**********************************************************************************

Copyright 2012 Michael Bell

This file is part of pyrat, the Python Radio Astronomy Toolkit.

pyrat is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pyrat is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pyrat.  If not, see <http://www.gnu.org/licenses/>.

**********************************************************************************

Module containing useful physical constants and unit conversions.

"""

PI = 3.14159265359
ii = complex(0,1)
C = 299792458. # speed of light, in m/s
C2 = C**2. # speec of light squared


ARCSEC_TO_RAD = 4.84813681e-6
RAD_TO_DEG = 180./PI
DEG_TO_RAD = PI/180.