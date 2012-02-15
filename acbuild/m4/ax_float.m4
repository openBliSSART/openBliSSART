dnl
dnl Copyright (c) 2007-2009, Alexander Lehmann <lehmanna@in.tum.de>
dnl                          Felix Weninger <felix@weninger.de>
dnl                          Bjoern Schuller <schuller@tum.de>
dnl
dnl Institute for Human-Machine Communication
dnl Technische Universitaet Muenchen (TUM), D-80333 Munich, Germany
dnl
dnl openBliSSART is free software: you can redistribute it and/or modify it under
dnl the terms of the GNU General Public License as published by the Free Software
dnl Foundation, either version 2 of the License, or (at your option) any later
dnl version.
dnl
dnl openBliSSART is distributed in the hope that it will be useful, but WITHOUT
dnl ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
dnl FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
dnl details.
dnl
dnl You should have received a copy of the GNU General Public License along with
dnl openBliSSART.  If not, see <http://www.gnu.org/licenses/>.
dnl


AC_DEFUN([AX_DETERMINE_FLOAT_TYPE],
[
    AC_ARG_WITH([single-prec],
        [AS_HELP_STRING([--with-single-prec],
                        [Use single precision in audio processing and NMF. Default is double precision.])],
        [AC_DEFINE([BLISSART_SINGLE_PREC], [1], [Define to 1 if you want to use single precision floats instead of doubles])], [NVCC_OPTIONS="$NVCC_OPTIONS -arch=sm_13"]
    )
    AC_SUBST(NVCC_OPTIONS)
    dnl AC_DEFINE_UNQUOTED([BLISSART_FLOAT_TYPE], [$FLOAT_TYPE], [The floating point type to use in all calculations.])
])

