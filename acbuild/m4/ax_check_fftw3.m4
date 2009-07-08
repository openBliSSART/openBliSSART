dnl
dnl $Id: ax_check_fftw3.m4 858 2009-06-10 08:24:44Z alex $
dnl
dnl This file is part of openBliSSART.
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


dnl AX_CHECK_FFTW3([ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
AC_DEFUN([AX_CHECK_FFTW3],
[
    AC_CHECK_HEADER([fftw3.h], [], [NO_FFTW3=yes])

    if test -z "$NO_FFTW3"; then
        AC_CHECK_LIB([fftw3], [fftw_plan_dft_r2c_1d], :, [NO_FFTW3=yes])
    fi

    dnl The result depends on $NO_FFTW3.
    if test -z "$NO_FFTW3"; then
        ifelse([$1], [], :, [$1])
    else
        ifelse([$2], [], :, [$2])
    fi
]) dnl AC_DEFUN
