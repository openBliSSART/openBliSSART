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


dnl AX_CHECK_SDL(MINIMUM-VERSION, [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl Test for SDL, and define SDL_CFLAGS and SDL_LIBS.
AC_DEFUN([AX_CHECK_MATLAB],
[
    dnl Provide --with-matlab
    AC_ARG_WITH([matlab],
        [AS_HELP_STRING([--with-matlab],
                        [Use MATLAB BLAS libraries. Otherwise ATLAS is used, if available.])],
        [MATLAB=yes], [MATLAB=no]
    ) dnl AC_ARG_WITH
    dnl Provide --with-matlab-lib=<dir> and --with-matlab-include=<dir>.
    AC_ARG_WITH([matlab-lib],
        [AS_HELP_STRING([--with-matlab-lib=<dir>],
                        [Path where MATLAB BLAS libraries are installed (optional)])],
        [MATLAB_LIB="$withval"], [MATLAB_LIB=""]
    ) dnl AC_ARG_WITH
    AC_ARG_WITH([matlab-include],
        [AS_HELP_STRING([--with-matlab-include=<dir>],
                        [Path where MATLAB BLAS includes are installed (optional)])],
        [MATLAB_INCLUDE="$withval"], [MATLAB_INCLUDE=""]
    ) dnl AC_ARG_WITH

    AC_MSG_CHECKING([whether to use Matlab libraries])
    if test "$MATLAB" == "yes"; then
        AC_MSG_RESULT([yes])
    else
        AC_MSG_RESULT([no])
    fi

    if test "$MATLAB" == "yes"; then
        AC_MSG_CHECKING([for Matlab])
        if test -n "$MATLAB_INCLUDE" -a -n "$MATLAB_LIB"; then
            save_LIBS="$LIBS"
            save_LDFLAGS="$LDFLAGS"
            save_CPPFLAGS="$CPPFLAGS"
            LIBS="$LIBS -lmwblas"
            CPPFLAGS="$CPPFLAGS -I$MATLAB_INCLUDE"
            LDFLAGS="$LDFLAGS -L$MATLAB_LIB -Wl,-rpath,$MATLAB_LIB"
            AC_LANG_PUSH([C])
            AC_RUN_IFELSE(
                AC_LANG_PROGRAM(
                    [#include <blas.h>],
                    [double t = 1.0; int n = 5; int inc = 0; return (ddot(&n, &t, &inc, &t, &inc) != 5.0);]),
                :,
                [NO_MATLAB=yes])
            AC_LANG_POP([C])
            CPPFLAGS="$save_CPPFLAGS"
            LDFLAGS="$save_LDFLAGS"
            LIBS="$save_LIBS"
            if test -z "$NO_MATLAB"; then
                AC_MSG_RESULT([yes])
            else
                AC_MSG_RESULT([no])
            fi
        else dnl include and lib path not given
            AC_MSG_RESULT([no])
            NO_MATLAB=yes
        fi dnl include and lib path given?
    else
        NO_MATLAB=yes
    fi dnl whether to check for Matlab

    if test -z "$NO_MATLAB"; then
        dnl adapt LDFLAGS globally, because LibLinAlg and thus Matlab currently 
        dnl has to be linked in all libraries and executables.
        dnl Probably hack, but works for me (fw).
        AC_DEFINE([HAVE_MATLAB], [1], [Define to 1 if you have Matlab and want to use it instead of ATLAS.])
        LDFLAGS="$LDFLAGS -L$MATLAB_LIB -Wl,-rpath,$MATLAB_LIB"
        LIBS="$LIBS -lmwblas"
        MATLAB_CPPFLAGS="-I$MATLAB_INCLUDE"
        ifelse([$1], [], :, [$1])     
    else
        ifelse([$2], [], :, [$2])
    fi

    AC_SUBST(MATLAB_CPPFLAGS)
    AC_SUBST(MATLAB_LIBS)
])
