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


AC_DEFUN([AX_CHECK_CUDA],
[
    dnl Provide --with-cuda
    AC_ARG_WITH([cuda],
        [AS_HELP_STRING([--with-cuda],
                        [Use CUDA and CUBLAS libraries for GPU linear algebra operations in NMF.])],
        [CUDA=yes], [CUDA=no]
    ) dnl AC_ARG_WITH
    dnl Provide --with-cuda-lib=<dir> and --with-cuda-include=<dir>.
    AC_ARG_WITH([cuda-lib],
        [AS_HELP_STRING([--with-cuda-lib=<dir>],
                        [Path where CUDA BLAS libraries are installed (default: /usr/local/cuda/lib64)])],
        [CUDA_LIB="$withval"], [CUDA_LIB="/usr/local/cuda/lib64"]
    ) dnl AC_ARG_WITH
    AC_ARG_WITH([cuda-include],
        [AS_HELP_STRING([--with-cuda-include=<dir>],
                        [Path where CUDA BLAS includes are installed (default: /usr/local/cuda/include)])],
        [CUDA_INCLUDE="$withval"], [CUDA_INCLUDE="/usr/local/cuda/include"]
    ) dnl AC_ARG_WITH

    AC_MSG_CHECKING([whether to use CUDA])
    if test "$CUDA" == "yes"; then
        AC_MSG_RESULT([yes])
    else
        AC_MSG_RESULT([no])
    fi

    if test "$CUDA" == "yes"; then
        AC_MSG_CHECKING([for CUDA])
        save_LIBS="$LIBS"
        save_LDFLAGS="$LDFLAGS"
        save_CPPFLAGS="$CPPFLAGS"
        LIBS="$LIBS -lcuda -lcublas"
        CPPFLAGS="$CPPFLAGS -I$CUDA_INCLUDE"
        LDFLAGS="$LDFLAGS -L$CUDA_LIB -Wl,-rpath,$CUDA_LIB"
        dnl LDFLAGS="$LDFLAGS -L$CUDA_LIB"
        AC_LANG_PUSH([C])
        AC_RUN_IFELSE(
            AC_LANG_PROGRAM(
                [#include <cublas.h>],
                [cublasHandle_t handle; return 0;]), :, [NO_CUDA=yes])
        AC_LANG_POP([C])
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"
        LIBS="$save_LIBS"
        if test -z "$NO_CUDA"; then
            AC_MSG_RESULT([yes])
        else
            AC_MSG_RESULT([no])
        fi
    else
        NO_CUDA=yes
    fi dnl whether to check for CUDA

    if test -z "$NO_CUDA"; then
        dnl adapt LDFLAGS globally, because LibLinAlg and thus CUDA currently 
        dnl has to be linked in all libraries and executables.
        dnl Probably hack, but works for me (fw).
        AC_DEFINE([HAVE_CUDA], [1], [Define to 1 if you have CUDA installed and want to use it for extra-fast NMF.])
        LDFLAGS="$LDFLAGS -L$CUDA_LIB -Wl,-rpath,$CUDA_LIB"
        LIBS="$LIBS -lcuda -lcublas"
        CUDA_CPPFLAGS="-I$CUDA_INCLUDE"
        ifelse([$1], [], :, [$1])     
    else
        ifelse([$2], [], :, [$2])
    fi

    AC_SUBST(CUDA_CPPFLAGS)
    AC_SUBST(CUDA_LIBS)
])
