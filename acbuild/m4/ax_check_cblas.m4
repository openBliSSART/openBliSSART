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


dnl AX_CHECK_CBLAS([ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl Test for the dnl availability of CBLAS and set HAVE_CBLAS_H and CBLAS_LIBS
dnl accordingly.
AC_DEFUN([AX_CHECK_CBLAS],
[
    dnl Check if the mandatory CBLAS header file is available.
    AC_CHECK_HEADERS([cblas.h], , [NO_CBLAS=yes])
    dnl Do NOT replace this   ^^^   empty space with a colon : or square
    dnl brackts [] because otherwise autoconf interprets action-if-found as
    dnl given!!!

    dnl In case that the CBLAS header isn't available, see if maybe Apple's
    dnl vecLib framework is accessible instead.
    if test -n "$NO_CBLAS"; then
        save_LIBS="$LIBS"
        LIBS="-framework vecLib $LIBS"
        AC_CHECK_FUNC(cblas_dgemm, [CBLAS_LIBS="-framework vecLib"])
        LIBS="$save_LIBS"
        if test -n "$CBLAS_LIBS"; then
            dnl The framework and hence cblas.h is accessible, thus unset
            dnl NO_CBLAS.
            unset NO_CBLAS
        fi
    fi

    dnl If CBLAS_LIBS hasn't already been set, try setting CBLAS_LIBS to some
    dnl other meaningful value.
    if test -z "$CBLAS_LIBS"; then
        CBLAS_LIBS="-lcblas -lf77blas -latlas"
    fi

    dnl See if everything works as expected iff NO_CBLAS is unset and
    dnl CBLAS_LIBS is non-empty.
    AC_MSG_CHECKING([for cblas availability])
    if test -z "$NO_CBLAS" -a -n "$CBLAS_LIBS"; then
        save_LIBS="$LIBS"
        LIBS="$CBLABS_LIBS $LIBS"
        AC_LANG_PUSH([C])
        AC_RUN_IFELSE(
            AC_LANG_PROGRAM(
                [#include <cblas.h>],
                [double t = 1.0; return (cblas_ddot(5, &t, 0, &t, 0) == 5.0);])
            :,
            [NO_CBLAS=yes])
        AC_LANG_POP([C]) 
        LIBS="$save_LIBS"
    fi

    dnl The result solely depends on $NO_CBLAS.
    if test -z "$NO_CBLAS"; then
        AC_MSG_RESULT([yes])
        AC_SUBST(CBLAS_LIBS)
        dnl Commented this out because HAVE_CBLAS_H is set by AC_CHECK_HEADER
        dnl and serves just as well.
        dnl AC_DEFINE([HAVE_CBLAS], [1], [Define to 1 if you have CBLAS.])
        ifelse([$1], [], :, [$1])
    else
        AC_MSG_RESULT([no])
        ifelse([$2], [], :, [$2])
    fi
]) dnl AC_DEFUN
