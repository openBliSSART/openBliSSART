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


dnl AX_ENABLE_CPU
dnl Provides the --enable-cpu option.
dnl Use this macro before AC_PROG_CC and friends!

AC_DEFUN([AX_ENABLE_CPU],
[
    dnl Provide --enable-debug
    AC_ARG_ENABLE([cpu],
        [AS_HELP_STRING([--enable-cpu],
                        [sets compiler flags for debugging])],
        [ENABLE_CPU="$enableval"], [ENABLE_CPU="yes"]
    ) dnl AC_ARG_ENABLE

    if test x"$ENABLE_CPU" = x"yes"; then
        NO_CUDA="yes"
        CUDA_CPPFLAGS=""
        CUDA_LIBS=""
        AC_MSG_NOTICE([CUDA will not be used even it is found.])
    fi

    AC_SUBST(NO_CUDA)
    AC_SUBST(CUDA_CPPFLAGS)
    AC_SUBST(CUDA_LIBS)
]) dnl AC_DEFUN
