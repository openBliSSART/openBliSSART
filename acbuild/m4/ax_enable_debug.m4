dnl
dnl $Id: ax_enable_debug.m4 858 2009-06-10 08:24:44Z alex $
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


dnl AX_ENABLE_DEBUG
dnl Provides the --enable-debug option. If debug is enabled, CFLAGS
dnl and CXXFLAGS are set accordingly.
dnl Sets the DEBUG_BUILD conditional.
dnl Use this macro before AC_PROG_CC and friends!
AC_DEFUN([AX_ENABLE_DEBUG],
[
    dnl Provide --enable-debug
    AC_ARG_ENABLE([debug],
        [AS_HELP_STRING([--enable-debug],
                        [sets compiler flags for debugging])],
        [ENABLE_DEBUG="$enableval"], [ENABLE_DEBUG="no"]
    ) dnl AC_ARG_ENABLE

    if test "$ENABLE_DEBUG" != "yes"; then
        dnl Never ever use -ftree-vectorize since it can lead to strange bugs!
        CFLAGS="-O3 $CFLAGS"
        CXXFLAGS="-O3 $CPPFLAGS"
    else
        CFLAGS="-g -O2 -Wall -D_DEBUG $CFLAGS"
        CXXFLAGS="-g -O2 -Wall -D_DEBUG $CPPFLAGS"
        AC_MSG_NOTICE([Compiler flags set for debugging.])
    fi

    AC_SUBST(CFLAGS)
    AC_SUBST(CPPFLAGS)

    AM_CONDITIONAL(DEBUG_BUILD, [test "$ENABLE_DEBUG" == "yes"])
]) dnl AC_DEFUN
