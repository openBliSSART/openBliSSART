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


dnl AX_GET_OS_AND_MACHINE_TYPE
dnl Determine the OS and machine type and set TYPE_OS and TYPE_MACHINE accordingly
AC_DEFUN([AX_GET_OS_AND_MACHINE_TYPE],
[
    AC_PATH_PROG(UNAME, uname, no)

    AC_MSG_CHECKING(os and machine type)
    if test "x$UNAME" != "x"; then
        TYPE_OS=`$UNAME -o`
        TYPE_MACHINE=`$UNAME -m`
    else
        TYPE_OS="unknown"
        TYPE_MACHINE="unknown"
    fi
    AC_MSG_RESULT($TYPE_OS $TYPE_MACHINE)

    AC_SUBST(TYPE_OS)
    AC_SUBST(TYPE_MACHINE)
])
